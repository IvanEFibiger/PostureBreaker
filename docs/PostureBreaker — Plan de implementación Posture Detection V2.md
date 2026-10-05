# PostureBreaker — Plan de implementación Posture Detection V2

## 1. Objetivo

Evolucionar el sistema actual de detección postural desde un conjunto reducido de desplazamientos 2D hacia un modelo más robusto capaz de:

- distinguir orientación de cabeza y torso;
- detectar rotación cervical real respecto del tronco;
- medir inclinación lateral de cabeza;
- medir flexión de cabeza;
- medir elevación y asimetría de hombros;
- mejorar la detección de cabeza adelantada;
- mejorar la detección de torso inclinado;
- soportar múltiples posiciones habituales de trabajo;
- evitar falsos positivos al cambiar de monitor;
- utilizar métricas opcionales según disponibilidad real de landmarks;
- incorporar confianza y cobertura;
- permitir diferentes políticas temporales según el problema postural;
- preparar el sistema para integrar Face Landmarker si fuera necesario.

La idea central es pasar de:

```text
4 desplazamientos 2D
        ↓
cantidad de métricas malas
        ↓
BUENA / MALA
```

a:

```text
MediaPipe
   │
   ├── landmarks 2D
   └── world landmarks 3D
            │
            ▼
    Geometría corporal
            │
    ┌───────┴────────┐
    │                │
 Cabeza            Tronco
 yaw/pitch/roll    yaw/lean/roll
    │                │
    └───────┬────────┘
            ▼
 Relaciones ergonómicas
            │
   ┌────────┼─────────┐
   ▼        ▼         ▼
 cuello   hombros    torso
            │
            ▼
       calibración
            │
            ▼
    severidad / riesgo
            │
            ▼
      PostureEngine
```

---

# 2. Principio de diseño

El detector no debe asumir que una geometría distinta significa automáticamente una mala postura.

Primero debe determinar:

```text
cómo está orientado el usuario
```

y después evaluar:

```text
cómo están relacionadas entre sí
cabeza
cuello
hombros
torso
```

Esto es especialmente importante para una configuración con múltiples monitores.

Ejemplo:

```text
Monitor lateral:
cabeza girada 35°
torso girado 5°
```

no es equivalente a:

```text
Monitor lateral:
cabeza girada 35°
torso girado 30°
```

En ambos casos la cabeza está orientada hacia el mismo monitor.

Pero la rotación cervical relativa es completamente diferente.

Por eso una métrica fundamental de V2 será:

```text
neck_yaw_delta
```

---

# 3. Fase 0 — Métricas nuevas en modo observación

Antes de que cualquier métrica nueva afecte:

```text
score
alertas
postura buena/mala
analytics productivos
```

debe ejecutarse en modo observación.

Durante esta fase:

```text
las métricas se calculan
se muestran en debug
se pueden registrar
pero NO generan alertas
```

Esto permitirá validar con condiciones reales:

```text
monitor 1
monitor 2
frontal
cabeza adelantada
mirar hacia abajo
cuello girado
cuello + torso girados
hombros elevados
cabeza inclinada lateralmente
torso inclinado
```

Objetivo:

> estabilizar primero la medición y después decidir la política.

---

# 4. Fase 1 — Separar landmarks de métricas

Actualmente `extract_metrics()` realiza varias responsabilidades:

```text
recibe resultado MediaPipe
elige lado
obtiene landmarks
valida visibilidad
calcula métricas
```

Con V2 conviene separar estas tareas.

Crear:

```text
posture_guard/
    geometry.py
```

o, si se quiere avanzar hacia una organización más clara:

```text
posture_guard/
    vision/
        landmarks.py
        geometry.py
```

No hace falta migrar todo el proyecto de una vez.

---

# 5. Modelo `Point2D`

Agregar:

```python
@dataclass
class Point2D:
    x: float
    y: float
    visibility: float
```

---

# 6. Modelo `Point3D`

Agregar:

```python
@dataclass
class Point3D:
    x: float
    y: float
    z: float
    visibility: float
```

---

# 7. Modelo `BodyLandmarks`

Crear:

```python
@dataclass
class BodyLandmarks:
    image: dict[str, Point2D]
    world: dict[str, Point3D]
```

Inicialmente extraer:

```text
nose

left_eye_inner
left_eye
left_eye_outer

right_eye_inner
right_eye
right_eye_outer

left_ear
right_ear

left_mouth
right_mouth

left_shoulder
right_shoulder

left_elbow
right_elbow

left_wrist
right_wrist

left_hip
right_hip
```

No implica utilizar todos inmediatamente.

El objetivo es disponer de una capa uniforme de landmarks.

---

# 8. Utilidades geométricas

Crear funciones reutilizables.

Ejemplo:

```python
def distance_2d(a: Point2D, b: Point2D) -> float:
    ...
```

```python
def distance_3d(a: Point3D, b: Point3D) -> float:
    ...
```

```python
def angle_degrees(dx: float, dy: float) -> float:
    ...
```

```python
def midpoint(a: Point2D, b: Point2D) -> Point2D:
    ...
```

```python
def visibility_confidence(*points: Point2D) -> float:
    ...
```

Esto evita repetir geometría dentro de `detection.py`.

---

# 9. Fase 2 — Confianza de métrica

El modelo actual:

```python
values: dict[str, float]
```

empieza a quedarse corto.

Una métrica puede estar disponible pero ser poco confiable.

Agregar:

```python
@dataclass
class DetectionMetrics:
    side: str
    values: dict[str, float]
    confidence: dict[str, float]
    points: dict[str, tuple[float, float]]
```

Ejemplo:

```python
metrics.values["head_yaw"] = 0.42
metrics.confidence["head_yaw"] = 0.91
```

Otro caso:

```python
metrics.values["shoulder_roll"] = 3.8
metrics.confidence["shoulder_roll"] = 0.48
```

La segunda debería tener menos influencia.

---

# 10. Configuración de confianza

Agregar:

```python
metric_min_confidence: float = 0.60
```

a `Config`.

También validar:

```text
0 <= metric_min_confidence <= 1
```

---

# 11. Fase 3 — Cobertura durante calibración

Actualmente una métrica puede aparecer solo en una pequeña cantidad de frames y aun así incorporarse al perfil.

Eso será problemático con métricas opcionales.

Agregar:

```python
calibration_min_metric_coverage: float = 0.75
```

Ejemplo:

```text
calibración = 90 frames

métrica presente en 85 frames
coverage = 94 %
→ válida

métrica presente en 20 frames
coverage = 22 %
→ excluir
```

---

# 12. Cambios en `Calibrator`

Actualmente:

```python
present = [
    sample[name]
    for sample in side_samples
    if name in sample
]
```

Agregar:

```python
coverage = len(present) / len(side_samples)

if coverage < config.calibration_min_metric_coverage:
    continue
```

El perfil debe almacenar solo métricas suficientemente observables.

---

# 13. Métricas de calidad de calibración

Ampliar el concepto actual de:

```text
quality_score
```

para considerar:

```text
coverage
stability
separability
confidence
```

Por métrica.

Posible estructura futura:

```python
@dataclass
class MetricCalibrationQuality:
    coverage: float
    stability: float
    separability: float
    confidence: float
```

No es obligatorio persistir todo desde la primera PR.

---

# 14. Fase 4 — RollingMetrics ponderado por confianza

Actualmente `RollingMetrics` utiliza media simple.

Con V2 usar:

```python
weighted_mean =
    sum(value * confidence)
    /
    sum(confidence)
```

Además exigir un mínimo de observaciones.

Nueva config:

```python
smoothing_min_observations: int = 6
```

Ejemplo:

```text
window = 12
minimum observations = 6
```

Si una métrica aparece solo una vez dentro de la ventana:

```text
no se publica todavía
```

---

# 15. Fase 5 — `ViewState`

Agregar:

```python
@dataclass
class ViewState:
    head_yaw: float | None
    head_pitch: float | None
    head_roll: float | None

    torso_yaw: float | None
    torso_pitch: float | None
    torso_roll: float | None

    orientation: str
    confidence: float
```

---

# 16. Orientación

Inicialmente:

```text
left
center
right
unknown
```

La categoría es útil para UI y debugging.

Pero las decisiones internas deben basarse principalmente en:

```text
head_yaw
torso_yaw
```

continuos.

---

# 17. Actualizar `DetectionMetrics`

Quedaría aproximadamente:

```python
@dataclass
class DetectionMetrics:
    side: str
    values: dict[str, float]
    confidence: dict[str, float]
    points: dict[str, tuple[float, float]]
    view: ViewState
```

---

# 18. Fase 6 — Head yaw

Objetivo:

```text
detectar giro izquierda/derecha de cabeza
```

No se debe presentar inicialmente como grados cervicales exactos.

Será una señal geométrica calibrada.

---

# 19. Head yaw mediante ojos y nariz

Calcular:

```python
eye_mid_x = (left_eye.x + right_eye.x) / 2
eye_span = abs(right_eye.x - left_eye.x)

head_yaw_eye = (
    nose.x - eye_mid_x
) / max(eye_span, EPSILON)
```

Frontal:

```text
nose.x ≈ eye_mid_x
→ ~0
```

Al girar:

```text
la nariz cambia su posición relativa
```

---

# 20. Head yaw mediante orejas

Otra señal:

```python
left_distance = abs(nose.x - left_ear.x)
right_distance = abs(right_ear.x - nose.x)

head_yaw_ear = (
    left_distance - right_distance
) / max(
    left_distance + right_distance,
    EPSILON
)
```

---

# 21. Fusionar señales de yaw

Si existen ambas:

```python
head_yaw =
    head_yaw_eye * eye_weight +
    head_yaw_ear * ear_weight
```

Los pesos pueden depender de confianza.

Ejemplo:

```python
head_yaw = weighted_average([
    (head_yaw_eye, eye_confidence),
    (head_yaw_ear, ear_confidence),
])
```

---

# 22. Tests de head yaw

Crear casos sintéticos:

```text
frontal
giro izquierdo
giro derecho
una oreja no visible
un ojo poco visible
ambas señales disponibles
información insuficiente
```

Validar:

```text
signo correcto
estabilidad
confianza
fallback correcto
```

---

# 23. Fase 7 — Head roll

Usar ambos ojos.

```python
head_roll = math.degrees(
    math.atan2(
        right_eye.y - left_eye.y,
        right_eye.x - left_eye.x,
    )
)
```

Interpretación:

```text
ojos horizontales
→ aproximadamente 0°

cabeza inclinada
→ valor positivo/negativo
```

---

# 24. Tests de head roll

Casos:

```text
ojos horizontales
ojo derecho más bajo
ojo izquierdo más bajo
distancia ocular muy pequeña
baja visibilidad
```

---

# 25. Fase 8 — Head pitch proxy

Con Pose Landmarker no tenemos mentón real.

Por lo tanto inicialmente medir:

```text
head_pitch_proxy
```

y no un ángulo anatómico exacto.

Usar:

```python
eye_mid_y = (left_eye.y + right_eye.y) / 2
eye_span = distance_2d(left_eye, right_eye)

head_pitch = (
    nose.y - eye_mid_y
) / max(eye_span, EPSILON)
```

La calibración se encargará del baseline personal.

---

# 26. Mantener `chin_drop` temporalmente

No eliminar todavía:

```text
chin_drop
```

Durante V2 debe permanecer para comparación.

Internamente puede marcarse como legacy.

Por ejemplo:

```text
chin_drop_legacy
```

pero no es imprescindible renombrarlo en la primera migración.

---

# 27. Fase 9 — Torso yaw mediante world landmarks

Usar:

```text
left_shoulder
right_shoulder
```

en coordenadas 3D.

Ejemplo:

```python
dx = right_shoulder.x - left_shoulder.x
dz = right_shoulder.z - left_shoulder.z

torso_yaw = math.degrees(
    math.atan2(dz, dx)
)
```

Debe validarse con datos reales antes de tratarlo como grado anatómico absoluto.

---

# 28. Fallback de torso yaw

Si `world_landmarks` no están disponibles o resultan inestables:

```text
usar proxy 2D
```

pero con menor confianza.

La API debería esconder esta decisión.

Ejemplo:

```python
value, confidence = estimate_torso_yaw(body)
```

---

# 29. Fase 10 — `neck_yaw_delta`

Métrica central:

```python
neck_yaw_delta =
    head_yaw - torso_yaw
```

Ejemplo:

### Solo cuello

```text
head_yaw  = 35
torso_yaw = 4

neck_yaw_delta = 31
```

### Cuerpo acompañando el giro

```text
head_yaw  = 35
torso_yaw = 28

neck_yaw_delta = 7
```

Esta métrica es mucho más útil ergonómicamente que `head_yaw` por sí sola.

---

# 30. Confianza de `neck_yaw_delta`

Debe depender de ambas señales.

Ejemplo:

```python
confidence =
    min(
        head_yaw_confidence,
        torso_yaw_confidence,
    )
```

O media armónica.

Nunca debe tener más confianza que sus componentes.

---

# 31. Fase 11 — Shoulder roll

Medir inclinación de la línea de hombros.

```python
shoulder_roll = math.degrees(
    math.atan2(
        right_shoulder.y - left_shoulder.y,
        right_shoulder.x - left_shoulder.x,
    )
)
```

UI:

```text
Hombros desnivelados
```

---

# 32. Fase 12 — `neck_roll_delta`

No interesa tanto la inclinación absoluta de la cabeza respecto de la cámara.

Interesa:

```python
neck_roll_delta =
    head_roll - shoulder_roll
```

Ejemplo:

```text
cámara/persona inclinada:

head roll      5°
shoulder roll  5°
delta          0°
```

vs.

```text
cabeza inclinada:

head roll      12°
shoulder roll   2°
delta          10°
```

---

# 33. Fase 13 — Shoulder elevation

Usar:

```text
ear
shoulder
hip
```

Calcular longitud de torso:

```python
torso_length =
    distance_2d(shoulder, hip)
```

Calcular:

```python
ear_shoulder_distance =
    distance_2d(ear, shoulder)
```

Entonces:

```python
shoulder_elevation_ratio =
    ear_shoulder_distance
    /
    max(torso_length, EPSILON)
```

---

# 34. Elevación izquierda/derecha

Si hay información suficiente:

```text
left_shoulder_elevation
right_shoulder_elevation
```

Además una métrica agregada:

```text
shoulder_elevation
```

que puede tomar:

```text
lado más confiable
```

o:

```text
lado con mayor severidad
```

según la política futura.

---

# 35. No hardcodear dirección

Aunque normalmente un hombro elevado reduce distancia oreja-hombro:

```text
la calibración debe aprender la dirección
```

Esto mantiene coherencia con el sistema actual:

```text
good_mean
bad_mean
direction
threshold
```

---

# 36. Fase 14 — Shoulder asymmetry

Si ambos hombros tienen confianza suficiente:

```python
shoulder_level_delta =
    left_shoulder.y - right_shoulder.y
```

Pero para clasificación preferir el ángulo:

```text
shoulder_roll
```

`shoulder_level_delta` puede mantenerse para debug.

---

# 37. Fase 15 — Torso forward angle

Reemplazar progresivamente:

```text
torso_lean_dx
```

por un ángulo.

Vista lateral:

```python
dx = shoulder.x - hip.x
dy = hip.y - shoulder.y

torso_forward_angle = math.degrees(
    math.atan2(dx, dy)
)
```

Interpretación:

```text
torso vertical
→ cerca de baseline

torso hacia delante
→ aumenta/disminuye según lado
```

La calibración resuelve dirección.

---

# 38. Fase 16 — Head forward ratio

Reemplazar progresivamente:

```text
ear_shoulder_dx
```

por:

```python
head_forward_ratio =
    horizontal_offset(ear, shoulder)
    /
    torso_length
```

Esto reduce sensibilidad a:

```text
distancia de cámara
resolución
crop
escala corporal
```

---

# 39. Disponibilidad según orientación

`head_forward_ratio` no debería publicarse siempre.

Por ejemplo:

```text
vista lateral o semi-lateral
→ disponible

vista frontal
→ unavailable
```

Lo mismo aplica a otras métricas.

Principio:

> una métrica ausente es mejor que una métrica falsa.

---

# 40. Fase 17 — Clasificación basada en métricas disponibles

Actualmente se calcula:

```python
enabled_metrics = sum(
    1
    for threshold in profile.thresholds.values()
    if threshold.mode != "disabled"
)
```

Eso debe cambiar.

Con V2 puede haber:

```text
8 métricas habilitadas en perfil

pero solo 4 observables en este frame
```

La clasificación debe considerar:

```text
habilitada
+
presente
+
confianza suficiente
```

---

# 41. Nuevo conjunto de métricas clasificables

Ejemplo:

```python
available_enabled = [
    name
    for name, value in metrics.values.items()
    if name in profile.thresholds
    and profile.thresholds[name].mode != "disabled"
    and metrics.confidence.get(name, 0.0)
        >= config.metric_min_confidence
]
```

`effective_min` debe calcularse sobre este conjunto.

---

# 42. Problema del conteo simple

Con pocas métricas:

```text
2 malas
→ mala postura
```

es razonable.

Con 10 métricas no escala bien.

Por eso V2 debe preparar una transición a riesgo ponderado.

---

# 43. Fase 18 — Peso por métrica

Extender:

```python
@dataclass
class MetricThreshold:
    threshold: float
    mode: str
    direction: int | None
    margin: float
    weight: float = 1.0
```

Pesos iniciales tentativos:

| Métrica | Peso |
|---|---:|
| `neck_yaw_delta` | 1.2 |
| `head_forward_ratio` | 1.5 |
| `head_pitch` | 1.0 |
| `neck_roll_delta` | 0.8 |
| `shoulder_elevation` | 1.1 |
| `shoulder_roll` | 0.7 |
| `torso_forward_angle` | 1.2 |

No activarlos todavía en la primera implementación.

---

# 44. Risk score

Futuro:

```python
risk = sum(
    severity[name]
    * weight[name]
    * confidence[name]
    for name in active_metrics
)
```

Esto permite que:

```text
una anomalía fuerte
```

pueda pesar más que:

```text
dos anomalías mínimas
```

---

# 45. Fase 19 — `PostureIssue`

Separar métrica técnica de problema visible al usuario.

Crear:

```python
class PostureIssue(StrEnum):
    HEAD_FORWARD = "head_forward"
    NECK_ROTATION = "neck_rotation"
    NECK_FLEXION = "neck_flexion"
    HEAD_TILT = "head_tilt"
    SHOULDER_ELEVATION = "shoulder_elevation"
    SHOULDER_ASYMMETRY = "shoulder_asymmetry"
    TORSO_FORWARD = "torso_forward"
```

---

# 46. Mapping métrica → issue

Ejemplo:

```text
HEAD_FORWARD
    ← head_forward_ratio

NECK_ROTATION
    ← head_yaw
    ← torso_yaw
    ← neck_yaw_delta

NECK_FLEXION
    ← head_pitch

HEAD_TILT
    ← head_roll
    ← shoulder_roll
    ← neck_roll_delta

SHOULDER_ELEVATION
    ← shoulder_elevation

SHOULDER_ASYMMETRY
    ← shoulder_roll

TORSO_FORWARD
    ← torso_forward_angle
```

---

# 47. Ventaja

La UI no muestra:

```text
neck_yaw_delta
```

sino:

```text
Cuello girado
```

No muestra:

```text
shoulder_roll
```

sino:

```text
Hombros desnivelados
```

---

# 48. Fase 20 — Multi-monitor mediante `ViewProfile`

No crear perfiles rígidos:

```text
LEFT
CENTER
RIGHT
```

Crear perfiles de posición habitual.

Ejemplo:

```text
Monitor 1
Monitor 2
```

---

# 49. Modelo `ViewProfile`

```python
@dataclass
class ViewProfile:
    id: str
    name: str

    head_yaw_mean: float
    torso_yaw_mean: float

    head_yaw_std: float
    torso_yaw_std: float

    calibration: CalibrationProfile
```

---

# 50. Diferencia entre vista y postura

Esto es fundamental.

`ViewProfile` sirve para entender:

```text
qué geometría de cámara/monitor es normal
```

pero no para declarar que cualquier postura dentro de esa vista es correcta.

Ejemplo:

```text
Monitor 1 baseline:

head yaw 35°
torso yaw 5°
```

Eso no significa:

```text
30° de giro cervical durante horas = perfecto
```

El sistema debe seguir evaluando:

```text
neck_yaw_delta
```

como problema postural.

---

# 51. Fase 21 — Selección automática de vista

Comparar orientación actual con perfiles.

Ejemplo:

```python
distance =
    abs(
        current.head_yaw -
        profile.head_yaw_mean
    ) * 0.7
    +
    abs(
        current.torso_yaw -
        profile.torso_yaw_mean
    ) * 0.3
```

Seleccionar menor distancia.

---

# 52. Hysteresis

Evitar cambios rápidos:

```text
Monitor 1
Monitor 2
Monitor 1
Monitor 2
```

cuando el usuario se encuentra entre ambos.

Exigir que un perfil gane durante:

```text
500–1000 ms
```

antes de cambiar.

Config:

```python
view_switch_stability_seconds: float = 0.75
```

---

# 53. Fase 22 — `CalibrationSet`

Actualmente existe un único:

```text
CalibrationProfile
```

Pasar a:

```python
@dataclass
class CalibrationSet:
    schema_version: int
    profiles: list[ViewProfile]
    active_profile_id: str | None
```

---

# 54. Formato de calibración V2

Ejemplo:

```json
{
  "schema_version": 2,
  "profiles": [
    {
      "id": "monitor_1",
      "name": "Monitor 1",
      "orientation": {
        "head_yaw_mean": 0.38,
        "torso_yaw_mean": 0.04,
        "head_yaw_std": 0.03,
        "torso_yaw_std": 0.02
      },
      "posture": {
        "good_mean": {},
        "good_std": {},
        "bad_mean": {},
        "thresholds": {}
      }
    },
    {
      "id": "monitor_2",
      "name": "Monitor 2",
      "orientation": {
        "head_yaw_mean": 0.03,
        "torso_yaw_mean": 0.01,
        "head_yaw_std": 0.02,
        "torso_yaw_std": 0.01
      }
    }
  ]
}
```

---

# 55. Migración V1 → V2

No romper calibraciones existentes.

Al cargar:

```python
if "schema_version" not in data:
```

interpretar como V1.

Crear automáticamente:

```text
CalibrationSet
└── ViewProfile
    └── "Principal"
```

Persistir V2 solo cuando corresponda.

---

# 56. Fase 23 — Wizard de calibración V2

## Paso 1 — Monitor principal

Mostrar:

```text
Sentate como trabajás normalmente.

Mirá hacia tu monitor principal.
```

Capturar:

```text
head orientation
torso orientation
shoulders
head/neck geometry
```

---

# 57. Paso 2 — Postura mala

Mostrar:

```text
Ahora adoptá tu postura mala habitual.
```

Objetivo:

```text
descubrir qué métricas son discriminantes
```

---

# 58. Paso 3 — Monitor adicional

Preguntar:

```text
¿Usás otra posición o pantalla habitualmente?
```

Botón:

```text
Agregar posición
```

Entonces:

```text
Mirá hacia ese monitor de forma normal.
```

Crear otro `ViewProfile`.

---

# 59. Calibración mala por perfil

En una primera versión no sería obligatorio repetir postura mala para cada monitor.

Podemos:

```text
usar calibración mala principal
+
adaptar baseline geométrico
```

Más adelante evaluar si cada vista necesita una calibración completa.

---

# 60. Fase 24 — Debug overlay

Antes de activar nuevas alertas, mostrar:

```text
VIEW
Monitor 1
confidence 92%

HEAD
yaw       +31.2
pitch      -3.4
roll       +1.1

TORSO
yaw        +6.8
forward    +2.0
roll       +0.7

RELATIVE
neck yaw  +24.4
neck roll  +0.4

SHOULDERS
left elevation    0.32
right elevation   0.30
roll              1.8
```

---

# 61. Estados visuales de métrica

Mostrar opcionalmente:

```text
✓ confiable
? baja confianza
— no disponible
```

Esto será importante para ajustar las fórmulas.

---

# 62. Fase 25 — Captura de snapshots de métricas

Agregar herramienta de diagnóstico:

```text
Guardar snapshot de métricas
```

Por privacidad, no necesita guardar imagen.

Ejemplo JSON:

```json
{
  "timestamp": "2026-10-05T10:00:00",
  "label": "monitor_1_good",
  "view": {
    "head_yaw": 0.32,
    "torso_yaw": 0.05
  },
  "metrics": {
    "neck_yaw_delta": 0.27,
    "shoulder_roll": 1.8
  }
}
```

---

# 63. Etiquetas manuales

Permitir marcar:

```text
monitor 1 correcta
monitor 2 correcta
cabeza adelantada
mirando abajo
cuello girado
hombros elevados
torso inclinado
```

Esto permite crear un dataset pequeño pero útil.

---

# 64. Fase 26 — Dataset controlado inicial

Registrar aproximadamente:

| Escenario | Duración |
|---|---:|
| Monitor 1 correcta | 30 s |
| Monitor 2 correcta | 30 s |
| Frontal | 30 s |
| Cabeza adelantada | 20 s |
| Mirando abajo | 20 s |
| Cuello girado sin torso | 20 s |
| Cuello + torso girado | 20 s |
| Hombros elevados | 20 s |
| Cabeza inclinada lateral | 20 s |
| Torso inclinado | 20 s |

---

# 65. Analizar dataset

Por métrica calcular:

```text
mean
std
min
max
coverage
confidence
separability
```

Ejemplo:

```text
neck_yaw_delta

Monitor 2 normal:
mean 4.2
std  1.4

Monitor 1 solo cuello:
mean 27.8
std  2.1

Monitor 1 torso acompañando:
mean 8.5
std  1.9
```

Esto permite decidir si la métrica realmente funciona.

---

# 66. Fase 27 — Carga temporal por issue

No todos los problemas deben ser tratados igual.

Por ejemplo:

```text
head forward fuerte durante 20 s
```

puede justificar alerta.

Pero:

```text
rotación cervical moderada
```

quizá solo debe alertar después de varios minutos.

---

# 67. Estado temporal por problema

En `PostureEngine` agregar:

```python
issue_streaks: dict[PostureIssue, float]
```

Ejemplo:

```python
issue_streaks = {
    PostureIssue.NECK_ROTATION: 43.2,
    PostureIssue.HEAD_FORWARD: 0.0,
}
```

---

# 68. Fase 28 — Carga acumulativa

Para problemas dependientes de tiempo:

```python
issue_load += severity * dt
```

Ejemplo:

```text
rotación 10°
durante 5 min
```

no pesa igual que:

```text
rotación 35°
durante 5 min
```

---

# 69. Política por issue

Futuro:

```python
@dataclass
class IssuePolicy:
    threshold_seconds: float
    cooldown_seconds: float
    weight: float
    load_based: bool
```

Ejemplo inicial:

| Problema | Tiempo |
|---|---:|
| Cabeza adelantada | 20 s |
| Hombro elevado | 30 s |
| Cabeza inclinada | 30 s |
| Torso inclinado | 20 s |
| Rotación cervical | 90–180 s |

Los valores definitivos deben salir de pruebas.

---

# 70. Eventos nuevos

Extender `Event`:

```python
class Event(StrEnum):
    ...
    ISSUE_STARTED = "issue_started"
    ISSUE_RECOVERED = "issue_recovered"
    ISSUE_ALERT = "issue_alert"
    POSTURE_LOAD_ALERT = "posture_load_alert"
    VIEW_CHANGED = "view_changed"
```

---

# 71. `EngineResult`

Agregar:

```python
active_view_id: str | None
active_view_name: str
dominant_issue: PostureIssue | None
issue_severity: float
issue_streak_seconds: float
```

Opcional:

```python
issue_states: dict[str, ...]
```

---

# 72. Fase 29 — Storage

No guardar todas las métricas permanentemente todavía.

Durante desarrollo crear una tabla separada.

Ejemplo:

```sql
CREATE TABLE debug_metric_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id INTEGER,
    view_profile TEXT,
    head_yaw REAL,
    torso_yaw REAL,
    neck_yaw_delta REAL,
    head_pitch REAL,
    head_roll REAL,
    shoulder_roll REAL,
    shoulder_elevation REAL,
    torso_forward_angle REAL
);
```

---

# 73. Retención

Esta tabla puede:

```text
existir solo en debug
```

o tener una retención corta.

Ejemplo:

```text
7 días
```

No hace falta convertirla en parte permanente del producto todavía.

---

# 74. Fase 30 — Analytics por issue

Cuando las métricas estén estabilizadas:

```text
Tiempo con cabeza adelantada
Tiempo con cuello girado
Tiempo con hombros elevados
Tiempo con torso inclinado
```

Dashboard:

```text
Hoy

Cabeza adelantada      18 min
Cuello girado          32 min
Hombros elevados       11 min
Torso inclinado         6 min
```

---

# 75. Analytics por monitor/vista

Posible evolución:

```text
Monitor 1
1h 42m
Postura 82%

Monitor 2
2h 11m
Postura 91%
```

Y:

```text
Monitor 1
principal problema:
rotación cervical
```

Esto permitiría detectar problemas del propio puesto de trabajo.

---

# 76. Fase 31 — No agregar Face Landmarker inicialmente

Primero intentar resolver el problema utilizando solo:

```text
Pose Landmarker
```

porque ya ofrece:

```text
nariz
ojos
orejas
hombros
caderas
world landmarks
```

Es suficiente para probar:

```text
yaw
roll
pitch proxy
torso orientation
hombros
```

---

# 77. Criterio para incorporar Face Landmarker

Solo incorporarlo si:

```text
head_yaw
head_pitch
```

resultan:

```text
demasiado ruidosos
insuficientes
ambiguos
```

durante las pruebas reales.

---

# 78. Fase 32 — Face Landmarker opcional

Si se incorpora:

```text
Pose Landmarker
→ cuerpo

Face Landmarker
→ cabeza
```

No ejecutar ambos necesariamente al mismo FPS.

Ejemplo:

```text
Pose
15 FPS

Face
5 FPS
```

Luego suavizar orientación entre frames.

---

# 79. Objetivo de Face Landmarker

Usarlo para:

```text
orientación precisa de cabeza
yaw
pitch
roll
mentón real
mandíbula
```

No para reemplazar detección corporal.

---

# 80. Mentón real

Solo en esta fase reemplazar:

```text
chin_drop
```

por algo anatómicamente más real:

```text
chin_neck_angle
head_pitch
```

Hasta entonces `chin_drop` debe considerarse únicamente un proxy de mirada/flexión.

---

# 81. Fase 33 — Tests unitarios de geometría

Crear:

```text
tests/test_geometry.py
```

Cubrir:

```text
distance
angles
midpoints
weighted average
visibility confidence
division by zero
```

---

# 82. Tests de orientación

Crear:

```text
tests/test_orientation.py
```

Cubrir:

```text
head yaw
head pitch
head roll
torso yaw
torso roll
neck yaw delta
neck roll delta
```

---

# 83. Tests de hombros

Cubrir:

```text
shoulder elevation
shoulder roll
asimetría
landmark faltante
confidence baja
```

---

# 84. Tests de métricas opcionales

Caso:

```text
perfil habilita 8 métricas
frame actual tiene 3
```

Verificar:

```text
clasificación utiliza solo las 3 disponibles
```

---

# 85. Tests de calibración coverage

Casos:

```text
90/90 frames
→ aceptar

80/90
→ aceptar

20/90
→ excluir
```

---

# 86. Tests de ViewProfile

Cubrir:

```text
perfil monitor 1
perfil monitor 2
selección correcta
hysteresis
unknown
```

---

# 87. Tests de carga temporal

Ejemplo:

```text
neck rotation severity 0.5
durante 10 s
```

vs:

```text
severity 2.0
durante 10 s
```

Verificar acumulación distinta.

---

# 88. Fase 34 — Compatibilidad con sistema actual

Durante la transición mantener:

```text
ear_shoulder_dx
nose_shoulder_dx
chin_drop
torso_lean_dx
```

junto a V2.

Esto permite comparar:

```text
legacy vs V2
```

sin romper alertas existentes.

---

# 89. Feature flag

Agregar:

```python
posture_v2_observe_only: bool = True
```

Mientras:

```text
true
```

las métricas V2:

```text
se calculan
se muestran
se registran
```

pero:

```text
no afectan clasificación
```

---

# 90. Activación progresiva

Orden:

```text
1. head yaw
2. torso yaw
3. neck yaw delta
4. shoulder roll
5. shoulder elevation
6. torso forward angle
7. head forward ratio
8. head pitch
9. neck roll delta
```

No activar todo de una vez.

---

# 91. Fase 35 — Retirar métricas legacy

Solo después de comparar resultados durante suficientes sesiones.

Posibles reemplazos:

```text
ear_shoulder_dx
→ head_forward_ratio

chin_drop
→ head_pitch

torso_lean_dx
→ torso_forward_angle
```

`nose_shoulder_dx` puede desaparecer si deja de aportar señal independiente.

---

# 92. Estructura objetivo aproximada

Después de V2:

```text
posture_guard/
    models.py
    config.py
    engine.py

    geometry.py
    landmarks.py
    orientation.py
    detection.py
    calibration.py

    storage.py
    state.py

    camera.py
    notifications.py
    ui.py
    dashboard.py

tests/
    test_geometry.py
    test_orientation.py
    test_detection.py
    test_calibration.py
    test_engine.py
```

No hace falta llegar a esto en una sola PR.

---

# 93. Orden recomendado de PRs

| PR | Contenido | Riesgo |
|---|---|---|
| V2-A | Confidence + clasificación solo con métricas disponibles | Bajo |
| V2-B | Coverage calibración + smoothing ponderado | Bajo |
| V2-C | `BodyLandmarks` + extractor 2D/3D | Medio |
| V2-D | `ViewState` + head yaw/roll/pitch debug | Medio |
| V2-E | torso yaw/roll + `neck_yaw_delta` | Medio |
| V2-F | shoulder elevation + shoulder roll | Medio |
| V2-G | torso forward angle + head forward ratio | Medio |
| V2-H | debug overlay + snapshot de métricas | Bajo |
| V2-I | dataset controlado + ajuste de fórmulas | Bajo |
| V2-J | `CalibrationSet` + `ViewProfile` + migración | Alto |
| V2-K | auto selección de perfil + hysteresis | Medio |
| V2-L | `PostureIssue` | Medio |
| V2-M | risk score ponderado | Medio |
| V2-N | issue streak/load temporal | Medio |
| V2-O | analytics por issue y view | Medio |
| V2-P | evaluar Face Landmarker | Experimental |

---

# 94. PR V2-A — Confidence y disponibilidad

## Objetivo

Preparar la arquitectura para métricas opcionales.

## Cambios

- agregar `confidence` a `DetectionMetrics`;
- agregar `metric_min_confidence`;
- cambiar `classify_posture()`;
- contar solo métricas:
  - habilitadas;
  - presentes;
  - suficientemente confiables.

## Tests

- métrica ausente;
- métrica baja confianza;
- combinación de métricas disponibles;
- todas ausentes;
- todas disabled.

---

# 95. PR V2-B — Coverage y smoothing

## Cambios

Agregar:

```text
calibration_min_metric_coverage
smoothing_min_observations
```

Modificar:

```text
Calibrator
RollingMetrics
```

para utilizar:

```text
coverage
confidence
weighted mean
```

---

# 96. PR V2-C — BodyLandmarks

## Cambios

Crear:

```text
Point2D
Point3D
BodyLandmarks
```

Extraer:

```text
2D landmarks
world landmarks
```

Mantener salida legacy actual.

No modificar todavía comportamiento productivo.

---

# 97. PR V2-D — Orientación de cabeza

Implementar:

```text
head_yaw
head_roll
head_pitch
ViewState
```

Solo:

```text
debug
telemetría
tests
```

No alertas.

---

# 98. PR V2-E — Orientación torso y cuello

Implementar:

```text
torso_yaw
torso_roll
neck_yaw_delta
neck_roll_delta
```

Agregar debug.

Validar:

```text
solo cuello
cuello + torso
```

---

# 99. PR V2-F — Hombros

Implementar:

```text
left_shoulder_elevation
right_shoulder_elevation
shoulder_elevation
shoulder_roll
```

Condicional según visibilidad.

---

# 100. PR V2-G — Cabeza y torso normalizados

Implementar:

```text
head_forward_ratio
torso_forward_angle
```

Mantener equivalentes legacy para comparación.

---

# 101. PR V2-H — Herramientas de validación

Agregar:

```text
overlay debug
snapshot metrics
manual labels
```

No afectar comportamiento normal.

---

# 102. PR V2-I — Dataset real

Realizar pruebas con:

```text
monitor 1
monitor 2
frontal
mala postura
hombros elevados
cuello girado
```

Analizar:

```text
stability
coverage
confidence
separability
```

Ajustar fórmulas antes de activar V2.

---

# 103. PR V2-J — Multi-view calibration

Crear:

```text
CalibrationSet
ViewProfile
schema_version 2
migración V1
```

Agregar UI para:

```text
Agregar posición
Eliminar posición
Renombrar posición
```

---

# 104. PR V2-K — Auto view

Implementar:

```text
profile distance
selection
hysteresis
VIEW_CHANGED
```

Mostrar:

```text
Vista activa: Monitor 1
```

---

# 105. PR V2-L — PostureIssue

Separar completamente:

```text
métrica técnica
```

de:

```text
problema ergonómico
```

Actualizar `issue_details()`.

---

# 106. PR V2-M — Risk score

Agregar:

```text
weight
confidence weighting
severity aggregation
```

Ejecutar inicialmente en paralelo con clasificación legacy.

Comparar resultados.

---

# 107. PR V2-N — Carga temporal

Agregar:

```text
issue_streaks
issue_load
```

Políticas distintas según problema.

Activar inicialmente solo:

```text
NECK_ROTATION
```

porque es el caso multi-monitor más claro.

---

# 108. PR V2-O — Analytics

Agregar:

```text
tiempo por issue
tiempo por view
principal issue por monitor
```

Actualizar dashboard.

---

# 109. PR V2-P — Face Landmarker

Solo si las métricas de cabeza basadas en Pose no alcanzan.

Evaluar:

```text
precisión
CPU
latencia
tamaño de modelo
estabilidad
```

Antes de incorporarlo.

---

# 110. Criterios de éxito V2-A/B

Debe cumplirse:

- una métrica ausente no cuenta como buena ni mala;
- una métrica con baja confianza no clasifica;
- calibración descarta métricas con mala cobertura;
- smoothing no publica observaciones aisladas;
- todos los tests existentes siguen verdes.

---

# 111. Criterios de éxito V2-C–G

Con usuario quieto:

```text
las métricas permanecen estables
```

Al hacer movimientos controlados:

```text
cada métrica cambia en dirección consistente
```

No se exige todavía que los umbrales finales sean perfectos.

---

# 112. Criterio de éxito multi-monitor

En una situación tipo:

```text
Monitor 1:
head yaw alto
torso yaw bajo
neck yaw delta alto
```

y:

```text
Monitor 2:
head yaw bajo
torso yaw bajo
neck yaw delta bajo
```

PostureBreaker debe distinguir ambos contextos.

Cambiar de monitor no debe producir automáticamente:

```text
Cabeza adelantada
```

si la relación cabeza-cuerpo sigue siendo razonable.

---

# 113. Criterio de éxito ViewProfile

Cambiar de monitor debe:

```text
detectar nueva vista
esperar hysteresis
activar perfil correcto
```

sin oscilaciones.

---

# 114. Criterio de éxito de hombros

Un movimiento deliberado:

```text
subir hombros hacia las orejas
```

debe modificar consistentemente:

```text
shoulder_elevation
```

Un hombro más alto que otro debe modificar:

```text
shoulder_roll
```

sin requerir ambos hombros cuando uno no es observable.

---

# 115. Criterio de éxito de rotación cervical

Debe distinguir:

```text
cabeza girada
torso frontal
```

de:

```text
cabeza girada
torso acompañando
```

mediante:

```text
neck_yaw_delta
```

---

# 116. Criterio de éxito de alertas

Una vez activadas:

```text
movimiento momentáneo
→ no alerta

posición sostenida
→ alerta según issue

rotación moderada prolongada
→ alerta de carga

corrección
→ recuperación
```

---

# 117. Prioridad inmediata

El trabajo inmediato debería ser:

```text
1. V2-A
2. V2-B
3. V2-C
4. V2-D
5. V2-E
```

Es decir:

```text
confidence
coverage
BodyLandmarks
head orientation
torso orientation
neck_yaw_delta
```

Antes de agregar nuevas alertas.

---

# 118. Primer milestone

Considerar terminado el primer milestone cuando PostureBreaker pueda mostrar de forma estable:

```text
Head yaw
Head pitch
Head roll
Torso yaw
Torso roll
Neck yaw delta
Neck roll delta
```

sin afectar todavía el comportamiento actual.

---

# 119. Segundo milestone

Agregar:

```text
Shoulder elevation
Shoulder roll
Torso forward angle
Head forward ratio
```

y validarlos con dataset controlado.

---

# 120. Tercer milestone

Implementar:

```text
ViewProfile
CalibrationSet V2
selección automática de monitor
```

---

# 121. Cuarto milestone

Migrar clasificación desde:

```text
cantidad de métricas malas
```

hacia:

```text
PostureIssue
+
severity
+
confidence
+
weight
+
time
```

---

# 122. Resultado objetivo

Al finalizar V2, el sistema debería entender situaciones del tipo:

```text
Estás mirando el Monitor 1.
Tu torso está casi frontal.
Tu cabeza lleva varios minutos girada respecto del torso.
```

o:

```text
Estás mirando el Monitor 2.
Cabeza y torso están alineados.
No hay problema cervical relevante.
```

o:

```text
Tenés los hombros elevados.
Bajalos y relajá el trapecio.
```

o:

```text
Tu torso está razonablemente bien,
pero la cabeza se adelantó respecto del hombro.
```

En lugar de limitarse a:

```text
2 métricas malas
→ mala postura
```

---

# 123. Conclusión

Posture Detection V2 debería desarrollarse como una evolución incremental del detector actual, no como una reescritura.

La prioridad no es sumar la mayor cantidad posible de puntos.

La prioridad es obtener señales:

```text
estables
normalizadas
confiables
interpretables
condicionales a visibilidad
```

y después relacionarlas entre sí.

Las relaciones más importantes serán:

```text
cabeza ↔ torso
cabeza ↔ hombros
hombros ↔ torso
```

En particular:

```text
neck_yaw_delta
neck_roll_delta
head_forward_ratio
shoulder_elevation
torso_forward_angle
```

deberían convertirse en piezas centrales del nuevo sistema.

El flujo correcto de desarrollo será:

```text
medir
↓
visualizar
↓
registrar
↓
validar
↓
calibrar
↓
clasificar
↓
alertar
```

y no:

```text
inventar métrica
↓
poner threshold
↓
alertar
```

Ese enfoque debería permitir que PostureBreaker evolucione hacia un monitor postural bastante más robusto y adaptable a configuraciones reales de trabajo con múltiples monitores.