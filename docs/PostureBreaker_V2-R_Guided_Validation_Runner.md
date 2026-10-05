# PostureBreaker — V2-R Guided Validation Runner

## 1. Objetivo

Implementar un **modo de validación V2 guiado y automático** para reemplazar la captura manual con teclas y snapshots aislados.

La aplicación debe:

- indicar qué postura realizar;
- dar unos segundos para acomodarse;
- capturar automáticamente una serie de muestras durante un intervalo fijo;
- avanzar al siguiente escenario sin intervención manual;
- conservar frames sin métricas para medir coverage real;
- no contaminar SQLite ni las estadísticas normales de uso;
- generar un dataset separado por corrida;
- generar un reporte automático al finalizar.

Esta ronda es exclusivamente de **validación experimental**.

No modificar:

- fórmulas V2;
- `MetricSpec`;
- calibración;
- `risk.py`;
- clasificación;
- thresholds;
- weights;
- `IssuePolicy`.

Mantener:

```text
posture_v2_observe_only = true
```

---

## 2. Premisas del entorno real

El caso de uso principal es:

- una cámara fija;
- cámara montada sobre el Monitor 2;
- Monitor 1 y Monitor 2 se distinguen por la orientación del usuario respecto de esa misma cámara;
- las caderas normalmente no están visibles;
- la validación principal es `DESK_CORE`;
- las métricas `FULL_BODY_OPTIONAL` quedan fuera de este protocolo.

El test debe trabajar sobre las mismas métricas suavizadas que después consume producción.

Usar:

```text
raw MediaPipe
    ↓
extract_metrics()
    ↓
RollingMetrics
    ↓
metrics = smoother.mean()
    ↓
ValidationRunner
```

La captura debe utilizar `metrics`, no `raw_metrics`.

---

## 3. Escenarios del protocolo desk

Usar como fuente única los IDs existentes en `SNAPSHOT_LABELS`.

Orden:

```text
1. good
2. head_forward
3. head_down
4. head_tilt_left
5. head_tilt_right
6. shoulders_up
7. shoulder_left_up
8. shoulder_right_up
9. head_turn_only
10. head_torso_turn
```

No incluir en esta batería normal:

```text
torso_forward
torso_lean_left
torso_lean_right
```

Esos escenarios pertenecen a `FULL_BODY_SNAPSHOT_LABELS`.

---

## 4. Instrucciones humanas

Cada escenario debe tener:

```python
id
title
instruction
prepare_seconds
capture_seconds
```

### `good`

**Título:** Postura normal

**Instrucción:**

> Sentate como trabajás normalmente y mirá tu monitor.

---

### `head_forward`

**Título:** Cabeza adelantada

**Instrucción:**

> Adelantá la cabeza hacia la pantalla sin mover los hombros.

---

### `head_down`

**Título:** Mirada hacia abajo

**Instrucción:**

> Bajá la mirada sin inclinar deliberadamente el torso.

---

### `head_tilt_left`

**Título:** Cabeza inclinada a la izquierda

**Instrucción:**

> Incliná la cabeza hacia la izquierda sin subir los hombros.

---

### `head_tilt_right`

**Título:** Cabeza inclinada a la derecha

**Instrucción:**

> Incliná la cabeza hacia la derecha sin subir los hombros.

---

### `shoulders_up`

**Título:** Ambos hombros elevados

**Instrucción:**

> Elevá ambos hombros manteniendo la cabeza lo más estable posible.

---

### `shoulder_left_up`

**Título:** Hombro izquierdo elevado

**Instrucción:**

> Elevá solo el hombro izquierdo.

---

### `shoulder_right_up`

**Título:** Hombro derecho elevado

**Instrucción:**

> Elevá solo el hombro derecho.

---

### `head_turn_only`

**Título:** Giro solo de cabeza

**Instrucción:**

> Girá la cabeza hacia un lado manteniendo los hombros quietos.

---

### `head_torso_turn`

**Título:** Giro de cabeza y hombros

**Instrucción:**

> Girá cabeza y hombros juntos hacia el mismo lado.

---

## 5. Intensidad de las posturas

No pedir posiciones extremas.

La consigna debe representar una alteración clara pero razonable de la postura habitual.

El objetivo es medir:

```text
baseline
vs
cambio controlado
```

no generar máximos articulares.

---

## 6. Timing inicial

Hacerlo configurable.

Defaults:

```python
prepare_seconds = 4.0
capture_seconds = 20.0
sample_interval_seconds = 0.5
```

Resultado aproximado:

```text
40 muestras / escenario
10 escenarios
≈ 400 muestras / corrida
```

No usar `sleep()` dentro del runner.

El runner debe avanzar usando tiempo monotónico / `dt`.

---

## 7. Nuevo módulo

Crear:

```text
posture_guard/validation.py
```

Estructuras sugeridas:

```python
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class ValidationScenario:
    id: str
    title: str
    instruction: str
    prepare_seconds: float
    capture_seconds: float


class ValidationPhase(StrEnum):
    IDLE = "idle"
    PREPARE = "prepare"
    CAPTURE = "capture"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
```

Implementar:

```python
class ValidationRunner:
    ...
```

Debe ser independiente de:

- OpenCV;
- MediaPipe;
- SQLite;
- CustomTkinter;
- notificaciones.

Debe poder testearse de forma aislada.

---

## 8. Responsabilidades de `ValidationRunner`

El runner debe manejar:

```text
IDLE
 ↓
PREPARE
 ↓
CAPTURE
 ↓
PREPARE siguiente escenario
 ↓
...
 ↓
COMPLETE
```

También:

```text
cualquier fase
 ↓
CANCELLED
```

Debe exponer estado suficiente para UI:

```text
phase
scenario_index
scenario_count
scenario_id
title
instruction
remaining_seconds
progress
sample_count
run_id
```

---

## 9. `run_id`

Cada ejecución completa debe tener un identificador único.

Formato sugerido:

```text
YYYYMMDD-HHMMSS
```

Ejemplo:

```text
20261005-181530
```

El mismo `run_id` debe conservarse durante toda la corrida.

No cambiarlo al pasar de escenario.

---

## 10. Vista de validación fija

Este punto es crítico.

Al iniciar el test:

1. debe existir una vista activa estable;
2. guardar:

```text
validation_view_id
validation_view_name
```

3. esa vista queda asociada a **toda la corrida**.

No reasignar muestras de escenario porque `ViewSelector` cambie durante la prueba.

---

## 11. Motivo

Escenarios como:

```text
head_turn_only
head_torso_turn
```

pueden alterar `head_yaw` / `torso_yaw`.

Por lo tanto `ViewSelector` podría decidir temporalmente que el usuario está mirando otra vista.

Eso es información útil, pero no debe cambiar la identidad del test.

Cada muestra debe guardar dos contextos:

```text
test_view
active_view
```

Ejemplo:

```text
scenario     = head_turn_only
test_view    = Monitor 2
active_view  = Monitor 1
```

Interpretación:

> la muestra sigue perteneciendo al test de Monitor 2, pero durante ese movimiento el selector habría cambiado de vista.

Eso sirve también para evaluar la robustez del selector.

---

## 12. Captura automática

Durante `CAPTURE`, generar una muestra cada:

```text
sample_interval_seconds
```

Usar:

```python
build_snapshot(...)
```

como base.

Capturar las métricas suavizadas:

```python
metrics = smoother.mean()
```

No usar `raw_metrics`.

---

## 13. Capturar también `metrics=None`

No guardar únicamente frames válidos.

Cada intervalo programado debe producir una muestra aunque:

```python
metrics is None
```

Motivo:

si se esperaban:

```text
40 muestras
```

y la métrica apareció solo en:

```text
32
```

la cobertura real debe ser:

```text
80%
```

No:

```text
100%
```

`build_snapshot()` ya soporta `metrics=None`; aprovecharlo.

---

## 14. Metadata de cada muestra

Agregar como mínimo:

```text
run_id
scenario
sample_index
scenario_elapsed_seconds

test_view_id
test_view_name

active_view_id
active_view_name
```

Conservar también lo actual:

```text
timestamp
side

view
shoulders
forward
observations

legacy metrics
confidence
```

---

## 15. Estructura sugerida del snapshot

Ejemplo:

```json
{
  "run_id": "20261005-181530",
  "scenario": "head_forward",
  "sample_index": 17,
  "scenario_elapsed_seconds": 8.5,

  "test_view": {
    "id": "view_1",
    "name": "Monitor 2"
  },

  "active_view": {
    "id": "view_1",
    "name": "Monitor 2"
  },

  "observations": {}
}
```

No es obligatorio usar exactamente esta forma si el esquema actual recomienda otra, pero deben conservarse esos conceptos.

---

## 16. Archivos de validación

No guardar el dataset en SQLite.

Crear por corrida:

```text
history/
└── validation/
    ├── <run_id>.jsonl
    └── <run_id>-report.txt
```

Ejemplo:

```text
history/validation/20261005-181530.jsonl
history/validation/20261005-181530-report.txt
```

No mezclar diferentes runs automáticamente en el mismo JSONL.

---

## 17. SQLite sigue reservado para uso real

Durante validación, la DB normal no debe recibir datos provocados por el protocolo.

No contaminar:

```text
daily_stats
hourly_stats
posture_samples
posture_events
break_events
issue_time
view_time
focus_periods
```

---

## 18. Suspender analytics durante validation

Mientras:

```python
validation_active is True
```

NO ejecutar acumulación de:

```text
work time
good posture time
bad posture time
issue time
view time
samples analytics
alert counters
focus statistics
```

El usuario está realizando posturas malas deliberadamente.

---

## 19. Suspender alertas y políticas temporales

Durante el test NO emitir:

```text
POSTURE_ALERT
ISSUE_ALERT
POSTURE_LOAD_ALERT
BREAK_ALERT
BREAK_ALERT_REPEAT
```

No mostrar correcciones automáticas contradictorias con la consigna del test.

Ejemplo incorrecto:

```text
TEST:
"Elevá ambos hombros"

ALERTA:
"Bajá los hombros"
```

---

## 20. Engine durante validation

Preferencia:

no llamar al engine para lógica temporal/productiva durante la captura.

Si por arquitectura conviene seguir llamándolo para obtener algún dato, impedir que:

- modifique streaks;
- acumule loads;
- emita eventos;
- afecte analytics.

La validación debe ser una sesión experimental aislada.

---

## 21. Reset al finalizar

Tanto al completar como al cancelar:

```python
smoother.clear()
engine.reset_posture_state()
```

Además limpiar cualquier estado temporal de validación.

Motivo:

no arrastrar una postura provocada al modo productivo.

---

## 22. `SharedState`

Agregar los campos necesarios.

Ejemplo:

```python
cmd_start_validation: bool = False
cmd_cancel_validation: bool = False

validation_active: bool = False
validation_phase: str = ""
validation_scenario: str = ""
validation_title: str = ""
validation_instruction: str = ""
validation_progress: float = 0.0
validation_remaining_seconds: float = 0.0
validation_scenario_index: int = 0
validation_scenario_count: int = 0
validation_sample_count: int = 0
validation_run_id: str = ""
validation_view_name: str = ""
```

Ajustar nombres al estilo existente si conviene.

---

## 23. Dashboard

Agregar una tarjeta dentro de `Herramientas`:

```text
Validación detector V2

Ejecuta una serie guiada de posturas y genera
un dataset para evaluar las métricas.

[ Ejecutar test V2 ]
```

---

## 24. Condiciones para habilitar el botón

Habilitar `Ejecutar test V2` solo si:

- cámara disponible;
- existe calibración;
- existe vista activa;
- no hay calibración en curso;
- no hay validación activa.

Si alguna condición falla, mostrar el motivo.

---

## 25. UI durante test

Mostrar una tarjeta destacada.

Ejemplo:

```text
VALIDACIÓN V2 — 3 / 10

Mirada hacia abajo

Bajá la mirada sin mover el torso.

CAPTURANDO · 12 s

██████████████░░░░░░

26 muestras

[ Cancelar ]
```

---

## 26. Fase PREPARE

Ejemplo:

```text
VALIDACIÓN V2 — 3 / 10

Mirada hacia abajo

Bajá la mirada sin mover el torso.

Preparación
Comenzamos en 3 s
```

No capturar muestras durante esta fase.

---

## 27. Fase CAPTURE

Ejemplo:

```text
VALIDACIÓN V2 — 3 / 10

CAPTURANDO

Mantené la posición.

12 s restantes

26 / 40 muestras
```

---

## 28. Overlay de cámara

Mostrar una instrucción grande también en la ventana de cámara.

No depender exclusivamente del dashboard.

### PREPARE

```text
TEST V2 · 3/10

MIRADA HACIA ABAJO

Bajá la mirada sin mover el torso.

Comenzamos en 3...
```

### CAPTURE

```text
CAPTURANDO

Mantené la posición

12 s
```

Mantener las guías de debug existentes.

La instrucción del test debe tener prioridad visual.

---

## 29. Cancelación

Agregar:

```text
[ Cancelar ]
```

Si el usuario cancela:

- cerrar correctamente el archivo;
- marcar runner `CANCELLED`;
- no generar un informe como si fuera una corrida completa;
- opcionalmente conservar el JSONL parcial con metadata `cancelled=true`;
- limpiar smoother;
- resetear engine;
- restaurar UI normal.

No borrar silenciosamente datos ya capturados.

---

## 30. Informe automático

Al completar la corrida:

1. cargar el JSONL generado;
2. resumir métricas;
3. usar baseline `good` de la vista de test;
4. comparar escenarios;
5. generar matriz de respuesta / cross-talk;
6. escribir:

```text
<run_id>-report.txt
```

Reutilizar:

```python
summarize_snapshots()
compare_to_view_baselines()
metric_response_matrix()
```

Adaptar el agrupamiento para usar `test_view` como identidad de la corrida, no el `active_view` cambiante.

---

## 31. No dar veredicto automático todavía

El reporte debe mostrar datos.

No marcar aún:

```text
PASS
FAIL
PRODUCTION READY
```

para las métricas.

Queremos evaluar primero:

- coverage;
- confidence;
- baseline stability;
- separabilidad;
- dirección;
- cross-talk;
- comportamiento por vista.

---

## 32. Métricas especialmente importantes

La validación debe permitir comparar:

```text
head_forward_ratio
head_depth_ratio
```

por vista.

Especialmente:

```text
Monitor 1
Monitor 2
```

Queremos determinar si:

- `head_depth_ratio` funciona frontalmente;
- `head_forward_ratio` sigue siendo útil lateralmente;
- ambas son complementarias;
- alguna tiene demasiado cross-talk.

---

## 33. Shoulder validation

Esperado:

### `shoulder_left_up`

Debe afectar principalmente:

```text
left_shoulder_elevation
shoulder_roll
```

y poco:

```text
right_shoulder_elevation
head_roll
```

### `shoulder_right_up`

Inverso.

### `shoulders_up`

Deberían reaccionar:

```text
left_shoulder_elevation
right_shoulder_elevation
```

---

## 34. Head roll validation

Esperado:

```text
head_tilt_left
head_tilt_right
```

deben producir cambios fuertes y opuestos en:

```text
head_roll
```

con relativamente poco cambio en:

```text
shoulder_roll
```

---

## 35. Head pitch validation

`head_down` debe producir principalmente respuesta en:

```text
head_pitch
```

y no disparar de forma fuerte señales no relacionadas.

---

## 36. View selector validation

Los escenarios:

```text
head_turn_only
head_torso_turn
```

también sirven para observar si:

```text
active_view
```

cambia indebidamente durante movimientos transitorios.

Registrar esos cambios, pero no cambiar `test_view`.

---

## 37. Repetibilidad

Cada corrida queda identificada por `run_id`.

Esto permitirá después comparar:

```text
Run A — Monitor 2
Run B — Monitor 2
```

sin contaminar los labels de escenario.

No crear labels:

```text
good_A
good_B
```

La repetibilidad debe salir de `run_id`.

---

## 38. Frecuencia de muestreo

No capturar cada frame.

Default:

```text
2 Hz
```

equivalente a:

```text
sample_interval_seconds = 0.5
```

Esto evita redundancia excesiva porque `RollingMetrics` ya suaviza múltiples frames.

---

## 39. Archivos parciales

Escribir incrementalmente el JSONL.

No acumular todo en memoria y escribir al final.

Así una interrupción no destruye toda la corrida.

---

## 40. Errores de escritura

Si falla el archivo de validación:

- cancelar la corrida;
- mostrar error en UI;
- volver al modo normal;
- no continuar una prueba cuyos datos no se están guardando.

---

## 41. Cámara perdida

Si la cámara se pierde durante validation:

- pausar o cancelar la corrida;
- no avanzar timers de CAPTURE como si hubiera datos;
- preferencia inicial: cancelar con error claro.

No generar un reporte “completo” con media corrida perdida.

---

## 42. Cambios de pose / `metrics=None`

`metrics=None` durante una captura normal:

- sí consume un intervalo de muestreo;
- sí genera snapshot;
- cuenta contra coverage;
- no cancela automáticamente la corrida.

Una pérdida prolongada de cámara completa sí es otro caso.

---

## 43. Compatibilidad con captura manual

No es obligatorio eliminar todavía:

```text
1–0
n/p
s
```

Puede conservarse como herramienta debug manual.

El Guided Validation Runner pasa a ser el flujo recomendado.

---

## 44. Tests del runner

Cubrir:

### Máquina de estados

```text
IDLE → PREPARE
PREPARE → CAPTURE
CAPTURE → siguiente PREPARE
último CAPTURE → COMPLETE
```

---

### Timing

Comprobar:

- preparación respeta duración;
- captura respeta duración;
- sampling usa el intervalo configurado;
- no hay samples durante PREPARE.

---

### Sampling

Verificar:

```text
sample_interval = 0.5
capture_seconds = 20
```

→ cantidad esperada aproximadamente 40, sin duplicados por mismo tick.

---

### Missing metrics

```python
metrics=None
```

debe generar snapshot.

---

### Vista

Cambiar:

```text
active_view
```

durante la corrida no debe modificar:

```text
test_view
```

---

### `run_id`

Debe mantenerse constante durante los 10 escenarios.

---

### Cancelación

Cancelar en:

```text
PREPARE
CAPTURE
```

debe dejar el sistema consistente.

---

### Analytics

Mientras validation está activa:

```text
daily_stats
hourly_stats
issue_time
view_time
posture_samples
posture_events
break_events
```

no deben modificarse por los escenarios provocados.

---

### Reset

Al terminar/cancelar:

```text
smoother vacío
issue_streaks vacíos
issue_loads vacíos
```

---

### Reporte

Una corrida completa debe generar:

```text
.jsonl
-report.txt
```

---

## 45. Tests de integración

Crear al menos un test que simule una corrida corta:

```text
2 escenarios
prepare = 0.1
capture = 0.5
interval = 0.1
```

y comprobar:

- orden;
- samples;
- metadata;
- transición final;
- archivo generado.

No depender de reloj real si puede inyectarse tiempo.

---

## 46. Configuración

Los valores deben poder quedar en `Config`.

Ejemplo:

```python
validation_prepare_seconds: float = 4.0
validation_capture_seconds: float = 20.0
validation_sample_interval_seconds: float = 0.5
```

No exponerlos todavía en UI salvo que sea trivial.

---

## 47. No tocar V2-P

No implementar Face Landmarker en esta ronda.

La validación actual debe determinar si Pose alcanza.

---

## 48. No tocar risk experimental

Mantener cualquier señal experimental en el estado actual de risk definido antes de esta ronda.

El test opera sobre `observations` independientemente de que una métrica esté o no habilitada para riesgo.

---

## 49. Definition of Done

La ronda queda cerrada cuando:

- existe botón `Ejecutar test V2`;
- el test requiere una vista activa;
- la vista queda bloqueada conceptualmente como `test_view`;
- hay 10 escenarios desk guiados;
- cada escenario tiene PREPARE + CAPTURE;
- se toman snapshots automáticos;
- se capturan también intervalos con `metrics=None`;
- el dataset se guarda fuera de SQLite;
- SQLite no se contamina;
- no se disparan alertas durante el test;
- no avanzan políticas temporales productivas;
- puede cancelarse;
- smoother/engine se resetean al salir;
- se genera JSONL;
- se genera reporte TXT;
- `run_id` permite comparar corridas;
- `posture_v2_observe_only=true`;
- suite completa verde;
- Ruff limpio.

---

# 50. Flujo esperado de usuario

```text
Abrir PostureBreaker
        ↓
mirar normalmente Monitor 2
        ↓
Ejecutar test V2
        ↓
10 escenarios guiados
        ↓
dataset + reporte
        ↓
volver a postura normal
        ↓
mirar Monitor 1
        ↓
Ejecutar test V2 nuevamente
        ↓
dataset + reporte
```

Después se repetirán ambos tests en otra corrida para medir reproducibilidad.

---

# 51. Objetivo posterior

No activar métricas automáticamente al finalizar.

La siguiente decisión se tomará después de analizar los datasets:

```text
coverage
confidence
reproducibilidad
separabilidad
cross-talk
comportamiento por vista
```

Con foco especial en:

```text
head_forward_ratio
vs
head_depth_ratio
```

y en la estabilidad de:

```text
head_pitch
head_roll
shoulder_roll
left_shoulder_elevation
right_shoulder_elevation
```
