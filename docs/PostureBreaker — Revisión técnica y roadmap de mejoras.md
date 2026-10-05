# PostureBreaker — Revisión técnica y roadmap de mejoras

## 1. Objetivo

Este documento resume una revisión técnica completa del proyecto **PostureBreaker**, identificando:

- problemas funcionales actuales;
- riesgos de arquitectura;
- inconsistencias en analytics;
- oportunidades de mejora en detección y calibración;
- problemas potenciales de concurrencia y lifecycle;
- mejoras de robustez para distribución como aplicación de escritorio;
- deuda de packaging, CI y documentación;
- una propuesta de roadmap priorizado.

La conclusión general es que **no hace falta reescribir el proyecto ni cambiar de stack**.

Python, MediaPipe, OpenCV, SQLite y CustomTkinter son tecnologías adecuadas para el tipo de aplicación.

La base actual además ya presenta una separación razonable entre:

- detección;
- calibración;
- alertas;
- configuración;
- estado compartido;
- persistencia;
- notificaciones;
- UI.

El principal problema no es la arquitectura general sino que `main.py` está absorbiendo cada vez más lógica temporal y de negocio, mientras algunos comportamientos cruzados entre módulos no están cubiertos por tests.

---

# 2. Estado general

## Fortalezas actuales

El proyecto ya dispone de una estructura bastante superior a un simple prototipo monolítico.

Actualmente existen módulos separados para:

```text
posture_guard/
  alerts.py
  calibration.py
  config.py
  dashboard.py
  detection.py
  main.py
  models.py
  notifications.py
  state.py
  storage.py
  ui.py
```

También existen tests para:

```text
alerts
calibration
config
detection
fetch_model
storage
```

La configuración está validada, las dependencias están fijadas, el modelo de MediaPipe dispone de descarga con verificación SHA256 y la persistencia ya fue migrada a SQLite.

Eso constituye una buena base.

No se recomienda realizar una reescritura total.

---

# 3. Resumen de prioridades

| Prioridad | Problema | Impacto | Acción recomendada |
|---|---|---|---|
| Alta | Repetición de pausas rota en modo foco | Los recordatorios pueden no mostrarse aunque se registren | Separar timestamps de repetición |
| Alta | Trend horario incorrecto | Las estadísticas por hora pueden ser engañosas | Calcular tiempo bueno/malo por intervalo |
| Alta | Calibración mala demasiado estricta | Se rechazan calibraciones útiles | Permitir métricas no discriminantes |
| Alta | `dt_seconds` vulnerable a suspensión/stalls | Puede sumar minutos u horas falsas | Detectar gaps temporales |
| Alta | Control de FPS incorrecto | `target_fps` no representa FPS real | Medir el ciclo completo |
| Alta | Cadera no validada por visibilidad | Puede generar falsos positivos de torso | Validar hip o excluir métrica |
| Alta | Excepciones del worker poco visibles | La app puede quedar aparentemente congelada | Logging + error state |
| Media | `main.py` concentra demasiada lógica | Aumenta complejidad y dificulta testing | Extraer `PostureEngine` |
| Media | SQLite sin versionado de schema | Cambios futuros pueden romper datos | `PRAGMA user_version` |
| Media | Datos junto al `.exe` | Problemas con instalaciones protegidas | Usar AppData |
| Media | Cámara sin recuperación | Fallos temporales exigen reinicio | Retry/backoff |
| Media | Tray con flags compartidos | Posibles race conditions | Usar `Queue` |
| Baja | Packaging duplicado | Build más difícil de mantener | Centralizar en `.spec` |
| Baja | README inconsistente | Instalación confusa | Unificar documentación |
| Baja | Sin CI | Los tests no protegen `main` | GitHub Actions |
| Baja | Sin releases/versionado formal | Distribución manual | Versionado y releases |

---

# 4. Bug: repetición de pausas en modo foco

## Problema

`BreakManager.update()` utiliza:

```python
if now_ts - self.state.last_break_alert_at >= self.config.break_repeat_alert_seconds:
    self.state.last_break_alert_at = now_ts
    events.append("break_alert_repeat")
```

Luego `main.py`, al recibir el evento, intenta comprobar:

```python
elif now_ts - alert_state.last_break_alert_at >= config.focus_break_repeat_seconds:
```

El problema es que `BreakManager` acaba de hacer:

```python
self.state.last_break_alert_at = now_ts
```

Por lo tanto:

```text
now_ts - last_break_alert_at ≈ 0
```

y la condición de `focus_break_repeat_seconds` no puede cumplirse en ese momento.

## Consecuencia

En modo foco pueden ocurrir dos cosas simultáneamente:

1. no se muestra la notificación esperada;
2. igualmente se registra un `break_alert_repeat` en la base.

Eso puede provocar que las estadísticas indiquen alertas que el usuario nunca recibió.

## Solución propuesta

Separar explícitamente el estado:

```text
break_due_since
last_break_reminder_at
last_focus_break_reminder_at
```

Además, `BreakManager` debería encargarse solamente de determinar el estado de la pausa.

La política de notificación debería estar separada.

Por ejemplo:

```text
BREAK_DUE
BREAK_PROGRESS
BREAK_INTERRUPTED
BREAK_COMPLETED
```

Y después otra capa decide:

```text
si debe mostrar notificación
si debe mostrar overlay
si debe emitir sonido
si debe registrar un reminder
```

---

# 5. Analytics horario incorrecto

## Problema

`record_sample()` almacena:

```python
score=self.score
```

pero `self.score` representa el score acumulado de todo el día.

Posteriormente `hourly_trend()` calcula:

```sql
SELECT strftime('%H', ts) AS hour_key, AVG(score) AS avg_score
FROM posture_samples
WHERE date(ts) = ?
GROUP BY hour_key
```

Esto no representa realmente la calidad postural de esa hora.

## Ejemplo

Supongamos:

```text
09:00–10:00 → 100 % buena postura
10:00–11:00 → 100 % mala postura
```

Durante la segunda hora el score acumulado va disminuyendo progresivamente desde aproximadamente 100 % hasta 50 %.

Promediar esos snapshots puede producir algo similar a:

```text
10:00–11:00 → 65–75 %
```

aunque en esa hora concreta la postura fue:

```text
0 % buena
100 % mala
```

## Solución

Las estadísticas temporales deberían derivarse del tiempo real:

```text
good_posture_seconds
bad_posture_seconds
```

Para cada hora:

```text
hour_score =
    good_seconds
    /
    (good_seconds + bad_seconds)
    * 100
```

## Alternativa

Crear una tabla agregada:

```sql
CREATE TABLE hourly_stats (
    day TEXT NOT NULL,
    hour INTEGER NOT NULL,
    good_posture_seconds REAL NOT NULL DEFAULT 0,
    bad_posture_seconds REAL NOT NULL DEFAULT 0,
    work_seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(day, hour)
);
```

Esto simplificaría los trends y evitaría calcularlos sobre snapshots acumulativos.

---

# 6. Card “Mala postura”

Actualmente el dashboard muestra:

```python
bad_streak_seconds
```

en la tarjeta:

```text
Mala postura
```

Pero este valor representa únicamente la racha continua actual de mala postura.

No representa:

```text
mala postura total del día
```

## Opciones

### Opción A

Cambiar el nombre a:

```text
Mala postura actual
```

### Opción B — recomendada

Mostrar:

```text
bad_posture_seconds del día
```

y dejar la racha actual dentro de la sección de corrección en vivo.

Ejemplo:

```text
Mala postura hoy
32m 14s
```

Y aparte:

```text
Cabeza adelantada
22 segundos continuos
```

---

# 7. Calibración mala demasiado estricta

## Problema

Durante `build_thresholds()`:

```python
if abs(bad_value - good_value) <= margin:
    indistinguishable.append(metric_name)
```

Después:

```python
if indistinguishable:
    raise ValueError(...)
```

Esto obliga a que todas las métricas sean suficientemente distintas entre la postura buena y mala.

Actualmente se evalúan aproximadamente:

```text
ear_shoulder_dx
nose_shoulder_dx
chin_drop
torso_lean_dx
```

Pero una mala postura real no necesariamente altera las cuatro métricas.

## Ejemplo

Una persona puede presentar:

```text
Cabeza adelantada       → cambio fuerte
Cuello adelantado       → cambio fuerte
Mentón                   → casi sin cambio
Torso                    → casi sin cambio
```

Esa calibración sigue siendo válida.

Sin embargo, actualmente puede rechazarse.

## Solución propuesta

Permitir tres estados para cada métrica:

```text
directional
absolute
disabled
```

### `directional`

Existe diferencia clara entre postura buena y mala.

### `absolute`

Solo existe calibración buena y se utiliza una tolerancia respecto del baseline.

### `disabled`

La métrica no aporta información discriminante para ese perfil.

## Resultado esperado

Una persona podría terminar con algo como:

```text
ear_shoulder_dx  → directional
nose_shoulder_dx → directional
chin_drop        → disabled
torso_lean_dx    → disabled
```

Mientras otra persona podría tener:

```text
ear_shoulder_dx  → absolute
nose_shoulder_dx → directional
chin_drop        → directional
torso_lean_dx    → directional
```

Esto haría que la calibración sea realmente personal.

## Validación mínima

En lugar de exigir que todas las métricas sean discriminantes:

```text
aceptar calibración si existen N métricas útiles
```

Por ejemplo:

```text
mínimo 2 métricas discriminantes
```

o adaptar automáticamente `posture_min_bad_metrics` al número de métricas habilitadas.

---

# 8. Validación incompleta de landmarks

## Problema

En `extract_metrics()` actualmente se valida:

```python
required = [ear, shoulder, nose]
```

Pero también se utiliza:

```python
hip
```

para:

```python
"torso_lean_dx": shoulder.x - hip.x
```

Sin verificar:

```python
hip.visibility
```

## Consecuencia

Una cadera parcialmente oculta o mal detectada puede producir un valor aparentemente válido y generar:

```text
Torso inclinado
```

cuando en realidad el landmark no es confiable.

## Solución

Agregar la cadera al conjunto requerido cuando se utilice esa métrica.

Por ejemplo:

```python
required = [ear, shoulder, hip, nose]
```

O, mejor todavía, permitir métricas parciales:

```text
si hip no es visible:
    no calcular torso_lean_dx

si ear no es visible:
    no calcular ear_shoulder_dx

etc.
```

Esto permitiría que la clasificación funcione con los landmarks disponibles en lugar de invalidar el frame completo.

---

# 9. Mejorar las métricas posturales

Actualmente varias métricas están basadas en diferencias de coordenadas normalizadas:

```python
ear.x - shoulder.x
nose.x - shoulder.x
shoulder.x - hip.x
```

Esto funciona razonablemente gracias a la calibración, pero sigue siendo sensible a:

```text
distancia a cámara
posición dentro del frame
zoom
altura de cámara
rotación del cuerpo
ángulo de visión
```

## Evolución recomendada

Migrar progresivamente hacia:

### Ángulos

Ejemplos:

```text
ángulo oreja-hombro-cadera
ángulo hombro-cadera respecto de vertical
ángulo nariz-oreja-hombro
```

### Ratios normalizados por cuerpo

Ejemplo:

```text
desplazamiento horizontal cabeza
/
longitud hombro-cadera
```

Esto reduce la dependencia de la escala.

## MediaPipe world landmarks

También debería evaluarse si:

```text
pose_world_landmarks
```

produce resultados más estables que las coordenadas 2D para este caso.

No conviene migrar directamente sin medir.

Primero deberían realizarse pruebas comparativas con:

```text
2D landmarks
world landmarks
ángulos 2D
ratios corporales
```

sobre varias sesiones y posiciones de cámara.

---

# 10. Problema con suspensión y gaps temporales

## Problema

Actualmente:

```python
last_frame_ts = time.monotonic()
...
dt_seconds = now_ts - last_frame_ts
```

`dt_seconds` se utiliza para:

```text
tiempo trabajado
tiempo de postura
racha de mala postura
pausas
modo foco
analytics
```

Si la computadora queda suspendida, bloqueada o el proceso se congela temporalmente, el siguiente ciclo puede tener un `dt_seconds` enorme.

## Ejemplo

PC suspendida durante 40 minutos:

```text
dt_seconds ≈ 2400
```

Al volver podrían sumarse:

```text
+40 minutos de trabajo
+40 minutos de postura
break inmediato
mala postura acumulada artificialmente
```

## Solución

Detectar gaps anormales:

```python
if dt_seconds > MAX_FRAME_GAP_SECONDS:
    reset_temporal_state()
    dt_seconds = 0.0
```

Otra posibilidad:

```python
expected_dt = 1.0 / target_fps
dt_seconds = min(dt_seconds, expected_dt * 2.5)
```

La primera opción es conceptualmente mejor porque diferencia:

```text
frame lento
```

de:

```text
sesión interrumpida
```

## Recomendación

Agregar un evento interno:

```text
SESSION_GAP
```

y resetear:

```text
bad_posture_streak
away_streak
back_streak
smoother
temporizadores dependientes de continuidad
```

sin necesariamente perder el trabajo acumulado previo.

---

# 11. Control de FPS incorrecto

## Problema

Actualmente se mide:

```python
now_ts = time.monotonic()
```

después de haber realizado parte importante del procesamiento:

```text
cap.read
flip
cvtColor
MediaPipe inference
extract_metrics
```

Luego:

```python
elapsed = time.monotonic() - now_ts
sleep_time = frame_interval - elapsed
```

Esto significa que el sleep solo contempla una fracción del loop.

Por lo tanto:

```text
target_fps = 15
```

no garantiza realmente 15 FPS.

## Solución

Mover el timestamp al comienzo del ciclo:

```python
while ...:
    frame_started_at = time.monotonic()

    ...
    capture
    inference
    processing
    analytics
    rendering
    ...

    elapsed = time.monotonic() - frame_started_at
    sleep_time = frame_interval - elapsed
```

## Métricas de rendimiento recomendadas

En modo debug sería útil medir:

```text
capture_ms
inference_ms
processing_ms
render_ms
loop_ms
actual_fps
```

Esto permitiría decidir objetivamente entre:

```text
Pose Heavy
Pose Full
Pose Lite
```

y evaluar el consumo de CPU.

---

# 12. `main.py` concentra demasiadas responsabilidades

Actualmente `main.py` contiene o coordina:

```text
captura de cámara
MediaPipe
detección
calibración
smoothing
posture classification
timers
modo foco
breaks
notificaciones
analytics
persistencia
estado UI
overlay
tray
ventana OpenCV
lifecycle
resources
threads
```

Todavía es manejable, pero ya está llegando al punto donde una modificación en un comportamiento puede afectar otro.

## Refactor recomendado

Extraer una única pieza central:

```text
PostureEngine
```

No hace falta fragmentar el proyecto en decenas de clases.

La idea sería que el engine reciba estado observable:

```python
result = engine.update(
    metrics=metrics,
    has_pose=has_pose,
    dt=dt_seconds,
    now=now_ts,
)
```

Y devuelva un resultado estructurado.

Por ejemplo:

```python
EngineResult(
    posture_bad=False,
    posture_score=94.0,
    issue_type=None,
    break_due=False,
    break_progress=0.34,
    events=[],
)
```

## Eventos posibles

```text
POSTURE_BAD_STARTED
POSTURE_RECOVERED
POSTURE_ALERT
BREAK_DUE
BREAK_REMINDER
BREAK_COMPLETED
WORK_RESET
FOCUS_STARTED
FOCUS_ENDED
CALIBRATION_COMPLETED
SESSION_GAP
```

## Beneficio

`_camera_worker()` pasaría a encargarse básicamente de:

```text
capturar frame
ejecutar MediaPipe
extraer métricas
llamar PostureEngine
procesar eventos
actualizar UI
guardar datos
```

El comportamiento central podría probarse sin:

```text
webcam
MediaPipe
Tkinter
OpenCV
threads
```

---

# 13. Tests: cobertura actual y faltantes

## Lo que actualmente está cubierto

Existen tests para:

```text
BreakManager
calibration thresholds
Config
posture classification
model fetch
AnalyticsStore
```

Esto está bien para el tamaño del proyecto.

## El problema

Los tests están enfocados principalmente en módulos individuales.

No existen suficientes pruebas de integración entre:

```text
BreakManager + focus mode
cooldown + notifications
analytics + temporal samples
worker + suspension
worker + camera error
calibration + commands
engine + persistence
```

## Ejemplo real

El bug del repeat en modo foco puede pasar con toda la suite verde porque:

```text
BreakManager funciona correctamente
```

pero:

```text
main.py interpreta incorrectamente el evento
```

## Tests recomendados tras extraer `PostureEngine`

### Pausas

```text
trabaja 45 min
→ BREAK_DUE

permanece visible
→ reminder después de cooldown

modo foco
→ reminder usa cooldown de foco

sale de cámara 90 s
→ BREAK_COMPLETED
```

### Postura

```text
mala postura durante 19 s
→ sin alerta

mala postura durante 21 s
→ POSTURE_ALERT

corrige postura
→ streak reset
```

### Suspensión

```text
dt = 1800 s
→ SESSION_GAP
→ no sumar 1800 s
```

### Cooldown

```text
alerta
10 s después sigue mal
→ no alertar

cooldown cumplido
→ nueva alerta
```

### Calibración

```text
2 métricas discriminantes
2 métricas inútiles
→ calibración válida
```

---

# 14. Excepciones y robustez del worker

La aplicación se empaqueta con:

```python
console=False
```

Por lo tanto, depender de:

```python
print(...)
```

para reportar errores no es suficiente.

## Riesgos actuales

Pueden fallar:

```text
load_calibration
SQLite
MediaPipe initialization
camera initialization
camera read
filesystem
notifications
UI
```

Una excepción no controlada dentro del worker puede dejar la aplicación en un estado ambiguo.

## Solución

Agregar un estado de error estructurado:

```text
status = "error"
error_code
error_message
```

Ejemplos:

```text
CAMERA_UNAVAILABLE
MODEL_MISSING
MODEL_INIT_FAILED
DATABASE_ERROR
CALIBRATION_CORRUPTED
CONFIG_INVALID
```

## UI

Mostrar algo como:

```text
No se pudo abrir la cámara.

Cámara configurada: 0

[Reintentar]
[Cambiar cámara]
[Abrir diagnóstico]
```

---

# 15. Logging

Implementar logging real.

## Propuesta

Usar:

```python
logging
logging.handlers.RotatingFileHandler
```

Guardar:

```text
logs/posturebreaker.log
```

Con rotación, por ejemplo:

```text
5 MB
3 backups
```

## Eventos útiles

```text
app_started
app_closed
camera_opened
camera_failed
camera_reconnected
model_loaded
calibration_loaded
calibration_saved
calibration_failed
database_opened
database_error
session_gap
worker_crashed
```

No hace falta registrar cada frame.

---

# 16. Ubicación de los datos

Actualmente en modo empaquetado los datos se guardan cerca del `.exe`.

Eso funciona si el programa está en una carpeta de usuario, pero puede fallar si algún día se instala en:

```text
C:\Program Files\PostureBreaker\
```

## Recomendación

Usar una carpeta específica de usuario.

En Windows:

```text
%LOCALAPPDATA%\PostureBreaker\
```

Ejemplo:

```text
PostureBreaker/
  config.json
  posture_calibration.json
  posture_guard.db
  logs/
```

Los recursos de la aplicación pueden permanecer junto al `.exe`:

```text
pose_landmarker.task
assets
```

## Librería recomendada

Podría utilizarse:

```text
platformdirs
```

para resolver correctamente las carpetas según el sistema operativo.

---

# 17. Migraciones SQLite

Actualmente `_ensure_schema()` utiliza:

```sql
CREATE TABLE IF NOT EXISTS
```

Esto funciona mientras las tablas no cambien.

Pero no permite modificar correctamente instalaciones existentes.

Por ejemplo:

```text
agregar columna
renombrar columna
cambiar índice
crear nueva relación
```

## Solución simple

No hace falta incorporar Alembic.

Usar:

```sql
PRAGMA user_version;
```

Ejemplo:

```text
schema v1
↓
schema v2
↓
schema v3
```

Pseudo flujo:

```python
version = get_schema_version()

if version < 1:
    migrate_v1()

if version < 2:
    migrate_v2()

if version < 3:
    migrate_v3()
```

Finalmente:

```sql
PRAGMA user_version = 3;
```

---

# 18. Recuperación automática de cámara

Actualmente, si:

```python
cap.read()
```

falla:

```python
status = "camera_error"
break
```

El worker termina.

Pero una webcam puede fallar temporalmente por:

```text
driver
USB
sleep/wake
otra aplicación
cambio de dispositivo
Windows
```

## Recomendación

Implementar retry/backoff.

Ejemplo:

```text
fallo
↓
esperar 1 s
↓
reintentar
↓
2 s
↓
5 s
↓
10 s máximo
```

Mientras:

```text
status = camera_reconnecting
```

Y permitir:

```text
Cambiar cámara
```

desde el dashboard.

---

# 19. Concurrencia y comandos del tray

`SharedState` utiliza correctamente un lock.

Sin embargo, `TrayIcon` mantiene flags como:

```text
_quit_requested
_recalibrate_good
_recalibrate_bad
_toggle_camera_requested
_toggle_focus_requested
```

que son escritos desde el thread de pystray y leídos desde otro thread.

En CPython normalmente funcionará debido al GIL, pero no es el mecanismo correcto para comunicación de eventos.

## Solución

Usar:

```python
queue.Queue
```

Por ejemplo:

```python
command_queue.put(Command.TOGGLE_CAMERA)
```

y el loop principal:

```python
while not command_queue.empty():
    command = command_queue.get()
```

Esto evita:

```text
eventos perdidos
flags simultáneos
condiciones de carrera
```

También podría reemplazarse parte de los booleanos de `SharedState`.

---

# 20. Configuración desde UI

Actualmente la configuración requiere editar:

```text
posture_break_guard.config.json
```

Esto está bien durante desarrollo.

Para una aplicación de escritorio conviene incorporar settings.

## Ajustes candidatos

```text
cámara
intervalo de pausas
duración de pausa
sensibilidad
cooldown postura
FPS
modelo
notificaciones
sonido
autostart
modo foco
```

No todos deben mostrarse inicialmente.

## Nivel usuario

Mostrar:

```text
Cámara
Sensibilidad
Intervalo entre pausas
Duración de pausa
Sonidos
Inicio automático
```

## Nivel avanzado

Mantener:

```text
min_visibility
smoothing_window
margins
target_fps
analytics sample rate
```

---

# 21. Selector de cámara

`camera_index` actualmente debe configurarse manualmente.

Debería existir un selector.

Ejemplo:

```text
Cámara
[ Logitech C920                 ▼ ]

Vista previa

[ Probar cámara ]
```

Si OpenCV no expone nombres fácilmente en todos los sistemas, al menos:

```text
Cámara 0
Cámara 1
Cámara 2
```

con preview.

---

# 22. Wizard de calibración

La calibración podría mejorar mucho desde UX.

## Flujo sugerido

### Paso 1

```text
Colocá la cámara de costado o semi-costado.
```

### Paso 2

```text
Sentate como querés trabajar normalmente.

[Iniciar calibración buena]
```

Countdown:

```text
3
2
1
```

### Paso 3

Mostrar calidad:

```text
Buena calibración

Estabilidad: 94 %
Landmarks visibles: OK
```

### Paso 4

```text
Ahora adoptá tu postura problemática habitual.

[Calibrar postura mala]
```

### Paso 5

Resultado:

```text
Perfil creado

Métricas detectadas:
✓ Cabeza adelantada
✓ Cuello adelantado
— Mentón sin diferencia clara
— Torso sin diferencia clara
```

Esto encaja particularmente bien con el sistema de métricas `disabled`.

---

# 23. Score y cantidad mínima de datos

Actualmente el score inicial puede ser:

```text
100 %
```

sin datos.

Desde una perspectiva matemática es válido como default interno, pero visualmente puede implicar que el usuario tiene postura perfecta antes de haber medido nada.

## Mejora

Mostrar:

```text
--
```

o:

```text
Sin datos
```

hasta acumular un mínimo.

Ejemplo:

```text
mínimo 60 segundos de postura válida
```

Después activar el score.

---

# 24. Racha de días

Actualmente una racha puede computarse con muy poco tiempo de datos diarios.

Existe una validación para hoy:

```text
total > 60 segundos
```

pero un día con apenas uno o dos minutos puede contar igual que una jornada completa.

## Mejora

Agregar:

```text
minimum_daily_tracking_seconds
```

Por ejemplo:

```text
30 minutos
```

Para que un día cuente para la racha:

```text
tracked_time >= mínimo
score >= objetivo
```

También conviene hacer configurable:

```text
target_score = 70 %
```

---

# 25. Pausas

El sistema actual considera pausa cuando el usuario desaparece de cámara.

Esto funciona bien como heurística simple.

Pero hay casos:

```text
usuario se reclina fuera del frame
usuario tapa cámara
usuario se levanta pero sigue visible
webcam pierde tracking
```

## Evolución futura

Podría diferenciar:

```text
no_pose
away
standing
camera_problem
```

No es necesario para una primera versión, pero permitiría validar pausas más correctamente.

---

# 26. Break Coach

La idea actual de rotar rutinas es buena.

Actualmente existen ejercicios como:

```text
Soltá hombros
Ojos lejos
Cuello y espalda
```

Podría evolucionar hacia categorías:

```text
visual
cervical
hombros
espalda
movilidad
respiración
```

y evitar repetir la misma categoría consecutivamente.

También podría existir:

```text
pausa corta
pausa normal
pausa larga
```

según tiempo acumulado.

---

# 27. Modo foco

El modo foco tiene potencial, pero debería formalizarse mejor.

Actualmente altera:

```text
posture threshold
cooldown
break repeat
overlay
```

Debería definirse como una política explícita.

Por ejemplo:

```python
FocusPolicy(
    posture_threshold_multiplier=2.0,
    posture_cooldown_multiplier=2.0,
    break_reminder_seconds=180,
    use_soft_overlay=True,
)
```

Esto evita repartir lógica condicional por `main.py`.

---

# 28. Single instance

La aplicación debería evitar múltiples instancias.

Actualmente un usuario podría abrir:

```text
PostureGuard.exe
PostureGuard.exe
PostureGuard.exe
```

y terminar con:

```text
múltiples cámaras
múltiples trays
múltiples conexiones SQLite
múltiples alertas
```

## Recomendación

Implementar single-instance lock.

En Windows puede utilizarse:

```text
named mutex
```

o un lock file robusto.

La segunda instancia debería:

```text
detectar la existente
traer la ventana al frente
cerrarse
```

---

# 29. Privacidad

El producto tiene una característica positiva:

```text
procesamiento completamente local
```

Esto debería mantenerse y documentarse.

No se almacenan imágenes de la webcam.

La base debería guardar solamente:

```text
métricas
scores
eventos
estadísticas
```

## README

Agregar una sección explícita:

```text
Privacidad

PostureBreaker procesa la cámara localmente.
Las imágenes no se almacenan ni se transmiten.
```

También podría existir:

```text
Borrar historial
Exportar historial
```

---

# 30. Retención de analytics

`posture_samples` puede crecer indefinidamente.

Con una muestra cada aproximadamente 10 segundos:

```text
6 muestras/min
360/hora
~2880 por jornada de 8 h
```

En un año puede superar fácilmente cientos de miles de registros.

SQLite puede manejarlo, pero no hay necesidad de conservar detalle infinito.

## Política recomendada

Por ejemplo:

```text
samples detallados → 30 días
daily_stats → permanente
events → 180 días
```

O consolidar datos históricos.

---

# 31. Índices de SQLite

Actualmente existen índices temporales básicos.

A medida que crezca analytics conviene agregar índices compuestos según consultas.

Por ejemplo:

```sql
CREATE INDEX idx_samples_ts_pose
ON posture_samples(ts, has_pose);
```

y posiblemente:

```sql
CREATE INDEX idx_posture_events_error_ts
ON posture_events(error_type, ts);
```

No es urgente mientras el volumen sea bajo.

---

# 32. Packaging

Actualmente existen:

```text
PostureGuard.spec
build_exe.ps1
```

pero ambos contienen configuración de build.

Eso genera dos fuentes de verdad.

## Recomendación

Usar:

```text
PostureGuard.spec
```

como fuente principal.

Y que:

```powershell
build_exe.ps1
```

solamente haga:

```text
verificar Python
instalar/verificar dependencias
descargar/verificar modelo
ejecutar tests
ejecutar PyInstaller usando PostureGuard.spec
crear artifact
```

Ejemplo:

```powershell
python scripts/fetch_model.py --variant full
python -m unittest discover -s tests -t . -v
python -m PyInstaller --noconfirm --clean PostureGuard.spec
```

---

# 33. Modelo dentro del build

Actualmente el modelo:

```text
pose_landmarker.task
```

se incluye mediante PyInstaller y también vuelve a copiarse manualmente.

Conviene elegir una sola estrategia.

Si el modelo debe poder reemplazarse fácilmente:

```text
dejarlo externo junto al ejecutable
```

puede ser la mejor opción.

Así el usuario puede cambiar:

```text
lite
full
heavy
```

sin recompilar.

---

# 34. Dependencias

Actualmente:

```text
mediapipe
opencv-contrib-python
numpy
customtkinter
```

## OpenCV

No se observó uso de módulos específicos de:

```text
opencv-contrib
```

Las funcionalidades utilizadas son principalmente:

```text
VideoCapture
cvtColor
flip
circle
line
putText
imshow
waitKey
```

Por lo tanto debería evaluarse reemplazar:

```text
opencv-contrib-python
```

por:

```text
opencv-python
```

Esto reduciría tamaño de dependencia y posiblemente del ejecutable.

Debe validarse mediante tests y build antes de cambiarlo.

## Python

Conviene declarar explícitamente las versiones soportadas.

Ejemplo inicial:

```text
Python 3.11–3.12
```

y probar mediante CI.

---

# 35. `pyproject.toml`

El proyecto debería evolucionar de:

```text
requirements.txt
```

como única definición a:

```text
pyproject.toml
```

Puede seguir generándose o manteniéndose un requirements si resulta cómodo.

El `pyproject.toml` permitiría definir:

```text
nombre
versión
Python mínimo
dependencias
dependencias opcionales
Ruff
typing
metadata
```

Ejemplo conceptual:

```toml
[project]
name = "posturebreaker"
version = "0.1.0"
requires-python = ">=3.11,<3.13"

dependencies = [
    "mediapipe==...",
    "opencv-python==...",
    "numpy==...",
    "customtkinter==..."
]

[project.optional-dependencies]
desktop = [
    "pystray==...",
    "Pillow==...",
    "plyer==..."
]
```

---

# 36. CI con GitHub Actions

Actualmente el repositorio no dispone de workflows.

Agregar CI debería ser una prioridad relativamente cercana porque la suite de tests ya existe.

## Workflow inicial

Ejecutar en:

```text
Windows
Python 3.11
Python 3.12
```

Pasos:

```text
checkout
setup-python
install dependencies
run tests
Ruff
build opcional
```

## Build

Podría existir un workflow separado:

```text
release.yml
```

que genere:

```text
PostureBreaker-Windows-x64.zip
```

cuando se crea un tag.

---

# 37. Linting y calidad

Agregar:

```text
Ruff
```

para:

```text
lint
imports
formatting
unused code
```

No hace falta introducir demasiadas herramientas.

Una combinación suficiente sería:

```text
Ruff
unittest
mypy o pyright opcional
```

---

# 38. README

Actualmente existe:

```text
README.md
README_posture_break_guard.md
```

El README principal es demasiado pequeño.

Debería transformarse en el documento principal.

## Secciones recomendadas

```text
PostureBreaker
Descripción
Capturas
Características
Cómo funciona
Privacidad
Requisitos
Instalación
Descarga del modelo
Ejecución
Configuración
Calibración
Modo foco
Pausas
Analytics
Build
Tests
Limitaciones
Roadmap
Licencia
```

`README_posture_break_guard.md` podría eliminarse posteriormente para evitar duplicación.

---

# 39. Naming

Actualmente aparecen ambos nombres:

```text
PostureBreaker
Posture Guard
PostureGuard
posture_break_guard
```

Conviene definir una única identidad.

## Recomendación

Producto:

```text
PostureBreaker
```

Ejecutable:

```text
PostureBreaker.exe
```

Paquete Python:

```text
posture_breaker
```

Base de datos:

```text
posturebreaker.db
```

Config:

```text
config.json
```

Esto simplifica documentación y distribución.

No es urgente, pero cuanto antes se estabilice el nombre, menos costo tendrá.

---

# 40. Roadmap recomendado

## Ronda 1 — Correctness

Objetivo:

> corregir errores funcionales antes de agregar nuevas características.

### Tareas

- corregir `break_alert_repeat` en modo foco;
- separar timestamps de reminders;
- corregir `hourly_trend`;
- mostrar mala postura diaria correctamente;
- detectar gaps grandes de `dt_seconds`;
- corregir cálculo real de FPS;
- validar visibilidad de hip;
- agregar tests para todos estos casos.

### Resultado esperado

Analytics confiable y timers robustos.

---

# 41. Ronda 2 — Calibración y detección

Objetivo:

> mejorar precisión y reducir falsos positivos.

### Tareas

- soportar métricas `disabled`;
- no exigir separación de todas las métricas;
- adaptar `posture_min_bad_metrics`;
- estudiar ratios normalizados;
- estudiar ángulos posturales;
- evaluar MediaPipe world landmarks;
- agregar quality score de calibración;
- agregar tests de `extract_metrics`;
- mejorar smoothing según disponibilidad de landmarks.

### Resultado esperado

Calibración más personal y detección más estable.

---

# 42. Ronda 3 — PostureEngine

Objetivo:

> extraer lógica temporal de `main.py`.

### Tareas

Crear:

```text
posture_guard/engine.py
```

Responsable de:

```text
posture state
bad streak
cooldowns
break state
focus policy
events
temporal transitions
```

Agregar eventos tipados.

Reducir `_camera_worker()` a infraestructura.

### Resultado esperado

Mayor testabilidad y menor acoplamiento.

---

# 43. Ronda 4 — Robustez desktop

Objetivo:

> convertir el programa en una aplicación que pueda permanecer ejecutándose durante horas o días.

### Tareas

- logging rotativo;
- error boundary del worker;
- estados de error;
- retry de cámara;
- manejo sleep/wake;
- single instance;
- datos en AppData;
- migraciones SQLite;
- graceful shutdown;
- recuperación de DB/calibration corrupta.

### Resultado esperado

Aplicación tolerante a fallos reales del escritorio.

---

# 44. Ronda 5 — UX y configuración

Objetivo:

> reducir dependencia de editar archivos manualmente.

### Tareas

- selector de cámara;
- settings;
- wizard de calibración;
- sensibilidad;
- configuración de pausas;
- autostart;
- minimize-to-tray;
- snooze;
- borrar historial;
- exportar datos;
- diagnóstico.

### Resultado esperado

Aplicación utilizable por una persona sin conocimiento técnico.

---

# 45. Ronda 6 — Ingeniería y distribución

Objetivo:

> convertir el repositorio en un proyecto reproducible.

### Tareas

- `pyproject.toml`;
- versión formal;
- Ruff;
- typing;
- GitHub Actions;
- build Windows reproducible;
- GitHub Releases;
- README completo;
- licencia;
- changelog.

### Resultado esperado

Cada versión puede construirse, probarse y distribuirse automáticamente.

---

# 46. Mejoras futuras opcionales

Una vez estabilizada la base podrían evaluarse características más avanzadas.

## Perfiles

```text
Trabajo
Gaming
Lectura
Modo foco
```

Cada uno con:

```text
intervalos
sensibilidad
tipo de pausas
```

## Score por problema

En lugar de un único score:

```text
Cabeza     84 %
Cuello     91 %
Torso      76 %
Global     83 %
```

## Tendencias

Mostrar:

```text
principal problema de la semana
horario con peor postura
promedio por jornada
tiempo sentado
cumplimiento de pausas
```

## Recomendaciones

Ejemplo:

```text
Esta semana tu error más frecuente fue cabeza adelantada.

Apareció principalmente entre las 10:00 y 12:00.
```

Debe mantenerse como feedback ergonómico y no presentarse como diagnóstico médico.

---

# 47. Cambios que NO se recomiendan

## No reescribir el proyecto

La arquitectura base es suficientemente buena.

Una reescritura completa agregaría riesgo sin resolver los problemas principales.

## No cambiar Python

Python es apropiado porque el proyecto depende fuertemente de:

```text
MediaPipe
OpenCV
procesamiento de imagen
desktop utility
```

## No introducir un backend

La aplicación es esencialmente local.

Agregar:

```text
NestJS
API
PostgreSQL
servidor
```

no aporta valor actualmente.

## No introducir arquitectura empresarial innecesaria

No hacen falta:

```text
CQRS
event sourcing
microservicios
repository pattern para cada tabla
dependency injection compleja
```

La separación recomendada es simple:

```text
infrastructure
engine
storage
UI
```

---

# 48. Arquitectura objetivo

Una posible estructura después de varias rondas:

```text
posture_breaker/
  app.py
  config.py
  models.py

  engine/
    posture_engine.py
    break_manager.py
    focus_policy.py
    events.py

  vision/
    detector.py
    metrics.py
    calibration.py
    smoothing.py

  infrastructure/
    camera.py
    notifications.py
    paths.py
    logging.py
    single_instance.py

  storage/
    database.py
    migrations.py
    analytics.py

  ui/
    dashboard.py
    overlay.py
    tray.py

tests/
  unit/
  integration/
```

No hace falta llegar inmediatamente a esta estructura.

Debe evolucionar gradualmente según las rondas propuestas.

---

# 49. Orden recomendado de implementación

El orden más eficiente sería:

```text
1. Bugs funcionales
2. Analytics
3. Calibración
4. PostureEngine
5. Robustez
6. UX
7. Packaging/CI
8. Features adicionales
```

Evitaría empezar ahora con:

```text
más gráficos
más rutinas
más configuraciones
nuevos modos
```

antes de corregir los problemas temporales y de analytics.

---

# 50. Criterios para considerar estable la siguiente versión

Una versión base estable debería cumplir al menos:

### Detección

- no produce errores con landmarks parcialmente visibles;
- calibración buena/mala puede completarse de forma consistente;
- las métricas inútiles no invalidan todo el perfil.

### Timers

- suspensión del sistema no agrega tiempo falso;
- los breaks ocurren según tiempo real trabajado;
- los cooldowns funcionan igual en modo normal y foco.

### Analytics

- score diario representa tiempo bueno/malo real;
- trend horario representa cada hora real;
- las alertas registradas coinciden con las alertas enviadas.

### Cámara

- fallo temporal no mata permanentemente la aplicación;
- existe feedback visible de error.

### Persistencia

- DB puede evolucionar mediante migraciones;
- datos se guardan en ubicación escribible;
- cierre de aplicación hace flush correctamente.

### Tests

- engine cubierto;
- breaks cubiertos;
- focus mode cubierto;
- suspensión cubierta;
- analytics horario cubierto;
- calibración cubierta.

### Distribución

- build reproducible;
- CI verde;
- versión identificable;
- artifact generado automáticamente.

---

# 51. Conclusión

PostureBreaker tiene una base técnica correcta y no necesita una reescritura.

Las mejoras con mayor retorno inmediato no son nuevas funcionalidades sino corregir algunos comportamientos internos:

1. repeat de pausa en modo foco;
2. analytics horario;
3. manejo de gaps temporales;
4. control real de FPS;
5. visibilidad de landmarks;
6. calibración parcialmente discriminante.

Después de eso, el cambio arquitectónico con mayor valor sería extraer un:

```text
PostureEngine
```

que concentre la lógica temporal y produzca eventos independientes de cámara, MediaPipe y UI.

Ese refactor permitiría que las reglas más importantes del producto puedan probarse completamente sin hardware ni interfaz gráfica.

A partir de ahí el proyecto puede evolucionar hacia una utilidad de escritorio sólida, con:

```text
calibración guiada
analytics confiable
reconexión automática
settings
tray
autostart
logs
migraciones
CI
releases
```

sin abandonar la arquitectura simple y local que actualmente es una de sus principales ventajas.