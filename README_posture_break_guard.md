# Posture Guard

Monitor postural con webcam e IA. Detecta mala postura, recuerda pausas y muestra estadísticas.

## Dependencias

Python **3.11 a 3.13**. En **3.14 no corre**: `mediapipe==0.10.33` falla por un cambio de GIL (`Fatal Python error: PyEval_RestoreThread`); `pyproject.toml` fija `requires-python = ">=3.11,<3.14"`.

Versiones pinneadas en `requirements.txt`:

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Opcionales** (mejoran la experiencia pero no son obligatorias):

```bash
pip install -r requirements-optional.txt
```

- `customtkinter`: dashboard visual (sin él corre en modo headless)
- `pystray` + `Pillow`: ícono en bandeja del sistema
- `plyer`: notificaciones nativas del SO (toast)

## Modelo

Necesita `pose_landmarker.task` de MediaPipe. Descargalo y verificá su integridad con:

```bash
python scripts/fetch_model.py --variant heavy
```

Variantes (`--variant`): `heavy` (default, ~29 MB), `full` (~9 MB), `lite` (~5,8 MB). El script verifica el SHA256 contra el CDN oficial y no vuelve a bajar el archivo si ya es válido. Dejalo en la carpeta raíz o configurá `model_path` en `posture_break_guard.config.json`.

## Estructura

```
posture_break_guard.py              ← entry point
posture_break_guard.config.json     ← configuración
posture_calibration.json            ← se genera al calibrar
requirements.txt                    ← dependencias núcleo (pinneadas)
requirements-optional.txt           ← bandeja + notificaciones (opcionales)
scripts/
  fetch_model.py              ← descarga + verifica el modelo
  analyze_snapshots.py        ← analiza snapshots de debug manuales
  compare_validation_runs.py  ← compara dos corridas de validación V2
tests/              ← suite de tests (unittest)
posture_guard/
  config.py         ← Config + validación + carga
  models.py         ← dataclasses compartidos
  engine.py         ← motor temporal (postura, pausas, foco) + eventos
  timing.py         ← detección de gaps temporales
  calibration.py    ← calibración + thresholds
  detection.py      ← landmarks + métricas + clasificación
  validation.py     ← runner guiado de validación V2 + reportes
  guides.py         ← geometría del overlay (grupos, transform display)
  alerts.py         ← pausas
  notifications.py  ← notificaciones nativas + sonido
  storage.py        ← SQLite: analytics, migraciones y recuperación
  paths.py          ← datos de usuario (AppData) y recursos
  logging_setup.py  ← logging rotativo a archivo
  camera.py         ← apertura de cámara con retry/backoff
  single_instance.py← instancia única (mutex de Windows)
  state.py          ← estado compartido entre threads
  ui.py             ← overlay, tray, helpers de dibujo
  dashboard.py      ← interfaz principal (CustomTkinter)
  main.py           ← orquestador (worker + dashboard)
```

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## Cómo correrlo

```bash
.venv\Scripts\python posture_break_guard.py
```

(o `python posture_break_guard.py` con un intérprete 3.11–3.13)

Se abre el **dashboard** con:
- Puntuación de postura del día (%)
- Anillo de progreso con color según score
- Racha de días consecutivos con buena postura
- Estadísticas: tiempo trabajado, pausas, alertas
- Botones: calibrar buena/mala, mostrar cámara, borrar calibración

La cámara corre en background — no necesitás tener la ventana de video abierta.

## Flujo recomendado

1. Poné la cámara fija en lateral o semi lateral
2. Abrí el programa
3. Click en **Calibrar buena** y quedate quieto unos segundos
4. Opcional: click en **Calibrar mala** y simulá tu postura mala típica
5. Dejá correr el monitor — el dashboard muestra tu score en tiempo real

## Alertas

Cuando detecta mala postura sostenida o es hora de pausa:

1. **Overlay always-on-top**: banner rojo/naranja que aparece sobre cualquier ventana
2. **Notificación toast** del sistema operativo (si tenés `plyer`)
3. **Beep sonoro** como fallback

## System tray

Si tenés `pystray` + `Pillow`, aparece un ícono en la bandeja con menú para:

- Mostrar/ocultar cámara
- Recalibrar
- Salir

Color dinámico: verde (OK), amarillo (dudoso), rojo (mala postura), naranja (pausa).

## Historial y scoring

Carpeta `history/` con un JSON por día:

- Tiempo de buena/mala postura
- Puntuación diaria (% de tiempo con buena postura)
- Racha de días consecutivos con score ≥ 70%
- Alertas y pausas completadas
- Log de eventos con hora

## Privacidad y datos

Todo el procesamiento es **local**: las imágenes de la cámara no se almacenan ni se transmiten. La base guarda solo métricas, scores y eventos.

Corriendo desde el código, los datos quedan en la carpeta del proyecto. Empaquetado (`.exe`), se guardan en `%LOCALAPPDATA%\PostureBreaker\` (config, calibración, base de datos y logs), para que funcione aunque se instale en `Program Files`; los recursos (modelo) quedan junto al ejecutable.

Log rotativo en `logs/posturebreaker.log` (5 MB, 3 backups). La app es single-instance.

## Ajustes

Editá `posture_break_guard.config.json`:

| Parámetro | Default | Descripción |
|---|---|---|
| `camera_index` | 0 | Índice de cámara |
| `min_visibility` | 0.55 | Visibilidad mínima de landmarks (0-1) |
| `sustained_bad_posture_seconds` | 20 | Segundos para disparar alerta |
| `posture_min_bad_metrics` | 2 | Métricas malas necesarias para marcarla como mala |
| `calibration_min_enabled_metrics` | 2 | Métricas discriminantes mínimas para aceptar la calibración |
| `posture_alert_cooldown_seconds` | 300 | Cooldown entre alertas |
| `break_interval_minutes` | 45 | Minutos entre pausas |
| `break_required_seconds` | 90 | Segundos para validar pausa |
| `target_fps` | 15 | FPS objetivo (reduce CPU) |
| `max_frame_gap_seconds` | 30 | Gap tratado como suspensión (no suma tiempo) |
| `snooze_minutes` | 15 | Minutos que silencia las alertas el botón snooze |
| `headless` | false | Arrancar sin ventana |
| `history_dir` | "history" | Carpeta de historial |

## Limitaciones

- Depende de que la cámara quede fija
- Si cambia mucho la iluminación, empeora
- Frontal funciona peor que lateral para cabeza adelantada
- No reemplaza mejorar el puesto de trabajo ni hacer pausas reales
