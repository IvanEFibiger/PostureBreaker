# PostureBreaker

Monitor de postura en tiempo real con webcam. Detecta mala postura, recuerda pausas y muestra estadísticas. Todo el procesamiento es **local**.

- **Detección**: MediaPipe Pose (modelo preentrenado) sobre la webcam.
- **Clasificación**: umbrales geométricos calibrados a *tu* postura (no ML genérico).
- **Pausas**: break coach con rutinas rotativas.
- **Modo foco**: alertas suaves para no interrumpirte.
- **Analytics**: score diario, tendencia horaria y semanal, errores más repetidos.
- **Privacidad**: la cámara no se graba ni se transmite. Nada sale de tu máquina.

## Cómo funciona

1. MediaPipe detecta 33 landmarks del cuerpo desde la cámara.
2. Se calculan métricas posturales (cabeza/cuello/mentón/torso) normalizadas por la calibración.
3. La lógica temporal (`PostureEngine`) decide estados, alertas, cooldowns y pausas.

Más detalle técnico y el roadmap en [`docs/`](docs/).

## Requisitos

- Python **3.11 a 3.13**. En **3.14 no arranca**: `mediapipe==0.10.33` revienta al invocar Python desde su thread pool sin el GIL (`Fatal Python error: PyEval_RestoreThread`). Por eso `pyproject.toml` fija `requires-python = ">=3.11,<3.14"`.
- Una webcam

## Instalación

Recomendado un entorno virtual en Python 3.13 (evita el `python` global si es 3.14):

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[desktop]"
```

O con los requirements (núcleo + opcionales):

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt   # bandeja + notificaciones
```

- `customtkinter`: dashboard visual (sin él corre en modo headless)
- `pystray` + `Pillow`: ícono en la bandeja
- `plyer`: notificaciones toast del sistema

## Descarga del modelo

```bash
python scripts/fetch_model.py --variant heavy
```

Variantes (`--variant`): `heavy` (default, ~29 MB), `full` (~9 MB), `lite` (~5,8 MB). El script verifica el **SHA256** contra el CDN oficial de MediaPipe y no vuelve a bajar si ya es válido.

## Ejecución

Con el entorno virtual (recomendado):

```bash
.venv\Scripts\python posture_break_guard.py
```

O con `python posture_break_guard.py` si tu intérprete es 3.11–3.13.

Se abre el dashboard con score, estadísticas, controles de calibración, selector de cámara, snooze y herramientas de datos.

## Herramientas de escritorio

- **Inicio automático**: toggle en el dashboard (entrada `Run` de Windows por usuario).
- **Minimizar a bandeja**: cerrar la ventana la oculta; se sale desde el menú del tray.
- **Snooze**: silencia las alertas durante `snooze_minutes`.
- **Diagnóstico**: genera `diagnostics.txt` con versión, Python, modelo, cámaras y rutas.
- **Selector de cámara**, **exportar** y **borrar historial** desde el panel.
- **Ajustes**: sliders de sensibilidad, intervalo y duración de pausas y segundos para alertar (se validan y persisten al aplicar).

## Calibración

1. Poné la cámara de costado o semi-costado.
2. **Calibrar buena**: sentate como trabajás normalmente.
3. Opcional **Calibrar mala**: adoptá tu postura problemática.
4. El perfil indica la calidad (%) y qué métricas detecta/ignora: las métricas que no distinguen bien/mala quedan `disabled` en vez de invalidar todo.

> La calibración guarda un `geometry_version`. Si cambia el pipeline de landmarks (p. ej. el fix de espejado), las calibraciones viejas se consideran incompatibles y piden recalibrar; no se migran valores.

## Validación V2 (observe-only)

El clasificador **V1 sigue siendo el productivo**. En paralelo corre un conjunto de señales **V2 en modo observación** (`posture_v2_observe_only = true`): se calculan, se muestran en el overlay y se guardan, pero **no disparan alertas**.

- **Runner guiado**: en el dashboard → *Validación detector V2* → *Ejecutar test V2*. Recorre ~12 escenarios desk (buena, cabeza adelantada/abajo/tilt, hombros, giros) con `RESET → PREPARE → CAPTURE` por escenario, y captura automáticamente.
- **Aislamiento**: durante una corrida no se escribe SQLite, no se disparan alertas ni políticas temporales, y `smoother`/engine se resetean al terminar o cancelar.
- **Datos**: cada corrida escribe `history/validation/<run_id>.jsonl` y un reporte `<run_id>-report.txt` (V2-only, con coverage `n/total`, `torso_yaw` legible y un chequeo de lateralidad L/R).
- **Comparar corridas** (reproducibilidad A/B):
  ```bash
  python -m scripts.compare_validation_runs history/validation/<A>.jsonl history/validation/<B>.jsonl
  ```

El overlay de cámara dibuja los landmarks con etiquetas anatómicas `L`/`R`. MediaPipe procesa el frame **sin espejar** y la ventana se muestra espejada solo para el usuario.

## Configuración

Se edita `posture_break_guard.config.json` (validado al arrancar; claves desconocidas o valores fuera de rango dan error claro). Tabla completa de parámetros en [`README_posture_break_guard.md`](README_posture_break_guard.md).

## Privacidad y datos

Procesamiento 100% local: las imágenes de la cámara no se almacenan ni se transmiten. La base guarda solo métricas, scores y eventos.

- Corriendo desde el código, los datos quedan en la carpeta del proyecto.
- Empaquetado (`.exe`), se guardan en `%LOCALAPPDATA%\PostureBreaker\` (config, calibración, base y logs); los recursos quedan junto al ejecutable.
- Log rotativo en `logs/posturebreaker.log` (5 MB, 3 backups).
- La app es single-instance.

## Tests

No requieren dependencias pesadas (ni MediaPipe ni cámara):

```bash
python -m unittest discover -s tests -t . -v
```

## Build

```bash
# Una sola vez:
pip install -r requirements-build.txt

# Build reproducible (descarga modelo, corre tests y empaqueta):
./build_exe.ps1
```

El `.spec` es la fuente de verdad del empaquetado; el modelo queda **externo** al ejecutable para poder cambiar de variante sin recompilar.

## CI / Releases

GitHub Actions corre lint (Ruff) + tests en Ubuntu y Windows (Python 3.11/3.12). Un tag `v*` dispara un build de Windows y publica `PostureBreaker-Windows-x64.zip`.

## Limitaciones

- Depende de que la cámara quede fija y con buena luz.
- Vista frontal funciona peor que lateral para cabeza adelantada.
- No reemplaza un buen puesto de trabajo ni pausas reales; es feedback ergonómico, no diagnóstico médico.

## Licencia

[MIT](LICENSE) © 2026 Ivan Fibiger.
