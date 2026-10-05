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

- Python **3.11+**
- Una webcam

## Instalación

```bash
pip install -r requirements.txt
```

Opcionales (bandeja del sistema y notificaciones nativas):

```bash
pip install -r requirements-optional.txt
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

```bash
python posture_break_guard.py
```

Se abre el dashboard con score, estadísticas, controles de calibración, selector de cámara, snooze y herramientas de datos.

## Calibración

1. Poné la cámara de costado o semi-costado.
2. **Calibrar buena**: sentate como trabajás normalmente.
3. Opcional **Calibrar mala**: adoptá tu postura problemática.
4. El perfil indica la calidad (%) y qué métricas detecta/ignora: las métricas que no distinguen bien/mala quedan `disabled` en vez de invalidar todo.

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
