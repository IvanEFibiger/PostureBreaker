# Posture Guard

Monitor postural con webcam e IA. Detecta mala postura, recuerda pausas y muestra estadísticas.

## Dependencias

```bash
pip install mediapipe opencv-python customtkinter
```

**Opcionales** (mejoran la experiencia pero no son obligatorias):

```bash
pip install pystray Pillow plyer
```

- `customtkinter`: dashboard visual (sin él corre en modo headless)
- `pystray` + `Pillow`: ícono en bandeja del sistema
- `plyer`: notificaciones nativas del SO (toast)

## Modelo

Necesita `pose_landmarker.task` de MediaPipe. Dejalo en la carpeta raíz o configurá la ruta en `posture_break_guard.config.json`.

## Estructura

```
posture_break_guard.py              ← entry point
posture_break_guard.config.json     ← configuración
posture_calibration.json            ← se genera al calibrar
posture_guard/
  config.py         ← Config + carga
  models.py         ← dataclasses compartidos
  calibration.py    ← calibración + thresholds
  detection.py      ← landmarks + métricas + clasificación
  alerts.py         ← pausas + historial
  notifications.py  ← notificaciones nativas + sonido
  scoring.py        ← puntuación diaria + rachas
  state.py          ← estado compartido entre threads
  ui.py             ← overlay, tray, helpers de dibujo
  dashboard.py      ← interfaz principal (CustomTkinter)
  main.py           ← orquestador (worker + dashboard)
```

## Cómo correrlo

```bash
python posture_break_guard.py
```

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

## Ajustes

Editá `posture_break_guard.config.json`:

| Parámetro | Default | Descripción |
|---|---|---|
| `camera_index` | 0 | Índice de cámara |
| `sustained_bad_posture_seconds` | 20 | Segundos para disparar alerta |
| `posture_alert_cooldown_seconds` | 300 | Cooldown entre alertas |
| `break_interval_minutes` | 50 | Minutos entre pausas |
| `break_required_seconds` | 90 | Segundos para validar pausa |
| `target_fps` | 15 | FPS objetivo (reduce CPU) |
| `headless` | false | Arrancar sin ventana |
| `history_dir` | "history" | Carpeta de historial |

## Limitaciones

- Depende de que la cámara quede fija
- Si cambia mucho la iluminación, empeora
- Frontal funciona peor que lateral para cabeza adelantada
- No reemplaza mejorar el puesto de trabajo ni hacer pausas reales
