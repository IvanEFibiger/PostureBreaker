# Changelog

Todos los cambios relevantes de este proyecto se documentan en este archivo.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/).

## [0.1.0] - 2026-10-05

Primera versión formal. Núcleo reestructurado a partir del roadmap técnico.

### Added

- `PostureEngine` con eventos tipados, independiente de cámara, MediaPipe y UI.
- Métricas de calibración con estados `directional`, `absolute` y `disabled`.
- Quality score de calibración.
- `posture_min_bad_metrics` y `calibration_min_enabled_metrics` configurables.
- Migraciones SQLite vía `PRAGMA user_version` y recuperación de DB corrupta.
- Logging rotativo a archivo.
- Datos de usuario en `%LOCALAPPDATA%` cuando corre empaquetado.
- Retry/backoff y reconexión de cámara.
- Instancia única (mutex de Windows).
- Snooze, selector de cámara, exportar y borrar historial.
- `scripts/fetch_model.py` con verificación SHA256.
- Suite de tests con `unittest` (sin dependencias externas).
- CI con GitHub Actions y workflow de release.

### Fixed

- Repetición de recordatorios de pausa en modo foco.
- Trend horario ahora se calcula sobre tiempo real y no sobre score acumulado.
- Detección de gaps temporales por suspensión (no suma tiempo falso).
- Pacing de FPS sobre el ciclo completo.
- Validación de visibilidad de cadera en la métrica de torso.

### Removed

- Código muerto (`scoring.py`, `Notifier`, `SessionHistory`, helpers sin uso).
