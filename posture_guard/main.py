from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from . import paths
from .autostart import Autostart
from .calibration import Calibrator, load_calibration, save_calibration
from .camera import CameraError, list_cameras, open_camera
from .config import Config, load_config, save_config
from .detection import RollingMetrics, extract_metrics
from .diagnostics import build_report, format_report
from .engine import EngineResult, Event, PostureEngine
from .logging_setup import setup_logging
from .notifications import Notifications
from .single_instance import SingleInstance
from .state import SharedState
from .storage import AnalyticsStore, open_store
from .ui import TrayIcon, VisualOverlay, draw_guides, draw_text_block

logger = logging.getLogger(__name__)

SINGLE_INSTANCE_NAME = "PostureBreaker.single_instance"

BREAK_ROUTINES = [
    {
        "title": "Solta hombros",
        "steps": [
            "Parate y afloja hombros 20s.",
            "Hace dos circulos lentos hacia atras.",
            "Respira profundo y baja la tension.",
        ],
        "rule": "Regla 20-20-20: mira lejos 20s cada 20 min.",
    },
    {
        "title": "Ojos lejos",
        "steps": [
            "Mira un punto lejano y parpadea lento.",
            "Aleja la vista de la pantalla 20s.",
            "Relaja mandibula y frente.",
        ],
        "rule": "Si estas muy concentrado, usa la pausa para resetear la vista.",
    },
    {
        "title": "Cuello y espalda",
        "steps": [
            "Gira el cuello suave a cada lado.",
            "Lleva el pecho arriba y junta escapulas.",
            "Volve a sentarte recien cuando se suelte la nuca.",
        ],
        "rule": "Una pausa corta vale mas que seguir duro 40 minutos mas.",
    },
]


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _severity_prefix(severity: float) -> str:
    if severity >= 2.2:
        return "Fuerte"
    if severity >= 1.0:
        return "Marcada"
    return "Leve"


def _current_break_routine(completed_breaks: int) -> dict[str, object]:
    return BREAK_ROUTINES[completed_breaks % len(BREAK_ROUTINES)]


def _break_payload(break_due: bool, progress: float, countdown_seconds: float, completed_breaks: int) -> tuple[str, str, str]:
    routine = _current_break_routine(completed_breaks)
    steps = routine["steps"]
    if break_due:
        if progress < 0.34:
            step = steps[0]
        elif progress < 0.67:
            step = steps[1]
        else:
            step = steps[2]
        return str(routine["title"]), str(step), str(routine["rule"])

    minutes = max(0, int(countdown_seconds) // 60)
    seconds = max(0, int(countdown_seconds) % 60)
    title = f"Proxima pausa en {minutes}m {seconds:02d}s"
    body = f"Siguiente rutina: {routine['title']}. {steps[0]}"
    return title, body, str(routine["rule"])


def _tray_status_from_state(status: str) -> str:
    mapping = {
        "ok": "ok",
        "bad": "bad",
        "break_due": "break",
        "no_calibration": "warning",
        "calibrating": "warning",
        "camera_error": "warning",
        "camera_reconnecting": "warning",
        "missing_model": "warning",
        "error": "bad",
    }
    return mapping.get(status, "off")


def _status_from_result(result: EngineResult) -> str:
    if not result.profile_ready:
        return "no_calibration"
    if result.break_due:
        return "break_due"
    if result.posture_active and result.bad_streak_seconds >= result.posture_threshold_seconds:
        return "bad"
    return "ok"


def _handle_events(
    result: EngineResult,
    store: AnalyticsStore,
    notifications: Notifications,
    smoother: RollingMetrics,
    config: Config,
) -> None:
    for event in result.events:
        if event is Event.SESSION_GAP:
            smoother.clear()
            store.log_break_event(
                "session_gap",
                detail=f"Gap de {result.session_gap_seconds:.0f}s",
                focus_mode=result.focus_mode,
            )
        elif event is Event.POSTURE_ALERT:
            detail = result.issue_guidance or "Corrige cabeza, hombros y espalda."
            notifications.posture_alert(result.issue_label or "Enderezate", detail)
            store.log_posture_event(
                "posture_alert",
                error_type=result.issue_type,
                severity=result.issue_severity,
                suppressed=False,
                detail=detail,
            )
        elif event is Event.POSTURE_ALERT_SUPPRESSED:
            store.log_posture_event(
                "posture_alert_suppressed",
                error_type=result.issue_type,
                severity=result.issue_severity,
                suppressed=True,
                detail=result.issue_guidance or "Corrige cabeza, hombros y espalda.",
            )
        elif event in (Event.BREAK_ALERT, Event.BREAK_ALERT_REPEAT):
            routine_title, routine_step, _ = _break_payload(
                True, result.break_progress, 0.0, result.completed_breaks
            )
            notifications.break_alert(f"{routine_title}. {routine_step}")
            store.log_break_event(
                event.value,
                detail=f"{routine_title}: {routine_step}",
                focus_mode=result.focus_mode,
            )
        elif event is Event.BREAK_COMPLETED:
            notifications.break_completed()
            store.log_break_event(
                "break_completed",
                detail="Break completado",
                focus_mode=result.focus_mode,
            )
        elif event is Event.WORK_RESET_ABSENCE:
            store.log_break_event(
                "work_reset_absence",
                detail=f"Reset por ausencia > {int(config.away_reset_seconds)}s",
                focus_mode=result.focus_mode,
            )


def _load_profile(calibration_path: Path, shared: SharedState):
    try:
        return load_calibration(calibration_path)
    except Exception:
        logger.exception("calibration_load_failed")
        shared.update(
            calibrated=False,
            error_code="CALIBRATION_CORRUPTED",
            error_message="No pude leer la calibración; se ignora hasta recalibrar.",
        )
        return None


def _autostart_command(entry_path: Path) -> str:
    executable = Path(sys.executable)
    if getattr(sys, "frozen", False):
        return f'"{executable}"'
    return f'"{executable}" "{entry_path}"'


_METRIC_LABELS = {
    "ear_shoulder_dx": "Cabeza",
    "nose_shoulder_dx": "Cuello",
    "chin_drop": "Menton",
    "torso_lean_dx": "Torso",
}


def _calibration_summary(profile) -> str:
    enabled = [
        _METRIC_LABELS.get(name, name)
        for name, threshold in profile.thresholds.items()
        if threshold.mode != "disabled"
    ]
    disabled = [
        _METRIC_LABELS.get(name, name)
        for name, threshold in profile.thresholds.items()
        if threshold.mode == "disabled"
    ]
    parts = [f"Calidad: {profile.quality_score:.0f}%"]
    if enabled:
        parts.append("Detecta: " + ", ".join(enabled))
    if disabled:
        parts.append("Ignora: " + ", ".join(disabled))
    return " | ".join(parts)


def _switch_camera(cap: Any, config: Config, shared: SharedState, new_index: int) -> Any:
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    shared.update(status="camera_reconnecting")
    try:
        new_cap = open_camera(cv2.VideoCapture, new_index)
    except CameraError:
        logger.error("camera_switch_failed index=%s", new_index)
        return None
    logger.info("camera_switched index=%s", new_index)
    return new_cap


def _reconnect_camera(cap: Any, config: Config, shared: SharedState) -> Any:
    logger.warning("camera_read_failed; reconnecting")
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    shared.update(status="camera_reconnecting")
    try:
        new_cap = open_camera(cv2.VideoCapture, config.camera_index)
    except CameraError:
        logger.error("camera_reconnect_failed")
        return None
    logger.info("camera_reconnected")
    return new_cap


def _camera_worker(
    shared: SharedState,
    config: Config,
    data_dir: Path,
    resource_dir: Path,
    calibration_path: Path,
) -> None:
    store = None
    cap = None
    try:
        profile = _load_profile(calibration_path, shared)
        notifications = Notifications()
        engine = PostureEngine(config, profile)
        calibrator = Calibrator(config.calibration_frames)
        smoother = RollingMetrics(config.smoothing_window)

        db_path = _resolve_path(data_dir, config.database_path)
        legacy_history_dir = _resolve_path(data_dir, config.history_dir)
        try:
            store = open_store(db_path, legacy_history_dir=legacy_history_dir)
        except sqlite3.DatabaseError:
            logger.exception("database_open_failed")
            shared.update(
                status="error",
                error_code="DATABASE_ERROR",
                error_message="No pude abrir la base de datos de analytics.",
            )
            return
        store.start_session(config.camera_index)

        config_path = data_dir / "posture_break_guard.config.json"
        available_cameras = list_cameras(cv2.VideoCapture)
        logger.info("cameras_available=%s", available_cameras)
        autostart = Autostart("PostureBreaker", _autostart_command(data_dir / "posture_break_guard.py"))

        show_camera = False
        shared.update(
            calibrated=profile is not None,
            status="no_calibration" if not profile else "ok",
            streak_days=store.load_streak(),
            hourly_trend=store.hourly_trend(),
            weekly_trend=store.weekly_trend(),
            top_errors=store.top_errors(),
            focus_mode=False,
            camera_visible=show_camera,
            available_cameras=available_cameras,
            camera_index=config.camera_index,
            autostart_enabled=autostart.is_enabled(),
        )

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        model_path = _resolve_path(data_dir, config.model_path)
        if not model_path.exists():
            bundled_model = _resolve_path(resource_dir, config.model_path)
            if bundled_model.exists():
                model_path = bundled_model
        if not model_path.exists():
            logger.error("model_missing path=%s", model_path)
            shared.update(
                status="error",
                error_code="MODEL_MISSING",
                error_message=f"No encontre el modelo en {model_path}. Descargalo con scripts/fetch_model.py.",
            )
            return

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        shared.update(status="camera_reconnecting")
        cap = open_camera(
            cv2.VideoCapture,
            config.camera_index,
            on_retry=lambda attempt, delay: logger.warning(
                "camera_retry attempt=%s delay=%.1f", attempt, delay
            ),
        )
        logger.info("camera_opened index=%s", config.camera_index)

        last_frame_ts = time.monotonic()
        frame_interval = 1.0 / max(1.0, config.target_fps)
        last_sample_at = 0.0
        last_sample_signature: tuple[object, ...] | None = None
        last_refresh_at = 0.0

        with PoseLandmarker.create_from_options(options) as landmarker:
            while not shared.consume_command("cmd_quit"):
                frame_started_at = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    cap = _reconnect_camera(cap, config, shared)
                    if cap is None:
                        shared.update(
                            status="error",
                            error_code="CAMERA_UNAVAILABLE",
                            error_message="Se perdió la camara y no pude reconectar.",
                        )
                        return
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.monotonic() * 1000)
                detection = landmarker.detect_for_video(mp_image, timestamp_ms)

                preferred_side = profile.side if profile else None
                raw_metrics = extract_metrics(detection, config, preferred_side=preferred_side)
                if raw_metrics:
                    smoother.append(raw_metrics)
                else:
                    smoother.clear()

                metrics = smoother.mean()
                has_pose = metrics is not None

                now_ts = time.monotonic()
                dt_seconds = max(0.0, now_ts - last_frame_ts)
                last_frame_ts = now_ts

                if shared.consume_command("cmd_calibrate_good"):
                    calibrator.start("good")
                    notifications.generic()
                if shared.consume_command("cmd_calibrate_bad"):
                    if profile:
                        calibrator.start("bad", expected_side=profile.side)
                        notifications.generic()
                if shared.consume_command("cmd_clear_calibration"):
                    profile = None
                    engine.clear_profile()
                    calibrator.cancel()
                    if calibration_path.exists():
                        calibration_path.unlink()
                    notifications.generic()
                    shared.update(
                        calibrated=False,
                        status="no_calibration",
                        calibration_quality=0.0,
                        calibration_summary="",
                    )
                if shared.consume_command("cmd_toggle_camera"):
                    show_camera = not show_camera
                if shared.consume_command("cmd_toggle_focus"):
                    engine.toggle_focus()
                    store.set_focus_mode(engine.focus_mode)
                if shared.consume_command("cmd_snooze"):
                    engine.snooze(now_ts, config.snooze_minutes)
                    notifications.generic()
                if shared.consume_command("cmd_clear_history"):
                    store.clear_history()
                    notifications.generic()
                if shared.consume_command("cmd_export_history"):
                    try:
                        dest = store.export_history(data_dir / "posture_export.json")
                        shared.update(calibration_summary=f"Datos exportados a {dest.name}")
                    except Exception:
                        logger.exception("export_failed")
                    notifications.generic()
                if shared.consume_command("cmd_set_camera"):
                    new_index = int(shared.consume_value("pending_camera_index", -1))
                    if new_index >= 0 and new_index != config.camera_index:
                        new_cap = _switch_camera(cap, config, shared, new_index)
                        if new_cap is None:
                            shared.update(
                                status="error",
                                error_code="CAMERA_UNAVAILABLE",
                                error_message="No pude abrir la camara seleccionada.",
                            )
                            return
                        cap = new_cap
                        config.camera_index = new_index
                        try:
                            save_config(config_path, config)
                        except Exception:
                            logger.exception("config_save_failed")
                        shared.update(camera_index=new_index)

                if shared.consume_command("cmd_toggle_autostart"):
                    if autostart.supported():
                        enabled = autostart.toggle()
                        shared.update(
                            autostart_enabled=enabled,
                            calibration_summary=(
                                f"Inicio automatico {'activado' if enabled else 'desactivado'}"
                            ),
                        )
                    else:
                        shared.update(calibration_summary="Inicio automatico no soportado en este sistema.")
                    notifications.generic()
                if shared.consume_command("cmd_diagnostics"):
                    try:
                        report = build_report(
                            config,
                            data_dir,
                            cameras=available_cameras,
                            log_path=data_dir / "logs" / "posturebreaker.log",
                        )
                        dest = data_dir / "diagnostics.txt"
                        dest.write_text(format_report(report), encoding="utf-8")
                        shared.update(calibration_summary=f"Diagnostico escrito en {dest.name}")
                    except Exception:
                        logger.exception("diagnostics_failed")
                    notifications.generic()

                if calibrator.is_running():
                    shared.update(
                        calibrating_mode=calibrator.mode or "",
                        calibration_progress=len(calibrator.samples) / max(1, calibrator.target_frames),
                        status="calibrating",
                    )
                    if metrics:
                        done = calibrator.add(metrics)
                        if done:
                            try:
                                profile = calibrator.build_profile(profile, config)
                                engine.set_profile(profile)
                                save_calibration(calibration_path, profile)
                                logger.info("calibration_saved quality=%.1f", profile.quality_score)
                                notifications.calibration_ok()
                                shared.update(
                                    calibrated=True,
                                    calibrating_mode="",
                                    calibration_progress=0.0,
                                    calibration_quality=profile.quality_score,
                                    calibration_summary=_calibration_summary(profile),
                                )
                            except Exception as exc:
                                calibrator.cancel()
                                shared.update(calibrating_mode="", calibration_progress=0.0)
                                logger.warning("calibration_failed: %s", exc)
                else:
                    shared.update(calibrating_mode="", calibration_progress=0.0)

                result = engine.update(metrics, has_pose, dt_seconds, now_ts)

                if result.work_seconds_delta > 0.0:
                    store.add_work_time(result.work_seconds_delta, focus_mode=result.focus_mode)
                if result.count_posture_time:
                    store.add_posture_time(result.posture_seconds_delta, result.posture_bad)

                _handle_events(result, store, notifications, smoother, config)

                break_title, break_instruction, break_rule = _break_payload(
                    result.break_due,
                    result.break_progress,
                    result.break_countdown_seconds,
                    result.completed_breaks,
                )

                if not calibrator.is_running():
                    today = store.today_snapshot()
                    issue_title = result.issue_label
                    if result.posture_bad and result.issue_label:
                        issue_title = f"{_severity_prefix(result.issue_severity)}: {result.issue_label}"
                    shared.update(
                        status=_status_from_result(result),
                        posture_score=store.score,
                        is_bad_posture=result.posture_bad,
                        has_pose=result.has_pose,
                        work_seconds_today=today["total_work_seconds"],
                        bad_streak_seconds=result.bad_streak_seconds,
                        bad_posture_seconds_today=today["bad_posture_seconds"],
                        posture_alerts=today["posture_alerts"],
                        suppressed_posture_alerts=today["suppressed_posture_alerts"],
                        break_alerts=today["break_alerts"],
                        breaks_completed=today["breaks_completed"],
                        break_due=result.break_due,
                        summary_today=store.summary_today(),
                        current_issue=(
                            issue_title
                            if result.posture_bad
                            else ("Modo foco" if result.focus_mode else "")
                        ),
                        current_guidance=(
                            result.issue_guidance
                            if result.posture_bad
                            else (
                                "Alertas suaves activas. Se retrasa el aviso y aparece un tinte rojo suave si seguis encorvado."
                                if result.focus_mode
                                else ""
                            )
                        ),
                        focus_mode=result.focus_mode,
                        focus_hint=(
                            f"Modo foco activo. Pausa en {int(result.break_countdown_seconds // 60)}m. Alertas suaves hoy: {today['suppressed_posture_alerts']}"
                            if result.focus_mode
                            else f"Tiempo en foco hoy: {int(today['focus_seconds'] // 60)} min"
                        ),
                        focus_seconds_today=today["focus_seconds"],
                        camera_visible=show_camera,
                        break_progress=result.break_progress,
                        break_countdown_seconds=result.break_countdown_seconds,
                        break_title=break_title,
                        break_instruction=break_instruction,
                        break_rule_text=break_rule,
                    )

                if result.break_due:
                    shared.update(
                        overlay_message=f"PAUSA ACTIVA\n{break_instruction}",
                        overlay_bg="#d97706",
                        overlay_mode="banner",
                        overlay_alpha=0.92,
                    )
                elif result.posture_active and result.bad_streak_seconds >= result.posture_threshold_seconds:
                    title = (result.issue_label or "Enderezate").upper()
                    if result.focus_mode:
                        shared.update(
                            overlay_message=f"MODO FOCO  |  {title}",
                            overlay_bg="#b91c1c",
                            overlay_mode="focus_tint",
                            overlay_alpha=0.16,
                        )
                    else:
                        shared.update(
                            overlay_message=f"{title}\n{result.issue_guidance}",
                            overlay_bg="#b91c1c",
                            overlay_mode="banner",
                            overlay_alpha=0.92,
                        )
                else:
                    shared.update(overlay_message="")

                sample_signature = (
                    result.has_pose,
                    result.posture_bad,
                    result.issue_type,
                    result.break_due,
                    result.focus_mode,
                )
                if (
                    now_ts - last_sample_at >= max(5.0, config.analytics_sample_seconds)
                    or sample_signature != last_sample_signature
                ):
                    store.record_sample(
                        timestamp=dt.datetime.now(),
                        has_pose=result.has_pose,
                        is_bad=result.posture_bad,
                        break_due=result.break_due,
                        focus_mode=result.focus_mode,
                        error_type=result.issue_type,
                        error_severity=result.issue_severity,
                    )
                    last_sample_at = now_ts
                    last_sample_signature = sample_signature

                if now_ts - last_refresh_at >= config.trend_refresh_seconds:
                    store.flush()
                    shared.update(
                        streak_days=store.load_streak(),
                        hourly_trend=store.hourly_trend(),
                        weekly_trend=store.weekly_trend(),
                        top_errors=store.top_errors(),
                    )
                    last_refresh_at = now_ts

                if show_camera:
                    draw_guides(frame, metrics)
                    side_text = profile.side if profile else "sin calibrar"
                    info_lines = [
                        f"Lado: {metrics.side if metrics else '-'} | Cal: {side_text}",
                        f"Score: {store.score:.0f}%",
                        f"Issue: {result.issue_label or '-'}",
                        (
                            "Modo foco ON"
                            if result.focus_mode
                            else f"Pausa en {int(result.break_countdown_seconds // 60)}m"
                        ),
                    ]
                    draw_text_block(frame, info_lines, (18, 30), (255, 255, 255), scale=0.48)
                    shared.update(camera_frame=frame.copy())
                else:
                    shared.update(camera_frame=None)

                elapsed = time.monotonic() - frame_started_at
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except CameraError as exc:
        logger.error("camera_unavailable: %s", exc)
        shared.update(
            status="error",
            error_code="CAMERA_UNAVAILABLE",
            error_message=str(exc),
        )
    except Exception as exc:
        logger.exception("worker_crashed")
        shared.update(
            status="error",
            error_code="WORKER_CRASHED",
            error_message=str(exc),
        )
    finally:
        if store is not None:
            store.close()
        if cap is not None:
            cap.release()


def main() -> None:
    data_dir, resource_dir = paths.resolve_dirs()
    setup_logging(data_dir / "logs")
    logger.info("app_started data_dir=%s", data_dir)

    instance = SingleInstance(SINGLE_INSTANCE_NAME)
    if not instance.acquire():
        logger.warning("another_instance_running")
        return

    try:
        config_path = data_dir / "posture_break_guard.config.json"
        calibration_path = data_dir / "posture_calibration.json"

        config = load_config(config_path)
        shared = SharedState()

        tray = TrayIcon()
        tray.start()

        worker = threading.Thread(
            target=_camera_worker,
            args=(shared, config, data_dir, resource_dir, calibration_path),
            daemon=True,
        )
        worker.start()

        camera_window = "Posture Guard - Camara"
        overlay = VisualOverlay()

        def _show_camera_frame() -> None:
            with shared.lock:
                frame = shared.camera_frame
            if frame is not None:
                cv2.imshow(camera_window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    shared.update(cmd_toggle_camera=True)
            else:
                try:
                    cv2.destroyWindow(camera_window)
                    cv2.waitKey(1)
                except Exception:
                    pass

        def _pump_overlay() -> None:
            with shared.lock:
                msg = shared.overlay_message
                bg = shared.overlay_bg
                mode = shared.overlay_mode
                alpha = shared.overlay_alpha
            if msg:
                overlay.show(msg, bg=bg, mode=mode, alpha=alpha)
            else:
                overlay.hide()
            overlay.pump()

        def _tray_poll() -> None:
            if not tray.enabled:
                return
            actions = tray.poll_actions()
            if actions["quit"]:
                shared.update(cmd_quit=True)
            if actions["recalibrate_good"]:
                shared.update(cmd_calibrate_good=True)
            if actions["recalibrate_bad"]:
                shared.update(cmd_calibrate_bad=True)
            if actions["toggle_camera"]:
                shared.update(cmd_toggle_camera=True)
            if actions["toggle_focus"]:
                shared.update(cmd_toggle_focus=True)
            if actions["show_window"]:
                shared.update(cmd_show_window=True)

        try:
            if config.headless:
                raise ImportError

            from .dashboard import Dashboard

            app = Dashboard(shared)

            def _ui_loop() -> None:
                _tray_poll()
                if shared.consume_command("cmd_hide_window"):
                    if tray.enabled:
                        app.withdraw()
                    else:
                        shared.update(cmd_quit=True)
                        app.destroy()
                        return
                if shared.consume_command("cmd_show_window"):
                    app.deiconify()
                tray.set_status(_tray_status_from_state(shared.snapshot().get("status", "starting")))
                _show_camera_frame()
                _pump_overlay()
                app.after(66, _ui_loop)

            app.after(500, _ui_loop)
            app.mainloop()
        except ImportError:
            logger.info("headless_mode")
            print("customtkinter no instalado o modo headless activo.")
            try:
                while worker.is_alive():
                    _tray_poll()
                    tray.set_status(_tray_status_from_state(shared.snapshot().get("status", "starting")))
                    _show_camera_frame()
                    _pump_overlay()
                    time.sleep(0.066)
            except KeyboardInterrupt:
                pass

        shared.update(cmd_quit=True)
        worker.join(timeout=5)
        overlay.close()
        tray.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        logger.info("app_closed")
    finally:
        instance.release()
