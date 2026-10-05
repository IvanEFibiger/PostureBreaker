from __future__ import annotations

import datetime as dt
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp

from .alerts import BreakManager
from .calibration import Calibrator, load_calibration, save_calibration
from .config import Config, load_config
from .detection import RollingMetrics, classify_posture, dominant_issue, extract_metrics
from .models import AlertState
from .notifications import Notifications
from .state import SharedState
from .storage import AnalyticsStore
from .ui import TrayIcon, VisualOverlay, draw_guides, draw_text_block

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


def _runtime_dirs() -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        data_dir = Path(sys.executable).resolve().parent
        resource_dir = Path(getattr(sys, "_MEIPASS", data_dir)).resolve()
        return data_dir, resource_dir
    source_dir = Path(__file__).resolve().parent.parent
    return source_dir, source_dir


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
        "missing_model": "warning",
    }
    return mapping.get(status, "off")


def _camera_worker(
    shared: SharedState,
    config: Config,
    data_dir: Path,
    resource_dir: Path,
    calibration_path: Path,
) -> None:
    profile = load_calibration(calibration_path)
    notifications = Notifications()
    alert_state = AlertState()
    break_manager = BreakManager(config, alert_state)
    calibrator = Calibrator(config.calibration_frames)
    smoother = RollingMetrics(config.smoothing_window)

    db_path = _resolve_path(data_dir, config.database_path)
    legacy_history_dir = _resolve_path(data_dir, config.history_dir)
    store = AnalyticsStore(db_path, legacy_history_dir=legacy_history_dir)
    store.start_session(config.camera_index)

    show_camera = False
    focus_mode = False
    shared.update(
        calibrated=profile is not None,
        status="no_calibration" if not profile else "ok",
        streak_days=store.load_streak(),
        hourly_trend=store.hourly_trend(),
        weekly_trend=store.weekly_trend(),
        top_errors=store.top_errors(),
        focus_mode=focus_mode,
        camera_visible=show_camera,
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
        shared.update(status="missing_model")
        print(
            f"No encontre el modelo en: {model_path}\n"
            "Descarga pose_landmarker.task y dejalo junto a este script o el .exe."
        )
        store.close()
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

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        shared.update(status="camera_error")
        print(f"No pude abrir la camara indice {config.camera_index}.")
        store.close()
        return

    last_frame_ts = time.monotonic()
    frame_interval = 1.0 / max(1.0, config.target_fps)
    last_sample_at = 0.0
    last_sample_signature: tuple[object, ...] | None = None
    last_refresh_at = 0.0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while not shared.consume_command("cmd_quit"):
            ok, frame = cap.read()
            if not ok:
                shared.update(status="camera_error")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            preferred_side = profile.side if profile else None
            raw_metrics = extract_metrics(result, config, preferred_side=preferred_side)
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
                calibrator.cancel()
                if calibration_path.exists():
                    calibration_path.unlink()
                notifications.generic()
                shared.update(calibrated=False, status="no_calibration")
            if shared.consume_command("cmd_toggle_camera"):
                show_camera = not show_camera
            if shared.consume_command("cmd_toggle_focus"):
                focus_mode = not focus_mode
                store.set_focus_mode(focus_mode)

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
                            save_calibration(calibration_path, profile)
                            notifications.calibration_ok()
                            shared.update(
                                calibrated=True,
                                calibrating_mode="",
                                calibration_progress=0.0,
                            )
                        except Exception as exc:
                            calibrator.cancel()
                            shared.update(calibrating_mode="", calibration_progress=0.0)
                            print(f"Error de calibracion: {exc}")
            else:
                shared.update(calibrating_mode="", calibration_progress=0.0)

            if has_pose:
                store.add_work_time(dt_seconds, focus_mode=focus_mode)

            posture_bad, bad_by_metric, severity = classify_posture(
                profile, metrics, config.posture_min_bad_metrics
            )
            issue_type, issue_label, issue_guidance, issue_severity = dominant_issue(bad_by_metric, severity)

            if has_pose and profile:
                store.add_posture_time(dt_seconds, posture_bad)

            posture_threshold = config.sustained_bad_posture_seconds * (config.focus_posture_multiplier if focus_mode else 1.0)
            posture_cooldown = config.posture_alert_cooldown_seconds * (config.focus_cooldown_multiplier if focus_mode else 1.0)

            if posture_bad:
                alert_state.bad_posture_streak += dt_seconds
                alert_state.posture_active = True
                if (
                    alert_state.bad_posture_streak >= posture_threshold
                    and now_ts - alert_state.last_posture_alert_at >= posture_cooldown
                ):
                    alert_state.last_posture_alert_at = now_ts
                    alert_state.posture_alert_count += 1
                    detail = issue_guidance or "Corrige cabeza, hombros y espalda."
                    if focus_mode:
                        store.log_posture_event(
                            "posture_alert_suppressed",
                            error_type=issue_type,
                            severity=issue_severity,
                            suppressed=True,
                            detail=detail,
                        )
                    else:
                        notifications.posture_alert(issue_label or "Enderezate", detail)
                        store.log_posture_event(
                            "posture_alert",
                            error_type=issue_type,
                            severity=issue_severity,
                            suppressed=False,
                            detail=detail,
                        )
            else:
                alert_state.bad_posture_streak = 0.0
                alert_state.posture_active = False

            repeat_interval = (
                config.focus_break_repeat_seconds if focus_mode else config.break_repeat_alert_seconds
            )
            for event in break_manager.update(has_pose, dt_seconds, now_ts, repeat_interval):
                current_progress = alert_state.away_during_break / max(config.break_required_seconds, 1.0)
                routine_title, routine_step, routine_rule = _break_payload(
                    True,
                    current_progress,
                    0.0,
                    alert_state.completed_breaks,
                )
                if event in {"break_alert", "break_alert_repeat"}:
                    notifications.break_alert(f"{routine_title}. {routine_step}")
                    store.log_break_event(event, detail=f"{routine_title}: {routine_step}", focus_mode=focus_mode)
                elif event == "break_completed":
                    notifications.break_completed()
                    store.log_break_event("break_completed", detail="Break completado", focus_mode=focus_mode)
                elif event == "work_reset_absence":
                    store.log_break_event(
                        "work_reset_absence",
                        detail=f"Reset por ausencia > {int(config.away_reset_seconds)}s",
                        focus_mode=focus_mode,
                    )

            break_progress = 0.0
            countdown_seconds = 0.0
            if alert_state.break_due:
                break_progress = min(1.0, alert_state.away_during_break / max(config.break_required_seconds, 1.0))
            else:
                interval_seconds = config.break_interval_minutes * 60.0
                worked = min(interval_seconds, alert_state.work_since_break)
                break_progress = min(1.0, worked / max(interval_seconds, 1.0))
                countdown_seconds = max(0.0, interval_seconds - alert_state.work_since_break)

            break_title, break_instruction, break_rule = _break_payload(
                alert_state.break_due,
                break_progress,
                countdown_seconds,
                alert_state.completed_breaks,
            )

            if not calibrator.is_running():
                if not profile:
                    status = "no_calibration"
                elif alert_state.break_due:
                    status = "break_due"
                elif alert_state.posture_active and alert_state.bad_posture_streak >= posture_threshold:
                    status = "bad"
                else:
                    status = "ok"

                today = store.today_snapshot()
                issue_title = issue_label
                if posture_bad and issue_label:
                    issue_title = f"{_severity_prefix(issue_severity)}: {issue_label}"
                shared.update(
                    status=status,
                    posture_score=store.score,
                    is_bad_posture=posture_bad,
                    has_pose=has_pose,
                    work_seconds_today=today["total_work_seconds"],
                    bad_streak_seconds=alert_state.bad_posture_streak,
                    bad_posture_seconds_today=today["bad_posture_seconds"],
                    posture_alerts=today["posture_alerts"],
                    suppressed_posture_alerts=today["suppressed_posture_alerts"],
                    break_alerts=today["break_alerts"],
                    breaks_completed=today["breaks_completed"],
                    break_due=alert_state.break_due,
                    summary_today=store.summary_today(),
                    current_issue=issue_title if posture_bad else ("Modo foco" if focus_mode else ""),
                    current_guidance=(
                        issue_guidance if posture_bad else ("Alertas suaves activas. Se retrasa el aviso y aparece un tinte rojo suave si seguis encorvado." if focus_mode else "")
                    ),
                    focus_mode=focus_mode,
                    focus_hint=(
                        f"Modo foco activo. Pausa en {int(countdown_seconds // 60)}m. Alertas suaves hoy: {today['suppressed_posture_alerts']}"
                        if focus_mode
                        else f"Tiempo en foco hoy: {int(today['focus_seconds'] // 60)} min"
                    ),
                    focus_seconds_today=today["focus_seconds"],
                    camera_visible=show_camera,
                    break_progress=break_progress,
                    break_countdown_seconds=countdown_seconds,
                    break_title=break_title,
                    break_instruction=break_instruction,
                    break_rule_text=break_rule,
                )

            if alert_state.break_due:
                shared.update(
                    overlay_message=f"PAUSA ACTIVA\n{break_instruction}",
                    overlay_bg="#d97706",
                    overlay_mode="banner",
                    overlay_alpha=0.92,
                )
            elif alert_state.posture_active and alert_state.bad_posture_streak >= posture_threshold:
                title = (issue_label or "Enderezate").upper()
                if focus_mode:
                    shared.update(
                        overlay_message=f"MODO FOCO  |  {title}",
                        overlay_bg="#b91c1c",
                        overlay_mode="focus_tint",
                        overlay_alpha=0.16,
                    )
                else:
                    shared.update(
                        overlay_message=f"{title}\n{issue_guidance}",
                        overlay_bg="#b91c1c",
                        overlay_mode="banner",
                        overlay_alpha=0.92,
                    )
            else:
                shared.update(overlay_message="")

            sample_signature = (
                has_pose,
                posture_bad,
                issue_type,
                alert_state.break_due,
                focus_mode,
            )
            if (
                now_ts - last_sample_at >= max(5.0, config.analytics_sample_seconds)
                or sample_signature != last_sample_signature
            ):
                store.record_sample(
                    timestamp=dt.datetime.now(),
                    has_pose=has_pose,
                    is_bad=posture_bad,
                    break_due=alert_state.break_due,
                    focus_mode=focus_mode,
                    error_type=issue_type,
                    error_severity=issue_severity,
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
                    f"Issue: {issue_label or '-'}",
                    "Modo foco ON" if focus_mode else f"Pausa en {int(countdown_seconds // 60)}m",
                ]
                draw_text_block(frame, info_lines, (18, 30), (255, 255, 255), scale=0.48)
                shared.update(camera_frame=frame.copy())
            else:
                shared.update(camera_frame=None)

            elapsed = time.monotonic() - now_ts
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    store.close()
    cap.release()


def main() -> None:
    data_dir, resource_dir = _runtime_dirs()
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

    try:
        if config.headless:
            raise ImportError

        from .dashboard import Dashboard

        app = Dashboard(shared)

        def _ui_loop() -> None:
            _tray_poll()
            tray.set_status(_tray_status_from_state(shared.snapshot().get("status", "starting")))
            _show_camera_frame()
            _pump_overlay()
            app.after(66, _ui_loop)

        app.after(500, _ui_loop)
        app.mainloop()
    except ImportError:
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
