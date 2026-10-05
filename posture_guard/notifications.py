from __future__ import annotations

import sys
import threading


class Notifications:
    """Native OS notifications with optional sound."""

    def __init__(self) -> None:
        self._is_windows = sys.platform.startswith("win")
        self._winsound = None
        self._plyer_notification = None

        if self._is_windows:
            try:
                import winsound  # type: ignore

                self._winsound = winsound
            except ImportError:
                pass

        try:
            from plyer import notification  # type: ignore

            self._plyer_notification = notification
        except ImportError:
            pass

    def _beep(self, freq: int, duration: int) -> None:
        if self._winsound:
            try:
                self._winsound.Beep(freq, duration)
            except Exception:
                pass
        else:
            print("\a", end="", flush=True)

    def _toast(self, title: str, message: str, timeout: int = 5) -> None:
        if not self._plyer_notification:
            return
        try:
            self._plyer_notification.notify(
                title=title,
                message=message,
                app_name="Posture Guard",
                timeout=timeout,
            )
        except Exception:
            pass

    def _send(self, title: str, message: str, freq: int, duration: int, timeout: int = 5) -> None:
        def _do() -> None:
            self._beep(freq, duration)
            self._toast(title, message, timeout)

        threading.Thread(target=_do, daemon=True).start()

    def posture_alert(self, title: str = "Enderezate", message: str = "Llevas un rato con mala postura.") -> None:
        self._send(title, message, 950, 220)

    def break_alert(self, message: str = "Llevas mucho tiempo trabajando. Levantate y estira.") -> None:
        self._send("Hora de pausa", message, 700, 500)

    def break_completed(self) -> None:
        self._send("Pausa completada", "Buen descanso. Segui asi.", 1100, 180)

    def calibration_ok(self) -> None:
        self._beep(1100, 180)

    def generic(self) -> None:
        self._beep(850, 180)
