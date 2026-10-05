from __future__ import annotations

import threading
import tkinter as tk
from typing import Any

import cv2

from .guides import (
    LATERAL_LANDMARKS,
    display_point,
    landmark_color,
    laterality_label,
    planned_segments,
    visible_landmarks,
)
from .models import DetectionMetrics


class VisualOverlay:
    def __init__(self) -> None:
        self.root: tk.Tk | None = None
        self.label: tk.Label | None = None
        self.visible = False
        self.enabled = True
        self._mode = "banner"
        try:
            root = tk.Tk()
            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            frame = tk.Frame(root, bg="#202124", bd=3, relief="solid")
            frame.pack(fill="both", expand=True)
            label = tk.Label(
                frame,
                text="",
                font=("Segoe UI", 18, "bold"),
                fg="white",
                bg="#202124",
                padx=28,
                pady=18,
                justify="center",
            )
            label.pack(fill="both", expand=True)
            self.root = root
            self.label = label
        except Exception:
            self.enabled = False

    def _place_banner(self, width: int = 620, height: int = 110) -> None:
        if not self.root:
            return
        screen_w = self.root.winfo_screenwidth()
        x = max(0, (screen_w - width) // 2)
        y = 32
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _place_focus_tint(self) -> None:
        if not self.root:
            return
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        height = max(120, int(screen_h * 0.18))
        self.root.geometry(f"{screen_w}x{height}+0+0")

    def show(
        self,
        message: str,
        bg: str = "#b00020",
        fg: str = "white",
        *,
        mode: str = "banner",
        alpha: float = 0.92,
    ) -> None:
        if not self.enabled or not self.root or not self.label:
            return
        self._mode = mode
        parent = self.label.master

        if mode == "focus_tint":
            self.root.configure(bg=bg)
            if parent is not None:
                parent.config(bg=bg, bd=0, relief="flat")
            self.label.config(
                text=message,
                bg=bg,
                fg=fg,
                padx=20,
                pady=10,
                font=("Segoe UI", 15, "bold"),
                anchor="center",
            )
            self._place_focus_tint()
        else:
            if parent is not None:
                parent.config(bg=bg, bd=3, relief="solid")
            self.label.config(
                text=message,
                bg=bg,
                fg=fg,
                padx=28,
                pady=18,
                font=("Segoe UI", 18, "bold"),
                anchor="center",
            )
            self.root.configure(bg=bg)
            self._place_banner()

        try:
            self.root.attributes("-alpha", alpha)
        except Exception:
            pass
        self.root.deiconify()
        self.root.lift()
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        self.visible = True

    def hide(self) -> None:
        if not self.enabled or not self.root:
            return
        self.root.withdraw()
        self.visible = False

    def pump(self) -> None:
        if not self.enabled or not self.root:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.enabled = False

    def close(self) -> None:
        if not self.enabled or not self.root:
            return
        try:
            self.root.destroy()
        except Exception:
            pass
        self.enabled = False


class TrayIcon:
    """System tray icon using pystray. Runs in a background thread."""

    STATUS_COLORS = {
        "ok": (0, 180, 0),
        "warning": (220, 180, 0),
        "bad": (200, 0, 0),
        "break": (220, 130, 0),
        "off": (120, 120, 120),
    }

    def __init__(self) -> None:
        self.enabled = False
        self._icon: Any = None
        self._thread: threading.Thread | None = None
        self._status = "off"
        self._quit_requested = False
        self._recalibrate_good = False
        self._recalibrate_bad = False
        self._add_view_requested = False
        self._toggle_camera_requested = False
        self._toggle_focus_requested = False
        self._show_window_requested = False

        try:
            import pystray  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore

            self._pystray = pystray
            self._PIL_Image = Image
            self._PIL_ImageDraw = ImageDraw
            self.enabled = True
        except (ImportError, Exception):
            pass

    def _create_image(self, color: tuple[int, int, int]) -> Any:
        img = self._PIL_Image.new("RGB", (64, 64), color=(30, 30, 30))
        draw = self._PIL_ImageDraw.Draw(img)
        draw.ellipse((12, 12, 52, 52), fill=color)
        return img

    def _build_menu(self) -> Any:
        pystray = self._pystray
        return pystray.Menu(
            pystray.MenuItem("Mostrar ventana", self._on_show_window),
            pystray.MenuItem("Mostrar/Ocultar camara", self._toggle_camera),
            pystray.MenuItem("Alternar modo foco", self._toggle_focus),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Calibrar postura buena", self._on_recal_good),
            pystray.MenuItem("Calibrar postura mala", self._on_recal_bad),
            pystray.MenuItem("Agregar posicion", self._on_add_view),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Salir", self._on_quit),
        )

    def _on_show_window(self, icon: Any = None, item: Any = None) -> None:
        self._show_window_requested = True

    def _toggle_camera(self, icon: Any = None, item: Any = None) -> None:
        self._toggle_camera_requested = True

    def _toggle_focus(self, icon: Any = None, item: Any = None) -> None:
        self._toggle_focus_requested = True

    def _on_recal_good(self, icon: Any = None, item: Any = None) -> None:
        self._recalibrate_good = True

    def _on_recal_bad(self, icon: Any = None, item: Any = None) -> None:
        self._recalibrate_bad = True

    def _on_add_view(self, icon: Any = None, item: Any = None) -> None:
        self._add_view_requested = True

    def _on_quit(self, icon: Any = None, item: Any = None) -> None:
        self._quit_requested = True
        if self._icon:
            self._icon.stop()

    def start(self) -> None:
        if not self.enabled:
            return
        color = self.STATUS_COLORS["off"]
        self._icon = self._pystray.Icon(
            "posture_guard",
            self._create_image(color),
            "Posture Guard",
            menu=self._build_menu(),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def set_status(self, status: str) -> None:
        if not self.enabled or not self._icon or status == self._status:
            return
        self._status = status
        color = self.STATUS_COLORS.get(status, self.STATUS_COLORS["off"])
        self._icon.icon = self._create_image(color)

    def poll_actions(self) -> dict[str, bool]:
        actions = {
            "quit": self._quit_requested,
            "recalibrate_good": self._recalibrate_good,
            "recalibrate_bad": self._recalibrate_bad,
            "add_view": self._add_view_requested,
            "toggle_camera": self._toggle_camera_requested,
            "toggle_focus": self._toggle_focus_requested,
            "show_window": self._show_window_requested,
        }
        self._recalibrate_good = False
        self._recalibrate_bad = False
        self._add_view_requested = False
        self._toggle_camera_requested = False
        self._toggle_focus_requested = False
        self._show_window_requested = False
        return actions
    def stop(self) -> None:
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass


def draw_text_block(
    frame: Any,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
    line_height: int = 22,
) -> None:
    x, y = origin
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def _px(x: float, y: float, width: int, height: int, mirror: bool) -> tuple[int, int]:
    display_x, display_y = display_point(x, y, mirror)
    return (int(display_x * width), int(display_y * height))


def _draw_v2_guides(frame: Any, metrics: DetectionMetrics, min_visibility: float, mirror: bool) -> None:
    h, w = frame.shape[:2]
    visible = visible_landmarks(metrics.debug_landmarks, min_visibility)

    px_points: dict[str, tuple[int, int]] = {}
    for name, landmark in visible.items():
        point = _px(landmark.x, landmark.y, w, h, mirror)
        px_points[name] = point
        cv2.circle(frame, point, 4, landmark_color(name), -1)
        if name in LATERAL_LANDMARKS:
            cv2.putText(
                frame,
                laterality_label(name),
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                landmark_color(name),
                2,
                cv2.LINE_AA,
            )

    for a, b, color, thickness in planned_segments(set(px_points)):
        cv2.line(frame, px_points[a], px_points[b], color, thickness)


def _draw_legacy_guides(frame: Any, metrics: DetectionMetrics, mirror: bool) -> None:
    h, w = frame.shape[:2]
    colors = {
        "ear": (0, 255, 255),
        "shoulder": (0, 255, 0),
        "hip": (255, 200, 0),
        "nose": (255, 0, 255),
    }

    px_points: dict[str, tuple[int, int]] = {}
    for name, point in metrics.points.items():
        px_points[name] = _px(point[0], point[1], w, h, mirror)
        cv2.circle(frame, px_points[name], 5, colors.get(name, (255, 255, 255)), -1)

    if {"ear", "shoulder", "hip", "nose"}.issubset(px_points.keys()):
        cv2.line(frame, px_points["ear"], px_points["shoulder"], (0, 255, 255), 2)
        cv2.line(frame, px_points["shoulder"], px_points["hip"], (255, 200, 0), 2)
        cv2.line(frame, px_points["nose"], px_points["shoulder"], (255, 0, 255), 1)


def draw_guides(
    frame: Any,
    metrics: DetectionMetrics | None,
    min_visibility: float = 0.0,
    mirror: bool = False,
) -> None:
    """Draw the V2 landmark overlay, falling back to the legacy points.

    Landmarks/points are in raw model coordinates; pass ``mirror=True`` when the
    frame being drawn is the horizontally flipped display preview, so only the
    rendering flips x (physics/geometry stay in raw coordinates).
    """
    if not metrics:
        return
    if metrics.debug_landmarks:
        _draw_v2_guides(frame, metrics, min_visibility, mirror)
    else:
        _draw_legacy_guides(frame, metrics, mirror)


