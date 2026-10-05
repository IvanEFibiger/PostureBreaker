from __future__ import annotations

from typing import Any

import customtkinter as ctk

from .detection import issue_details
from .issues import issue_label_for
from .state import SharedState

BG_COLOR = "#FFF7F1"
CARD_COLOR = "#FFFFFF"
TEXT_PRIMARY = "#1A1A2E"
TEXT_SECONDARY = "#6B7280"
GREEN = "#22C55E"
YELLOW = "#EAB308"
RED = "#EF4444"
ORANGE = "#F97316"
ACCENT = "#F97316"
BLUE = "#2563EB"
SOFT = "#F3F4F6"


def _score_color(score: float) -> str:
    if score >= 80:
        return GREEN
    if score >= 60:
        return YELLOW
    return RED


def _status_label(status: str) -> tuple[str, str]:
    mapping = {
        "starting": ("Iniciando...", TEXT_SECONDARY),
        "no_calibration": ("Sin calibrar", ORANGE),
        "ok": ("Buena postura", GREEN),
        "bad": ("Mala postura", RED),
        "break_due": ("Hora de pausa", ORANGE),
        "calibrating": ("Calibrando...", YELLOW),
        "missing_model": ("Falta modelo", RED),
        "camera_error": ("Error camara", RED),
        "camera_reconnecting": ("Reconectando...", ORANGE),
        "error": ("Error", RED),
    }
    return mapping.get(status, ("-", TEXT_SECONDARY))


def _format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _format_short(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


class ScoreRing(ctk.CTkCanvas):
    def __init__(self, master: Any, size: int = 160, **kwargs: Any) -> None:
        super().__init__(master, width=size, height=size, highlightthickness=0, bg=CARD_COLOR, **kwargs)
        self._size = size
        self._score = 100.0
        self._color = GREEN
        self._draw()

    def set_score(self, score: float) -> None:
        self._score = score
        self._color = _score_color(score)
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        s = self._size
        pad = 14
        width = 12
        self.create_arc(pad, pad, s - pad, s - pad, start=0, extent=360, outline="#E5E7EB", width=width, style="arc")
        extent = self._score / 100.0 * 360.0
        self.create_arc(pad, pad, s - pad, s - pad, start=90, extent=-extent, outline=self._color, width=width, style="arc")
        self.create_text(s // 2, s // 2 - 8, text=f"{self._score:.0f}%", font=("Segoe UI", 28, "bold"), fill=TEXT_PRIMARY)
        self.create_text(s // 2, s // 2 + 20, text="postura", font=("Segoe UI", 11), fill=TEXT_SECONDARY)


class StatCard(ctk.CTkFrame):
    def __init__(self, master: Any, label: str, value: str = "-", **kwargs: Any) -> None:
        super().__init__(master, fg_color=CARD_COLOR, corner_radius=12, **kwargs)
        self._label = ctk.CTkLabel(self, text=label, font=("Segoe UI", 11), text_color=TEXT_SECONDARY)
        self._label.pack(pady=(10, 0), padx=12)
        self._value = ctk.CTkLabel(self, text=value, font=("Segoe UI", 18, "bold"), text_color=TEXT_PRIMARY)
        self._value.pack(pady=(0, 10), padx=12)

    def set_value(self, value: str) -> None:
        self._value.configure(text=value)


class TrendChart(ctk.CTkCanvas):
    def __init__(self, master: Any, width: int = 360, height: int = 120, bar_color: str = ACCENT, **kwargs: Any) -> None:
        super().__init__(master, width=width, height=height, highlightthickness=0, bg=CARD_COLOR, **kwargs)
        self._width = width
        self._height = height
        self._bar_color = bar_color
        self._series: list[dict[str, Any]] = []
        self._title = ""

    def set_series(self, title: str, series: list[dict[str, Any]]) -> None:
        self._title = title
        self._series = series
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        self.create_text(12, 14, anchor="w", text=self._title, font=("Segoe UI", 12, "bold"), fill=TEXT_PRIMARY)
        if not self._series:
            self.create_text(self._width // 2, self._height // 2, text="Sin datos", fill=TEXT_SECONDARY, font=("Segoe UI", 11))
            return

        chart_top = 30
        chart_bottom = self._height - 24
        chart_height = chart_bottom - chart_top
        usable_width = self._width - 24
        count = max(1, len(self._series))
        step = usable_width / count
        bar_width = max(12, int(step * 0.52))

        for idx, item in enumerate(self._series):
            value = max(0.0, min(100.0, float(item.get("value", 0.0))))
            label = str(item.get("label", ""))
            x0 = 12 + idx * step + (step - bar_width) / 2
            x1 = x0 + bar_width
            y1 = chart_bottom
            y0 = y1 - (chart_height * value / 100.0)
            self.create_rectangle(x0, chart_top, x1, y1, fill=SOFT, outline="")
            self.create_rectangle(x0, y0, x1, y1, fill=self._bar_color, outline="")
            self.create_text((x0 + x1) / 2, y0 - 10, text=f"{value:.0f}", fill=TEXT_SECONDARY, font=("Segoe UI", 9))
            self.create_text((x0 + x1) / 2, self._height - 10, text=label, fill=TEXT_SECONDARY, font=("Segoe UI", 9))


class Dashboard(ctk.CTk):
    POLL_MS = 500

    def __init__(self, shared: SharedState) -> None:
        super().__init__()
        self._shared = shared

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("Posture Guard")
        self.geometry("500x900")
        self.minsize(460, 760)
        self.configure(fg_color=BG_COLOR)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._settings_loaded = False
        self._build_ui()
        self._poll()

    def _build_ui(self) -> None:
        self._scroll = ctk.CTkScrollableFrame(self, fg_color=BG_COLOR, corner_radius=0)
        self._scroll.pack(fill="both", expand=True)
        self._scroll.grid_columnconfigure(0, weight=1)
        host = self._scroll

        header = ctk.CTkFrame(host, fg_color=BG_COLOR)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 0))
        ctk.CTkLabel(header, text="Posture Guard", font=("Segoe UI", 24, "bold"), text_color=TEXT_PRIMARY).pack(side="left")
        self._status_badge = ctk.CTkLabel(
            header,
            text="  Iniciando...  ",
            font=("Segoe UI", 12, "bold"),
            text_color="white",
            fg_color=TEXT_SECONDARY,
            corner_radius=10,
        )
        self._status_badge.pack(side="right")

        ring_frame = ctk.CTkFrame(host, fg_color=BG_COLOR)
        ring_frame.grid(row=1, column=0, pady=(18, 8), sticky="n")
        self._score_ring = ScoreRing(ring_frame, size=170)
        self._score_ring.pack()

        self._streak_label = ctk.CTkLabel(host, text="", font=("Segoe UI", 13), text_color=TEXT_SECONDARY)
        self._streak_label.grid(row=2, column=0, pady=(0, 8))

        cards_frame = ctk.CTkFrame(host, fg_color=BG_COLOR)
        cards_frame.grid(row=3, column=0, sticky="ew", padx=24, pady=(4, 8))
        cards_frame.columnconfigure((0, 1), weight=1)
        self._card_work = StatCard(cards_frame, "Trabajo hoy")
        self._card_work.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self._card_breaks = StatCard(cards_frame, "Pausas")
        self._card_breaks.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self._card_alerts = StatCard(cards_frame, "Alertas postura")
        self._card_alerts.grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        self._card_bad = StatCard(cards_frame, "Mala postura hoy")
        self._card_bad.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        self._issue_frame = ctk.CTkFrame(host, fg_color=CARD_COLOR, corner_radius=16)
        self._issue_frame.grid(row=4, column=0, sticky="ew", padx=24, pady=(2, 8))
        ctk.CTkLabel(self._issue_frame, text="Correccion actual (V1)", font=("Segoe UI", 12, "bold"), text_color=TEXT_SECONDARY).pack(anchor="w", padx=16, pady=(14, 0))
        self._issue_title = ctk.CTkLabel(self._issue_frame, text="Esperando postura...", font=("Segoe UI", 20, "bold"), text_color=TEXT_PRIMARY)
        self._issue_title.pack(anchor="w", padx=16, pady=(2, 0))
        self._issue_body = ctk.CTkLabel(
            self._issue_frame,
            text="Aca vas a ver el tipo de error dominante y la correccion sugerida.",
            font=("Segoe UI", 12),
            text_color=TEXT_SECONDARY,
            justify="left",
            wraplength=420,
        )
        self._issue_body.pack(anchor="w", padx=16, pady=(4, 14))

        btn_frame = ctk.CTkFrame(host, fg_color=BG_COLOR)
        btn_frame.grid(row=5, column=0, sticky="ew", padx=24, pady=(8, 4))
        btn_frame.columnconfigure((0, 1), weight=1)
        self._btn_good = ctk.CTkButton(btn_frame, text="Calibrar buena", fg_color=GREEN, hover_color="#16A34A", font=("Segoe UI", 13, "bold"), command=self._on_cal_good)
        self._btn_good.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self._btn_bad = ctk.CTkButton(btn_frame, text="Calibrar mala", fg_color=ORANGE, hover_color="#EA580C", font=("Segoe UI", 13, "bold"), command=self._on_cal_bad)
        self._btn_bad.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self._cal_hint = ctk.CTkLabel(
            btn_frame,
            text="Camara de costado. Calibra 'buena' sentado normal y 'mala' con tu postura problema.",
            font=("Segoe UI", 10),
            text_color=TEXT_SECONDARY,
            wraplength=420,
            justify="left",
        )
        self._cal_hint.grid(row=1, column=0, columnspan=2, padx=6, pady=(0, 6), sticky="w")

        btn_frame2 = ctk.CTkFrame(host, fg_color=BG_COLOR)
        btn_frame2.grid(row=6, column=0, sticky="ew", padx=24, pady=(0, 4))
        btn_frame2.columnconfigure((0, 1), weight=1)
        self._btn_camera = ctk.CTkButton(btn_frame2, text="Mostrar camara", fg_color=BLUE, hover_color="#1D4ED8", font=("Segoe UI", 13), command=self._on_toggle_camera)
        self._btn_camera.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self._btn_clear = ctk.CTkButton(btn_frame2, text="Borrar calibracion", fg_color="#9CA3AF", hover_color="#6B7280", font=("Segoe UI", 13), command=self._on_clear_cal)
        self._btn_clear.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        btn_frame3 = ctk.CTkFrame(host, fg_color=BG_COLOR)
        btn_frame3.grid(row=7, column=0, sticky="ew", padx=24, pady=(0, 4))
        btn_frame3.columnconfigure(0, weight=1)
        self._btn_focus = ctk.CTkButton(btn_frame3, text="Activar modo foco", fg_color="#111827", hover_color="#374151", font=("Segoe UI", 13, "bold"), command=self._on_toggle_focus)
        self._btn_focus.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        self._focus_label = ctk.CTkLabel(host, text="", font=("Segoe UI", 12), text_color=TEXT_SECONDARY, wraplength=420, justify="left")
        self._focus_label.grid(row=8, column=0, padx=24, pady=(0, 6), sticky="w")

        self._cal_frame = ctk.CTkFrame(host, fg_color=BG_COLOR)
        self._cal_frame.grid(row=9, column=0, sticky="ew", padx=24, pady=(0, 6))
        self._cal_label = ctk.CTkLabel(self._cal_frame, text="", font=("Segoe UI", 12), text_color=TEXT_SECONDARY)
        self._cal_label.pack(anchor="w")
        self._cal_bar = ctk.CTkProgressBar(self._cal_frame, fg_color="#E5E7EB", progress_color=ACCENT)
        self._cal_bar.pack(fill="x", pady=(4, 0))
        self._cal_bar.set(0)
        self._cal_frame.grid_remove()

        self._break_frame = ctk.CTkFrame(host, fg_color="#FFF1E6", corner_radius=16)
        self._break_frame.grid(row=10, column=0, sticky="ew", padx=24, pady=(2, 10))
        ctk.CTkLabel(self._break_frame, text="Break coach", font=("Segoe UI", 12, "bold"), text_color=ORANGE).pack(anchor="w", padx=16, pady=(14, 0))
        self._break_title = ctk.CTkLabel(self._break_frame, text="Proxima pausa", font=("Segoe UI", 18, "bold"), text_color=TEXT_PRIMARY)
        self._break_title.pack(anchor="w", padx=16, pady=(2, 0))
        self._break_body = ctk.CTkLabel(self._break_frame, text="Cuando toque una pausa, aca vas a ver una consigna corta y progreso.", font=("Segoe UI", 12), text_color=TEXT_SECONDARY, wraplength=420, justify="left")
        self._break_body.pack(anchor="w", padx=16, pady=(4, 8))
        self._break_bar = ctk.CTkProgressBar(self._break_frame, fg_color="#F4C7A1", progress_color=ORANGE)
        self._break_bar.pack(fill="x", padx=16, pady=(0, 8))
        self._break_bar.set(0)
        self._break_meta = ctk.CTkLabel(self._break_frame, text="", font=("Segoe UI", 11, "bold"), text_color=TEXT_PRIMARY)
        self._break_meta.pack(anchor="w", padx=16, pady=(0, 4))
        self._break_rule = ctk.CTkLabel(self._break_frame, text="", font=("Segoe UI", 11), text_color=TEXT_SECONDARY, wraplength=420, justify="left")
        self._break_rule.pack(anchor="w", padx=16, pady=(0, 14))

        trends = ctk.CTkFrame(host, fg_color=BG_COLOR)
        trends.grid(row=11, column=0, sticky="ew", padx=24, pady=(0, 12))
        self._hour_chart = TrendChart(trends, width=420, height=130, bar_color=ACCENT)
        self._hour_chart.pack(fill="x", pady=(0, 10))
        self._week_chart = TrendChart(trends, width=420, height=130, bar_color=BLUE)
        self._week_chart.pack(fill="x")

        self._errors_frame = ctk.CTkFrame(host, fg_color=CARD_COLOR, corner_radius=16)
        self._errors_frame.grid(row=12, column=0, sticky="ew", padx=24, pady=(0, 24))
        ctk.CTkLabel(self._errors_frame, text="Errores mas repetidos", font=("Segoe UI", 12, "bold"), text_color=TEXT_SECONDARY).pack(anchor="w", padx=16, pady=(14, 0))
        self._errors_body = ctk.CTkLabel(self._errors_frame, text="Aun no hay datos suficientes.", font=("Segoe UI", 12), text_color=TEXT_PRIMARY, justify="left", wraplength=420)
        self._errors_body.pack(anchor="w", padx=16, pady=(6, 14))

        tools = ctk.CTkFrame(host, fg_color=CARD_COLOR, corner_radius=16)
        tools.grid(row=13, column=0, sticky="ew", padx=24, pady=(0, 24))
        ctk.CTkLabel(tools, text="Herramientas", font=("Segoe UI", 12, "bold"), text_color=TEXT_SECONDARY).pack(anchor="w", padx=16, pady=(14, 0))
        self._camera_menu = ctk.CTkOptionMenu(tools, values=["0"], font=("Segoe UI", 12))
        self._camera_menu.pack(fill="x", padx=16, pady=(6, 6))
        ctk.CTkButton(tools, text="Aplicar camara", command=self._on_apply_camera, fg_color=BLUE, hover_color="#1D4ED8", font=("Segoe UI", 12)).pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkButton(tools, text="Silenciar alertas (snooze)", command=self._on_snooze, fg_color="#6B7280", hover_color="#4B5563", font=("Segoe UI", 12)).pack(fill="x", padx=16, pady=(0, 6))
        data_row = ctk.CTkFrame(tools, fg_color="transparent")
        data_row.pack(fill="x", padx=16, pady=(0, 6))
        data_row.columnconfigure((0, 1), weight=1)
        ctk.CTkButton(data_row, text="Exportar datos", command=self._on_export, fg_color=BLUE, hover_color="#1D4ED8", font=("Segoe UI", 12)).grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(data_row, text="Borrar historial", command=self._on_clear_history, fg_color="#9CA3AF", hover_color="#6B7280", font=("Segoe UI", 12)).grid(row=0, column=1, padx=(6, 0), sticky="ew")
        self._autostart_btn = ctk.CTkButton(tools, text="Inicio automatico: off", command=self._on_toggle_autostart, fg_color="#6B7280", hover_color="#4B5563", font=("Segoe UI", 12))
        self._autostart_btn.pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkButton(tools, text="Generar diagnostico", command=self._on_diagnostics, fg_color="#6B7280", hover_color="#4B5563", font=("Segoe UI", 12)).pack(fill="x", padx=16, pady=(0, 6))
        self._tools_hint = ctk.CTkLabel(tools, text="", font=("Segoe UI", 11), text_color=TEXT_SECONDARY, wraplength=420, justify="left")
        self._tools_hint.pack(anchor="w", padx=16, pady=(0, 14))

        settings = ctk.CTkFrame(host, fg_color=CARD_COLOR, corner_radius=16)
        settings.grid(row=14, column=0, sticky="ew", padx=24, pady=(0, 24))
        ctk.CTkLabel(settings, text="Ajustes", font=("Segoe UI", 12, "bold"), text_color=TEXT_SECONDARY).pack(anchor="w", padx=16, pady=(14, 0))
        self._slider_sens = self._add_slider(settings, "Sensibilidad (visibilidad minima)", 0.30, 0.95, 13)
        self._slider_break = self._add_slider(settings, "Intervalo de pausas (min)", 10, 90, 80)
        self._slider_break_len = self._add_slider(settings, "Duracion de pausa (s)", 30, 180, 150)
        self._slider_sustain = self._add_slider(settings, "Segundos de mala postura para alertar", 5, 60, 55)
        self._settings_hint = ctk.CTkLabel(settings, text="", font=("Segoe UI", 11), text_color=TEXT_SECONDARY, wraplength=420, justify="left")
        self._settings_hint.pack(anchor="w", padx=16, pady=(8, 4))
        ctk.CTkButton(settings, text="Aplicar ajustes", command=self._on_apply_settings, fg_color=ACCENT, hover_color="#EA580C", font=("Segoe UI", 12, "bold")).pack(fill="x", padx=16, pady=(0, 14))

        validation = ctk.CTkFrame(host, fg_color=CARD_COLOR, corner_radius=16)
        validation.grid(row=15, column=0, sticky="ew", padx=24, pady=(0, 24))
        ctk.CTkLabel(validation, text="Validacion detector V2", font=("Segoe UI", 12, "bold"), text_color=TEXT_SECONDARY).pack(anchor="w", padx=16, pady=(14, 0))
        self._validation_title = ctk.CTkLabel(validation, text="Validacion guiada (observe-only)", font=("Segoe UI", 18, "bold"), text_color=TEXT_PRIMARY)
        self._validation_title.pack(anchor="w", padx=16, pady=(2, 0))
        self._validation_body = ctk.CTkLabel(
            validation,
            text="Ejecuta una serie guiada de posturas y genera un dataset para evaluar las metricas.",
            font=("Segoe UI", 12),
            text_color=TEXT_SECONDARY,
            wraplength=420,
            justify="left",
        )
        self._validation_body.pack(anchor="w", padx=16, pady=(4, 8))
        self._validation_bar = ctk.CTkProgressBar(validation, fg_color="#E5E7EB", progress_color=ACCENT)
        self._validation_bar.pack(fill="x", padx=16, pady=(0, 8))
        self._validation_bar.set(0)
        self._validation_meta = ctk.CTkLabel(validation, text="", font=("Segoe UI", 11, "bold"), text_color=TEXT_PRIMARY, wraplength=420, justify="left")
        self._validation_meta.pack(anchor="w", padx=16, pady=(0, 6))
        self._btn_validation = ctk.CTkButton(validation, text="Ejecutar test V2", command=self._on_start_validation, fg_color=ACCENT, hover_color="#EA580C", font=("Segoe UI", 13, "bold"))
        self._btn_validation.pack(fill="x", padx=16, pady=(0, 6))
        self._btn_cancel_validation = ctk.CTkButton(validation, text="Cancelar validacion", command=self._on_cancel_validation, fg_color="#9CA3AF", hover_color="#6B7280", font=("Segoe UI", 12))

    def _update_validation(self, snap: dict[str, Any]) -> None:
        active = bool(snap.get("validation_active", False))
        phase = str(snap.get("validation_phase", ""))
        message = str(snap.get("validation_message", ""))
        if active or phase in ("prepare", "capture"):
            title = str(snap.get("validation_title", ""))
            instruction = str(snap.get("validation_instruction", ""))
            index = int(snap.get("validation_scenario_index", 0))
            count = int(snap.get("validation_scenario_count", 0))
            remaining = float(snap.get("validation_remaining_seconds", 0.0))
            samples = int(snap.get("validation_sample_count", 0))
            self._validation_title.configure(text=f"VALIDACION V2 - {index} / {count}")
            if phase == "prepare":
                body = f"{title}\n\n{instruction}\n\nPreparacion: comenzamos en {int(round(remaining))} s"
            else:
                body = f"{title}\n\nCAPTURANDO: {int(round(remaining))} s\n{samples} muestras"
            self._validation_body.configure(text=body)
            self._validation_bar.set(float(snap.get("validation_progress", 0.0)))
            view_name = str(snap.get("validation_view_name", ""))
            run_id = str(snap.get("validation_run_id", ""))
            self._validation_meta.configure(text=f"Vista: {view_name} | run: {run_id} | {samples} muestras")
            self._btn_validation.pack_forget()
            self._btn_cancel_validation.pack(fill="x", padx=16, pady=(0, 14))
            return

        self._validation_title.configure(text="Validacion guiada (observe-only)")
        self._validation_body.configure(
            text=message or "Ejecuta una serie guiada de posturas y genera un dataset para evaluar las metricas."
        )
        self._validation_bar.set(0.0)
        self._btn_cancel_validation.pack_forget()

        calibrated = bool(snap.get("calibrated", False))
        calibrating = bool(snap.get("calibrating_mode", ""))
        status = str(snap.get("status", ""))
        cameras = list(snap.get("available_cameras", []))
        can_run = calibrated and not calibrating and bool(cameras) and status not in ("error", "camera_error")
        if can_run:
            self._validation_meta.configure(text="")
        else:
            if calibrating:
                reason = "Calibracion en curso."
            elif not calibrated:
                reason = "Necesitas calibracion y una vista activa."
            else:
                reason = "Camara no disponible."
            self._validation_meta.configure(text=reason)
        self._btn_validation.configure(state="normal" if can_run else "disabled", text="Ejecutar test V2")
        self._btn_validation.pack(fill="x", padx=16, pady=(0, 6))

    def _format_analytics(self, snap: dict[str, Any]) -> str:
        lines: list[str] = []

        top_errors = list(snap.get("top_errors", []))
        if top_errors:
            lines.append("Mas repetidos (7 dias):")
            for item in top_errors:
                label, _ = issue_details(str(item.get("error_type") or ""))
                label = label or str(item.get("error_type") or "Postura inestable")
                lines.append(f"  {label}: {int(item.get('count', 0))}")

        issue_times = list(snap.get("issue_times_today", []))
        if issue_times:
            lines.append("Tiempo por problema (hoy):")
            for item in issue_times[:5]:
                label = issue_label_for(str(item.get("issue") or "")) or str(item.get("issue") or "?")
                lines.append(f"  {label}: {_format_short(float(item.get('seconds', 0.0)))}")

        view_times = list(snap.get("view_times_today", []))
        if view_times:
            lines.append("Tiempo por vista (hoy):")
            for item in view_times[:5]:
                name = str(item.get("view_name") or item.get("view_id") or "?")
                lines.append(f"  {name}: {_format_short(float(item.get('seconds', 0.0)))}")

        return "\n".join(lines) if lines else "Aun no hay datos suficientes."

    def _add_slider(self, parent: Any, label: str, minimum: float, maximum: float, steps: int) -> Any:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(6, 0))
        ctk.CTkLabel(frame, text=label, font=("Segoe UI", 11), text_color=TEXT_SECONDARY).pack(anchor="w")
        slider = ctk.CTkSlider(frame, from_=minimum, to=maximum, number_of_steps=steps)
        slider.pack(fill="x")
        return slider

    def _poll(self) -> None:
        snap = self._shared.snapshot()
        self._score_ring.set_score(float(snap["posture_score"]))

        status_text, status_color = _status_label(str(snap["status"]))
        self._status_badge.configure(text=f"  {status_text}  ", fg_color=status_color)

        streak = int(snap.get("streak_days", 0))
        self._streak_label.configure(text=f"Racha: {streak} dia{'s' if streak != 1 else ''}" if streak > 0 else "")

        self._card_work.set_value(_format_time(float(snap["work_seconds_today"])))
        self._card_breaks.set_value(str(snap["breaks_completed"]))
        self._card_alerts.set_value(str(snap["posture_alerts"]))
        self._card_bad.set_value(_format_time(float(snap.get("bad_posture_seconds_today", 0.0))))

        current_issue = str(snap.get("current_issue", ""))
        current_guidance = str(snap.get("current_guidance", ""))
        bad_streak = float(snap.get("bad_streak_seconds", 0.0))
        if current_issue:
            self._issue_title.configure(text=current_issue)
            body = current_guidance or "Sin sugerencia puntual."
            if bad_streak > 0:
                body = f"{body}\nRacha actual: {_format_short(bad_streak)}"
            self._issue_body.configure(text=body)
        else:
            self._issue_title.configure(text="Postura estable")
            self._issue_body.configure(text="Sin correccion puntual ahora mismo.")

        cal_mode = str(snap.get("calibrating_mode", ""))
        if cal_mode:
            self._cal_frame.grid()
            label = "buena" if cal_mode == "good" else "mala"
            progress = float(snap.get("calibration_progress", 0.0))
            self._cal_label.configure(text=f"Calibrando postura {label}... quedate quieto")
            self._cal_bar.set(progress)
        else:
            self._cal_frame.grid_remove()

        camera_visible = bool(snap.get("camera_visible", False))
        self._btn_camera.configure(text="Ocultar camara" if camera_visible else "Mostrar camara")

        focus_mode = bool(snap.get("focus_mode", False))
        self._btn_focus.configure(
            text="Desactivar modo foco" if focus_mode else "Activar modo foco",
            fg_color="#111827" if not focus_mode else "#0F766E",
            hover_color="#374151" if not focus_mode else "#115E59",
        )
        self._focus_label.configure(text=str(snap.get("focus_hint", "")))

        self._break_title.configure(text=str(snap.get("break_title", "Proxima pausa")))
        self._break_body.configure(text=str(snap.get("break_instruction", "")))
        self._break_rule.configure(text=str(snap.get("break_rule_text", "")))
        self._break_bar.set(float(snap.get("break_progress", 0.0)))
        break_due = bool(snap.get("break_due", False))
        if break_due:
            self._break_meta.configure(text="Pausa activa: completa la barra fuera de camara")
        else:
            self._break_meta.configure(
                text=(
                    f"Tiempo restante: {_format_short(float(snap.get('break_countdown_seconds', 0.0)))}"
                    f" | Foco hoy: {_format_short(float(snap.get('focus_seconds_today', 0.0)))}"
                )
            )

        self._hour_chart.set_series("Hoy por hora", list(snap.get("hourly_trend", [])))
        self._week_chart.set_series("Ultimos 7 dias", list(snap.get("weekly_trend", [])))
        self._errors_body.configure(text=self._format_analytics(snap))

        cameras = [str(index) for index in snap.get("available_cameras", [])]
        if cameras and list(self._camera_menu.cget("values")) != cameras:
            self._camera_menu.configure(values=cameras)
        autostart_on = bool(snap.get("autostart_enabled", False))
        self._autostart_btn.configure(
            text=f"Inicio automatico: {'on' if autostart_on else 'off'}",
            fg_color="#0F766E" if autostart_on else "#6B7280",
            hover_color="#115E59" if autostart_on else "#4B5563",
        )
        self._tools_hint.configure(text=str(snap.get("calibration_summary", "")))
        self._update_validation(snap)

        settings = snap.get("settings") or {}
        if not self._settings_loaded and settings:
            self._slider_sens.set(float(settings.get("min_visibility", 0.55)))
            self._slider_break.set(float(settings.get("break_interval_minutes", 45)))
            self._slider_break_len.set(float(settings.get("break_required_seconds", 90)))
            self._slider_sustain.set(float(settings.get("sustained_bad_posture_seconds", 20)))
            self._settings_loaded = True
        if settings:
            self._settings_hint.configure(
                text=(
                    f"Actual: sensibilidad {float(settings.get('min_visibility', 0)):.2f} | "
                    f"pausas cada {float(settings.get('break_interval_minutes', 0)):.0f} min | "
                    f"pausa {float(settings.get('break_required_seconds', 0)):.0f}s | "
                    f"alerta {float(settings.get('sustained_bad_posture_seconds', 0)):.0f}s"
                )
            )

        self.after(self.POLL_MS, self._poll)

    def _on_cal_good(self) -> None:
        self._shared.update(cmd_calibrate_good=True)

    def _on_cal_bad(self) -> None:
        self._shared.update(cmd_calibrate_bad=True)

    def _on_clear_cal(self) -> None:
        self._shared.update(cmd_clear_calibration=True)

    def _on_toggle_camera(self) -> None:
        self._shared.update(cmd_toggle_camera=True)

    def _on_toggle_focus(self) -> None:
        self._shared.update(cmd_toggle_focus=True)

    def _on_apply_camera(self) -> None:
        try:
            index = int(self._camera_menu.get())
        except (TypeError, ValueError):
            return
        self._shared.update(pending_camera_index=index, cmd_set_camera=True)

    def _on_snooze(self) -> None:
        self._shared.update(cmd_snooze=True)

    def _on_export(self) -> None:
        self._shared.update(cmd_export_history=True)

    def _on_clear_history(self) -> None:
        self._shared.update(cmd_clear_history=True)

    def _on_toggle_autostart(self) -> None:
        self._shared.update(cmd_toggle_autostart=True)

    def _on_diagnostics(self) -> None:
        self._shared.update(cmd_diagnostics=True)

    def _on_start_validation(self) -> None:
        self._shared.update(cmd_start_validation=True, validation_message="")

    def _on_cancel_validation(self) -> None:
        self._shared.update(cmd_cancel_validation=True)

    def _on_apply_settings(self) -> None:
        self._shared.update(
            pending_settings={
                "min_visibility": round(float(self._slider_sens.get()), 2),
                "break_interval_minutes": round(float(self._slider_break.get()), 1),
                "break_required_seconds": round(float(self._slider_break_len.get()), 1),
                "sustained_bad_posture_seconds": round(float(self._slider_sustain.get()), 1),
            },
            cmd_apply_settings=True,
        )

    def _on_close(self) -> None:
        self._shared.update(cmd_hide_window=True)
