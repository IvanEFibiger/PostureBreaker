from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class AnalyticsStore:
    def __init__(self, db_path: Path, legacy_history_dir: Path | None = None) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._session_id: int | None = None
        self._focus_period_id: int | None = None
        self._today = dt.date.today()

        self._migrate()
        if legacy_history_dir is not None:
            self._import_legacy_history(legacy_history_dir)

        self._daily = self._load_daily(self._today)
        self._ensure_daily_row(self._today)
        self._hourly = self._load_hourly(self._today)

    def _schema_version(self) -> int:
        row = self.conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def _migrate(self) -> None:
        try:
            version = self._schema_version()
            if version < 1:
                self._apply_base_schema()
                self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()
        except sqlite3.DatabaseError:
            self.conn.close()
            raise

    def _apply_base_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                camera_index INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                day TEXT PRIMARY KEY,
                total_work_seconds REAL NOT NULL DEFAULT 0,
                good_posture_seconds REAL NOT NULL DEFAULT 0,
                bad_posture_seconds REAL NOT NULL DEFAULT 0,
                posture_alerts INTEGER NOT NULL DEFAULT 0,
                suppressed_posture_alerts INTEGER NOT NULL DEFAULT 0,
                break_alerts INTEGER NOT NULL DEFAULT 0,
                breaks_completed INTEGER NOT NULL DEFAULT 0,
                focus_seconds REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS posture_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                ts TEXT NOT NULL,
                score REAL NOT NULL,
                is_bad INTEGER NOT NULL,
                has_pose INTEGER NOT NULL,
                break_due INTEGER NOT NULL,
                focus_mode INTEGER NOT NULL,
                error_type TEXT,
                error_severity REAL NOT NULL DEFAULT 0,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS posture_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                error_type TEXT,
                severity REAL NOT NULL DEFAULT 0,
                suppressed INTEGER NOT NULL DEFAULT 0,
                detail TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS break_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '',
                focus_mode INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS focus_periods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS hourly_stats (
                day TEXT NOT NULL,
                hour INTEGER NOT NULL,
                good_posture_seconds REAL NOT NULL DEFAULT 0,
                bad_posture_seconds REAL NOT NULL DEFAULT 0,
                work_seconds REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(day, hour)
            );

            CREATE INDEX IF NOT EXISTS idx_posture_samples_ts ON posture_samples(ts);
            CREATE INDEX IF NOT EXISTS idx_posture_events_ts ON posture_events(ts);
            CREATE INDEX IF NOT EXISTS idx_break_events_ts ON break_events(ts);
            CREATE INDEX IF NOT EXISTS idx_hourly_stats_day ON hourly_stats(day);
            """
        )
        self.conn.commit()

    def _empty_daily(self, day: dt.date) -> dict[str, Any]:
        return {
            "day": day.isoformat(),
            "total_work_seconds": 0.0,
            "good_posture_seconds": 0.0,
            "bad_posture_seconds": 0.0,
            "posture_alerts": 0,
            "suppressed_posture_alerts": 0,
            "break_alerts": 0,
            "breaks_completed": 0,
            "focus_seconds": 0.0,
        }

    def _ensure_daily_row(self, day: dt.date) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO daily_stats(day) VALUES (?)",
            (day.isoformat(),),
        )

    def _load_daily(self, day: dt.date) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM daily_stats WHERE day = ?",
            (day.isoformat(),),
        ).fetchone()
        if row is None:
            return self._empty_daily(day)
        return dict(row)

    def _empty_hourly(self) -> dict[int, dict[str, float]]:
        return {
            hour: {"good_posture_seconds": 0.0, "bad_posture_seconds": 0.0, "work_seconds": 0.0}
            for hour in range(24)
        }

    def _load_hourly(self, day: dt.date) -> dict[int, dict[str, float]]:
        hourly = self._empty_hourly()
        rows = self.conn.execute(
            """
            SELECT hour, good_posture_seconds, bad_posture_seconds, work_seconds
            FROM hourly_stats
            WHERE day = ?
            """,
            (day.isoformat(),),
        ).fetchall()
        for row in rows:
            hourly[int(row["hour"])] = {
                "good_posture_seconds": float(row["good_posture_seconds"]),
                "bad_posture_seconds": float(row["bad_posture_seconds"]),
                "work_seconds": float(row["work_seconds"]),
            }
        return hourly

    def _current_hour(self) -> int:
        return dt.datetime.now().hour

    def _score_from_payload(self, payload: dict[str, Any]) -> float:
        good = float(payload.get("good_posture_seconds", 0.0))
        bad = float(payload.get("bad_posture_seconds", 0.0))
        total = good + bad
        if total < 1.0:
            return 100.0
        return round(good / total * 100.0, 1)

    def _import_legacy_history(self, history_dir: Path) -> None:
        if not history_dir.exists():
            return
        existing = self.conn.execute("SELECT COUNT(*) AS total FROM daily_stats").fetchone()[0]
        if existing > 0:
            return

        for path in sorted(history_dir.glob("*.json")):
            if path.name == "streak.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            day = data.get("date") or path.stem
            self.conn.execute(
                """
                INSERT OR REPLACE INTO daily_stats(
                    day,
                    total_work_seconds,
                    good_posture_seconds,
                    bad_posture_seconds,
                    posture_alerts,
                    break_alerts,
                    breaks_completed,
                    focus_seconds,
                    suppressed_posture_alerts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    day,
                    float(data.get("total_work_seconds", 0.0)),
                    float(data.get("good_posture_seconds", 0.0)),
                    float(data.get("bad_posture_seconds", 0.0)),
                    int(data.get("posture_alerts", 0)),
                    int(data.get("break_alerts", 0)),
                    int(data.get("breaks_completed", 0)),
                    float(data.get("focus_seconds", 0.0)),
                    int(data.get("suppressed_posture_alerts", 0)),
                ),
            )
        self.conn.commit()

    def _roll_day(self) -> None:
        today = dt.date.today()
        if today == self._today:
            return
        self.flush()
        self._today = today
        self._daily = self._load_daily(today)
        self._ensure_daily_row(today)
        self._hourly = self._load_hourly(today)

    def start_session(self, camera_index: int) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO sessions(started_at, camera_index) VALUES (?, ?)",
            (now, camera_index),
        )
        self._session_id = int(cur.lastrowid)

    def end_session(self) -> None:
        if self._focus_period_id is not None:
            self.set_focus_mode(False)
        if self._session_id is None:
            return
        now = dt.datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (now, self._session_id),
        )
        self._session_id = None

    def set_focus_mode(self, enabled: bool) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        if enabled and self._focus_period_id is None:
            cur = self.conn.execute(
                "INSERT INTO focus_periods(session_id, started_at) VALUES (?, ?)",
                (self._session_id, now),
            )
            self._focus_period_id = int(cur.lastrowid)
        elif not enabled and self._focus_period_id is not None:
            self.conn.execute(
                "UPDATE focus_periods SET ended_at = ? WHERE id = ?",
                (now, self._focus_period_id),
            )
            self._focus_period_id = None

    def add_work_time(self, seconds: float, *, focus_mode: bool) -> None:
        self._roll_day()
        self._daily["total_work_seconds"] += seconds
        if focus_mode:
            self._daily["focus_seconds"] += seconds
        self._hourly[self._current_hour()]["work_seconds"] += seconds

    def add_posture_time(self, seconds: float, is_bad: bool) -> None:
        self._roll_day()
        key = "bad_posture_seconds" if is_bad else "good_posture_seconds"
        self._daily[key] += seconds
        self._hourly[self._current_hour()][key] += seconds

    def record_sample(
        self,
        *,
        timestamp: dt.datetime,
        has_pose: bool,
        is_bad: bool,
        break_due: bool,
        focus_mode: bool,
        error_type: str | None,
        error_severity: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO posture_samples(
                session_id,
                ts,
                score,
                is_bad,
                has_pose,
                break_due,
                focus_mode,
                error_type,
                error_severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._session_id,
                timestamp.isoformat(timespec="seconds"),
                self.score,
                int(is_bad),
                int(has_pose),
                int(break_due),
                int(focus_mode),
                error_type,
                error_severity,
            ),
        )

    def log_posture_event(
        self,
        event_type: str,
        *,
        error_type: str | None,
        severity: float,
        suppressed: bool,
        detail: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO posture_events(session_id, ts, event_type, error_type, severity, suppressed, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._session_id,
                dt.datetime.now().isoformat(timespec="seconds"),
                event_type,
                error_type,
                severity,
                int(suppressed),
                detail,
            ),
        )
        if suppressed:
            self._daily["suppressed_posture_alerts"] += 1
        else:
            self._daily["posture_alerts"] += 1

    def log_break_event(self, event_type: str, *, detail: str, focus_mode: bool) -> None:
        self.conn.execute(
            """
            INSERT INTO break_events(session_id, ts, event_type, detail, focus_mode)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self._session_id,
                dt.datetime.now().isoformat(timespec="seconds"),
                event_type,
                detail,
                int(focus_mode),
            ),
        )
        if event_type in {"break_alert", "break_alert_repeat"}:
            self._daily["break_alerts"] += 1
        elif event_type == "break_completed":
            self._daily["breaks_completed"] += 1

    @property
    def score(self) -> float:
        return self._score_from_payload(self._daily)

    def today_snapshot(self) -> dict[str, Any]:
        self._roll_day()
        payload = dict(self._daily)
        payload["posture_score"] = self.score
        return payload

    def summary_today(self) -> str:
        work_min = self._daily["total_work_seconds"] / 60.0
        return (
            f"Hoy: {work_min:.0f} min trabajo | "
            f"{self._daily['posture_alerts']} alertas postura | "
            f"{self._daily['break_alerts']} alertas pausa | "
            f"{self._daily['breaks_completed']} pausas hechas"
        )

    def _score_for_day(self, day: dt.date) -> float | None:
        row = self.conn.execute(
            "SELECT good_posture_seconds, bad_posture_seconds FROM daily_stats WHERE day = ?",
            (day.isoformat(),),
        ).fetchone()
        if row is None:
            return None
        return self._score_from_payload(dict(row))

    def load_streak(self) -> int:
        streak = 0
        day = dt.date.today() - dt.timedelta(days=1)
        while True:
            score = self._score_for_day(day)
            if score is None or score < 70.0:
                break
            streak += 1
            day -= dt.timedelta(days=1)
        total = self._daily["good_posture_seconds"] + self._daily["bad_posture_seconds"]
        if total > 60 and self.score >= 70.0:
            streak += 1
        return streak

    def hourly_trend(self, hours: int = 8) -> list[dict[str, Any]]:
        now = dt.datetime.now()
        by_hour: dict[int, float] = {}
        for hour, payload in self._hourly.items():
            good = payload["good_posture_seconds"]
            bad = payload["bad_posture_seconds"]
            total = good + bad
            if total >= 1.0:
                by_hour[hour] = round(good / total * 100.0, 1)
        start_hour = max(0, now.hour - hours + 1)
        return [
            {"label": f"{hour:02d}", "value": by_hour.get(hour, 0.0)}
            for hour in range(start_hour, now.hour + 1)
        ]

    def weekly_trend(self, days: int = 7) -> list[dict[str, Any]]:
        start_day = self._today - dt.timedelta(days=days - 1)
        rows = self.conn.execute(
            """
            SELECT day, good_posture_seconds, bad_posture_seconds
            FROM daily_stats
            WHERE day >= ?
            ORDER BY day ASC
            """,
            (start_day.isoformat(),),
        ).fetchall()
        by_day = {
            row["day"]: self._score_from_payload(dict(row))
            for row in rows
        }
        items: list[dict[str, Any]] = []
        for offset in range(days):
            day = start_day + dt.timedelta(days=offset)
            items.append({"label": day.strftime("%a")[:2], "value": by_day.get(day.isoformat(), 0.0)})
        return items

    def top_errors(self, days: int = 7, limit: int = 3) -> list[dict[str, Any]]:
        cutoff = dt.datetime.now() - dt.timedelta(days=days)
        rows = self.conn.execute(
            """
            SELECT error_type, COUNT(*) AS total
            FROM posture_events
            WHERE ts >= ? AND error_type IS NOT NULL AND error_type != ''
            GROUP BY error_type
            ORDER BY total DESC, error_type ASC
            LIMIT ?
            """,
            (cutoff.isoformat(timespec="seconds"), limit),
        ).fetchall()
        return [
            {"error_type": str(row["error_type"]), "count": int(row["total"])}
            for row in rows
        ]

    def flush(self) -> None:
        self._ensure_daily_row(self._today)
        self.conn.execute(
            """
            INSERT INTO daily_stats(
                day,
                total_work_seconds,
                good_posture_seconds,
                bad_posture_seconds,
                posture_alerts,
                suppressed_posture_alerts,
                break_alerts,
                breaks_completed,
                focus_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(day) DO UPDATE SET
                total_work_seconds = excluded.total_work_seconds,
                good_posture_seconds = excluded.good_posture_seconds,
                bad_posture_seconds = excluded.bad_posture_seconds,
                posture_alerts = excluded.posture_alerts,
                suppressed_posture_alerts = excluded.suppressed_posture_alerts,
                break_alerts = excluded.break_alerts,
                breaks_completed = excluded.breaks_completed,
                focus_seconds = excluded.focus_seconds
            """,
            (
                self._daily["day"],
                self._daily["total_work_seconds"],
                self._daily["good_posture_seconds"],
                self._daily["bad_posture_seconds"],
                self._daily["posture_alerts"],
                self._daily["suppressed_posture_alerts"],
                self._daily["break_alerts"],
                self._daily["breaks_completed"],
                self._daily["focus_seconds"],
            ),
        )
        for hour, payload in self._hourly.items():
            if not (payload["good_posture_seconds"] or payload["bad_posture_seconds"] or payload["work_seconds"]):
                continue
            self.conn.execute(
                """
                INSERT INTO hourly_stats(day, hour, good_posture_seconds, bad_posture_seconds, work_seconds)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(day, hour) DO UPDATE SET
                    good_posture_seconds = excluded.good_posture_seconds,
                    bad_posture_seconds = excluded.bad_posture_seconds,
                    work_seconds = excluded.work_seconds
                """,
                (
                    self._today.isoformat(),
                    hour,
                    payload["good_posture_seconds"],
                    payload["bad_posture_seconds"],
                    payload["work_seconds"],
                ),
            )
        self.conn.commit()

    def close(self) -> None:
        self.end_session()
        self.flush()
        self.conn.close()

    def clear_history(self) -> None:
        for table in (
            "posture_samples",
            "posture_events",
            "break_events",
            "focus_periods",
            "sessions",
            "daily_stats",
            "hourly_stats",
        ):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
        self._daily = self._empty_daily(self._today)
        self._ensure_daily_row(self._today)
        self._hourly = self._empty_hourly()
        self._focus_period_id = None
        self._session_id = None

    def export_history(self, dest: Path) -> Path:
        rows = self.conn.execute("SELECT * FROM daily_stats ORDER BY day").fetchall()
        payload = {
            "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
            "daily_stats": [dict(row) for row in rows],
        }
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest


def open_store(db_path: Path, legacy_history_dir: Path | None = None) -> AnalyticsStore:
    """Open the analytics database, quarantining and recreating a corrupt file."""
    try:
        return AnalyticsStore(db_path, legacy_history_dir=legacy_history_dir)
    except sqlite3.DatabaseError:
        corrupt_path = db_path.with_name(db_path.name + ".corrupt")
        if db_path.exists():
            db_path.replace(corrupt_path)
        return AnalyticsStore(db_path, legacy_history_dir=legacy_history_dir)
