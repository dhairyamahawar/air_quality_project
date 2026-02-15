"""
SQLite persistence layer for sensor readings.

Design goals:
- Zero setup: creates the DB and table automatically
- Simple, safe API: insert/fetch helpers
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

from simulator import SensorReading


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_unix INTEGER NOT NULL,
  ts_iso TEXT NOT NULL,
  temperature_c REAL NOT NULL,
  humidity_rh REAL NOT NULL,
  co2_ppm REAL NOT NULL,
  pm25_ug_m3 REAL NOT NULL,
  aqi INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_readings_ts_unix ON readings(ts_unix);
"""


class Database:
    def __init__(self, db_path: str = "air_quality.db") -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def insert_reading(self, reading: SensorReading, *, aqi: int) -> None:
        ts_unix = int(reading.ts_utc.timestamp())
        ts_iso = reading.ts_utc.isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO readings (
                  ts_unix, ts_iso, temperature_c, humidity_rh, co2_ppm, pm25_ug_m3, aqi
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_unix,
                    ts_iso,
                    float(reading.temperature_c),
                    float(reading.humidity_rh),
                    float(reading.co2_ppm),
                    float(reading.pm25_ug_m3),
                    int(aqi),
                ),
            )
            conn.commit()

    def fetch_latest(self, limit: int = 720) -> List[Dict[str, Any]]:
        """
        Fetch most recent rows (default ~1 hour at 5s interval).
        """
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM readings ORDER BY ts_unix DESC LIMIT ?",
                (int(limit),),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return list(reversed(rows))

    def fetch_since_unix(self, ts_unix: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM readings WHERE ts_unix >= ? ORDER BY ts_unix ASC",
                (int(ts_unix),),
            )
            return [dict(r) for r in cur.fetchall()]

    def fetch_range_unix(self, start_unix: int, end_unix: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT * FROM readings
                WHERE ts_unix BETWEEN ? AND ?
                ORDER BY ts_unix ASC
                """,
                (int(start_unix), int(end_unix)),
            )
            return [dict(r) for r in cur.fetchall()]
