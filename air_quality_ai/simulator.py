"""
Simulated indoor air-quality sensors.

Generates realistic-ish values with:
- Slow trends (sinusoids / random walk)
- Gaussian noise
- Occasional spikes (events like cooking, ventilation changes, occupancy)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import random
from typing import Optional


@dataclass(frozen=True)
class SensorReading:
    ts_utc: datetime
    temperature_c: float
    humidity_rh: float
    co2_ppm: float
    pm25_ug_m3: float


class SensorSimulator:
    """
    Stateful simulator that produces plausible indoor environmental readings.
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        base_temperature_c: float = 23.0,
        base_humidity_rh: float = 45.0,
        base_co2_ppm: float = 650.0,
        base_pm25_ug_m3: float = 8.0,
    ) -> None:
        self._rng = random.Random(seed)

        self._temp = base_temperature_c
        self._rh = base_humidity_rh
        self._co2 = base_co2_ppm
        self._pm25 = base_pm25_ug_m3

        self._spike_decay = {
            "co2": 0.0,
            "pm25": 0.0,
        }

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def next_reading(self, now_utc: Optional[datetime] = None) -> SensorReading:
        """
        Produce the next reading. Intended to be called every ~5 seconds.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        # Daily cycle (subtle indoor drift)
        seconds_in_day = 24 * 60 * 60
        t = now_utc.timestamp() % seconds_in_day
        daily_phase = 2.0 * math.pi * (t / seconds_in_day)

        # Temperature: gentle sinusoid + random walk + noise
        temp_target = 23.0 + 1.2 * math.sin(daily_phase)  # 21.8..24.2
        self._temp += 0.02 * (temp_target - self._temp) + self._rng.gauss(0.0, 0.03)
        self._temp = self._clamp(self._temp, 18.0, 30.0)

        # Humidity: inversely correlated with temperature + noise
        rh_target = 45.0 - 4.0 * math.sin(daily_phase)  # 41..49
        self._rh += 0.03 * (rh_target - self._rh) + self._rng.gauss(0.0, 0.12)
        self._rh = self._clamp(self._rh, 20.0, 75.0)

        # CO2: occupancy-like random walk; ventilation events cause drops
        occupancy_push = self._rng.choice([0.0, 0.2, 0.5, 0.9, 1.2]) * self._rng.random()
        ventilation_pull = self._rng.choice([0.0, 0.0, 0.0, -1.5, -3.0]) * self._rng.random()
        self._co2 += occupancy_push + ventilation_pull + self._rng.gauss(0.0, 3.0)

        # Occasionally: big CO2 spike (meeting/people enter)
        if self._rng.random() < 0.01:
            self._spike_decay["co2"] += self._rng.uniform(150.0, 450.0)

        # Exponential decay of spikes
        self._spike_decay["co2"] *= 0.92
        self._co2 += self._spike_decay["co2"]
        self._co2 = self._clamp(self._co2, 400.0, 3000.0)

        # PM2.5: baseline + occasional sharp spikes (cooking, dust) + decay
        self._pm25 += self._rng.gauss(0.0, 0.15)
        self._pm25 += 0.01 * (8.0 - self._pm25)  # mean reversion to baseline

        if self._rng.random() < 0.008:
            self._spike_decay["pm25"] += self._rng.uniform(8.0, 60.0)

        self._spike_decay["pm25"] *= 0.88
        self._pm25 += self._spike_decay["pm25"]
        self._pm25 = self._clamp(self._pm25, 0.0, 500.0)

        return SensorReading(
            ts_utc=now_utc,
            temperature_c=round(self._temp, 2),
            humidity_rh=round(self._rh, 2),
            co2_ppm=round(self._co2, 1),
            pm25_ug_m3=round(self._pm25, 1),
        )

