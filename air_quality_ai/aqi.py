"""
AQI (Air Quality Index) calculation utilities.

This implementation focuses on PM2.5 -> AQI using US EPA breakpoints.
CO2 is not part of EPA AQI; we still track CO2 separately for alerts.
"""

from __future__ import annotations

from typing import Tuple


# US EPA PM2.5 (24-hour) AQI breakpoints (ug/m3).
# Source: EPA AQI breakpoints (commonly referenced); used here for a simulation demo.
_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25_ug_m3: float) -> int:
    """
    Convert PM2.5 concentration to AQI via piecewise linear interpolation.
    """
    c = max(0.0, float(pm25_ug_m3))
    for c_lo, c_hi, i_lo, i_hi in _PM25_BREAKPOINTS:
        if c_lo <= c <= c_hi:
            # Linear interpolation:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo
            return int(round(aqi))
    # Beyond defined range: clamp to 500
    return 500


def aqi_status(aqi: int) -> str:
    """
    Return a simplified status label required by the project spec.
    """
    a = int(aqi)
    if a <= 50:
        return "Good"
    if a <= 100:
        return "Moderate"
    return "Unhealthy"


def calculate_aqi(pm25_ug_m3: float) -> Tuple[int, str]:
    aqi = pm25_to_aqi(pm25_ug_m3)
    return aqi, aqi_status(aqi)
