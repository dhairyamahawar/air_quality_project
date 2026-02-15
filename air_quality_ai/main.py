"""
Main data collection loop.

Runs the simulator every 5 seconds, computes AQI, and stores to SQLite.
"""

from __future__ import annotations

import argparse
import logging
import time

from aqi import calculate_aqi
from database import Database
from simulator import SensorSimulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart Room Air Quality Simulator")
    p.add_argument("--db", default="air_quality.db", help="SQLite DB path")
    p.add_argument("--interval", type=float, default=5.0, help="Sampling interval (seconds)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for simulator")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    db = Database(args.db)
    db.initialize()

    sim = SensorSimulator(seed=args.seed)

    logging.info("Collecting readings every %.1fs -> %s", args.interval, args.db)
    try:
        while True:
            reading = sim.next_reading()
            aqi, status = calculate_aqi(reading.pm25_ug_m3)
            db.insert_reading(reading, aqi=aqi)

            logging.info(
                "T=%.2fC RH=%.1f%% CO2=%.0fppm PM2.5=%.1f ug/m3 | AQI=%d (%s)",
                reading.temperature_c,
                reading.humidity_rh,
                reading.co2_ppm,
                reading.pm25_ug_m3,
                aqi,
                status,
            )
            time.sleep(max(0.1, float(args.interval)))
    except KeyboardInterrupt:
        logging.info("Stopped.")


if __name__ == "__main__":
    main()
