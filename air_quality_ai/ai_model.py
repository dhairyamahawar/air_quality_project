"""
AI layer:
- Moving average smoothing
- RandomForestRegressor forecasting (next 1 hour AQI)
- IsolationForest anomaly detection
- Joblib persistence
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_MODEL_PATH = MODEL_DIR / "aqi_forecast_rf.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_iforest.joblib"
META_PATH = MODEL_DIR / "model_meta.joblib"


def moving_average(values: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Simple moving average (SMA). window=12 corresponds to 1 minute at 5s intervals.
    """
    window = int(max(1, window))
    if values.size == 0:
        return values
    if window == 1:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / window
    # "same" keeps length unchanged; edges are less accurate but fine for dashboard smoothing.
    return np.convolve(values.astype(float), kernel, mode="same")


def _ensure_dataframe(rows_or_df: pd.DataFrame | list[dict]) -> pd.DataFrame:
    if isinstance(rows_or_df, pd.DataFrame):
        df = rows_or_df.copy()
    else:
        df = pd.DataFrame(rows_or_df)

    if df.empty:
        return df

    # Normalize timestamps (stored as ts_unix in seconds)
    if "ts_unix" in df.columns:
        df["ts_unix"] = df["ts_unix"].astype(int)
        df["ts"] = pd.to_datetime(df["ts_unix"], unit="s", utc=True)
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["ts_unix"] = (df["ts"].astype("int64") // 1_000_000_000).astype(int)
    else:
        raise ValueError("Expected 'ts_unix' or 'ts' in data.")

    df = df.sort_values("ts_unix").reset_index(drop=True)
    return df


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Supervised learning frame:
    - Features from lagged AQI & PM2.5 and current sensor values
    - Target is next-step AQI (t+1)
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    needed = {"temperature_c", "humidity_rh", "co2_ppm", "pm25_ug_m3", "aqi"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    X = pd.DataFrame(index=df.index)

    # Current sensors
    X["temperature_c"] = df["temperature_c"].astype(float)
    X["humidity_rh"] = df["humidity_rh"].astype(float)
    X["co2_ppm"] = df["co2_ppm"].astype(float)
    X["pm25_ug_m3"] = df["pm25_ug_m3"].astype(float)
    X["aqi"] = df["aqi"].astype(float)

    # Time-of-day features (help with daily drift patterns)
    seconds_in_day = 24 * 60 * 60
    tod = (df["ts_unix"] % seconds_in_day).astype(float)
    X["tod_sin"] = np.sin(2 * np.pi * tod / seconds_in_day)
    X["tod_cos"] = np.cos(2 * np.pi * tod / seconds_in_day)

    # Lag features (5s step): 1, 12 (~1 min), 60 (~5 min), 120 (~10 min)
    for lag in (1, 12, 60, 120):
        X[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
        X[f"pm25_lag_{lag}"] = df["pm25_ug_m3"].shift(lag)

    # Rolling means
    X["aqi_ma_12"] = df["aqi"].rolling(window=12, min_periods=1).mean()
    X["pm25_ma_12"] = df["pm25_ug_m3"].rolling(window=12, min_periods=1).mean()

    y = df["aqi"].shift(-1)

    # Drop NaNs from lagging/shift
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].astype(float)
    y = y.loc[valid].astype(float)
    return X, y


@dataclass(frozen=True)
class TrainReport:
    rows_used: int
    mae: float


def train_models(rows_or_df: pd.DataFrame | list[dict]) -> TrainReport:
    """
    Train forecast and anomaly models from historical DB data and persist them.
    """
    df = _ensure_dataframe(rows_or_df)
    if len(df) < 400:
        # ~33 minutes of data at 5s interval
        raise ValueError("Not enough data to train yet (need ~400+ rows).")

    X, y = _build_features(df)
    if len(X) < 200:
        raise ValueError("Not enough valid feature rows after lagging.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    forecast = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    forecast.fit(X_train, y_train)

    preds = forecast.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))

    # Anomaly model: use sensor + aqi features (no target)
    anomaly_features = df[["temperature_c", "humidity_rh", "co2_ppm", "pm25_ug_m3", "aqi"]].astype(float)
    iforest = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=42,
    )
    iforest.fit(anomaly_features)

    joblib.dump(forecast, FORECAST_MODEL_PATH)
    joblib.dump(iforest, ANOMALY_MODEL_PATH)
    joblib.dump({"trained_at": datetime.now(timezone.utc).isoformat(), "mae": mae}, META_PATH)

    return TrainReport(rows_used=len(X), mae=mae)


def load_models() -> Tuple[Optional[RandomForestRegressor], Optional[IsolationForest], Dict]:
    forecast = None
    anomaly = None
    meta: Dict = {}
    if FORECAST_MODEL_PATH.exists():
        forecast = joblib.load(FORECAST_MODEL_PATH)
    if ANOMALY_MODEL_PATH.exists():
        anomaly = joblib.load(ANOMALY_MODEL_PATH)
    if META_PATH.exists():
        meta = joblib.load(META_PATH)
    return forecast, anomaly, meta


def predict_next_hour(
    rows_or_df: pd.DataFrame | list[dict],
    *,
    step_seconds: int = 5,
    horizon_minutes: int = 60,
) -> pd.DataFrame:
    """
    Predict AQI for the next hour as a time series.

    Implementation details:
    - Trains model to predict next-step AQI (t+1)
    - Forecasts horizon by iteratively feeding predictions back as lag features
    - Assumes sensor values hold near their last observed values with mild drift
    """
    forecast, _, _ = load_models()
    if forecast is None:
        raise FileNotFoundError("Forecast model not found. Train the model first.")

    df = _ensure_dataframe(rows_or_df)
    if df.empty or len(df) < 200:
        raise ValueError("Need recent history (~200+ rows) to forecast.")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts_unix"], unit="s", utc=True)

    # Start state from the last observed row
    last = df.iloc[-1]
    temp = float(last["temperature_c"])
    rh = float(last["humidity_rh"])
    co2 = float(last["co2_ppm"])
    pm25 = float(last["pm25_ug_m3"])

    # Keep a rolling history of AQI/PM2.5 for lag features.
    hist_aqi = df["aqi"].astype(float).to_list()
    hist_pm25 = df["pm25_ug_m3"].astype(float).to_list()

    steps = int((horizon_minutes * 60) / step_seconds)
    start_ts = int(last["ts_unix"])

    out_ts = []
    out_aqi = []

    rng = np.random.default_rng(42)

    for i in range(1, steps + 1):
        ts_unix = start_ts + i * step_seconds

        # Mild sensor drift assumption (random walk with mean reversion)
        temp += float(rng.normal(0.0, 0.01)) + 0.002 * (23.0 - temp)
        rh += float(rng.normal(0.0, 0.03)) + 0.003 * (45.0 - rh)
        co2 += float(rng.normal(0.0, 2.0)) + 0.01 * (700.0 - co2)
        pm25 += float(rng.normal(0.0, 0.05)) + 0.01 * (8.0 - pm25)

        # Manually build feature row (faster/clearer than trying to shift on a 1-row DF)
        seconds_in_day = 24 * 60 * 60
        tod = float(ts_unix % seconds_in_day)
        feat = {
            "temperature_c": float(temp),
            "humidity_rh": float(rh),
            "co2_ppm": float(co2),
            "pm25_ug_m3": float(pm25),
            "aqi": float(hist_aqi[-1]),
            "tod_sin": float(np.sin(2 * np.pi * tod / seconds_in_day)),
            "tod_cos": float(np.cos(2 * np.pi * tod / seconds_in_day)),
            "aqi_ma_12": float(np.mean(hist_aqi[-12:])) if len(hist_aqi) >= 12 else float(np.mean(hist_aqi)),
            "pm25_ma_12": float(np.mean(hist_pm25[-12:])) if len(hist_pm25) >= 12 else float(np.mean(hist_pm25)),
        }
        for lag in (1, 12, 60, 120):
            feat[f"aqi_lag_{lag}"] = float(hist_aqi[-lag]) if len(hist_aqi) >= lag else float(hist_aqi[0])
            feat[f"pm25_lag_{lag}"] = float(hist_pm25[-lag]) if len(hist_pm25) >= lag else float(hist_pm25[0])

        X_one = pd.DataFrame([feat]).astype(float)
        pred_aqi = float(forecast.predict(X_one)[0])
        pred_aqi = float(np.clip(pred_aqi, 0.0, 500.0))

        out_ts.append(pd.to_datetime(ts_unix, unit="s", utc=True))
        out_aqi.append(pred_aqi)

        hist_aqi.append(pred_aqi)
        hist_pm25.append(pm25)

    out = pd.DataFrame({"ts": out_ts, "pred_aqi": out_aqi})
    out["pred_aqi_smooth"] = moving_average(out["pred_aqi"].to_numpy(), window=12)
    return out


def detect_anomalies(rows_or_df: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """
    Label each row with anomaly flags using the persisted IsolationForest model.
    """
    _, anomaly, _ = load_models()
    if anomaly is None:
        raise FileNotFoundError("Anomaly model not found. Train the model first.")

    df = _ensure_dataframe(rows_or_df)
    if df.empty:
        return df

    feats = df[["temperature_c", "humidity_rh", "co2_ppm", "pm25_ug_m3", "aqi"]].astype(float)
    # IsolationForest: -1 indicates anomaly
    labels = anomaly.predict(feats)
    scores = anomaly.decision_function(feats)
    out = df.copy()
    out["anomaly"] = labels == -1
    out["anomaly_score"] = scores
    return out
