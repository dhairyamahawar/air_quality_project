"""
Streamlit dashboard:
- Live charts for sensors and AQI
- AQI gauge
- Alerts
- Predictions (next 1 hour)
- Anomaly warnings

Run:
  streamlit run dashboard.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from aqi import aqi_status
from ai_model import detect_anomalies, load_models, predict_next_hour, train_models
from database import Database


st.set_page_config(page_title="Smart Room Air Quality AI", layout="wide")


def _gauge(aqi: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=int(aqi),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "AQI"},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 50], "color": "#2ecc71"},
                    {"range": [50, 100], "color": "#f1c40f"},
                    {"range": [100, 500], "color": "#e74c3c"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=260)
    return fig


def _load_df(db_path: str, limit: int) -> pd.DataFrame:
    db = Database(db_path)
    db.initialize()
    rows = db.fetch_latest(limit=limit)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_unix"], unit="s", utc=True)
    return df


st.title("Smart Room Air Quality Monitoring + AI Predictions (Simulated)")

with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("SQLite DB path", value="air_quality.db")
    points = st.slider("History points", min_value=120, max_value=3000, value=720, step=120)
    auto = st.toggle("Auto refresh", value=True)
    refresh_secs = st.slider("Refresh seconds", min_value=2, max_value=30, value=5, step=1)
    if auto:
        st_autorefresh(interval=refresh_secs * 1000, key="refresh")


df = _load_df(db_path, limit=int(points))

if df.empty:
    st.info("No data yet. Start the collector first: `python main.py`")
    st.stop()

latest = df.iloc[-1]
latest_aqi = int(latest["aqi"])
status = aqi_status(latest_aqi)

col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
with col1:
    st.plotly_chart(_gauge(latest_aqi), use_container_width=True)
with col2:
    st.metric("Status", status)
    st.metric("PM2.5 (ug/m3)", f'{float(latest["pm25_ug_m3"]):.1f}')
with col3:
    st.metric("CO2 (ppm)", f'{float(latest["co2_ppm"]):.0f}')
    st.metric("Humidity (%)", f'{float(latest["humidity_rh"]):.1f}')
with col4:
    st.metric("Temperature (C)", f'{float(latest["temperature_c"]):.2f}')
    st.caption(f'Last update: {pd.to_datetime(int(latest["ts_unix"]), unit="s", utc=True).to_pydatetime().isoformat()}')


st.subheader("Live Trends")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["aqi"], name="AQI"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
with chart_col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["pm25_ug_m3"], name="PM2.5 (ug/m3)"))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["co2_ppm"], name="CO2 (ppm)", yaxis="y2"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="PM2.5"),
        yaxis2=dict(title="CO2", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["ts"], y=df["temperature_c"], name="Temp (C)"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["humidity_rh"], name="Humidity (%)", yaxis="y2"))
fig.update_layout(
    height=280,
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="Temp"),
    yaxis2=dict(title="Humidity", overlaying="y", side="right"),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig, use_container_width=True)


st.subheader("Alerts")
alerts = []
if float(latest["co2_ppm"]) >= 1200:
    alerts.append("High CO2 detected (consider ventilation).")
if float(latest["pm25_ug_m3"]) >= 35:
    alerts.append("Elevated PM2.5 detected (possible smoke/dust/cooking).")
if latest_aqi > 100:
    alerts.append("AQI is Unhealthy - take action.")
elif latest_aqi > 50:
    alerts.append("AQI is Moderate - monitor conditions.")

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("No active alerts.")


st.subheader("AI Models")
forecast_model, anomaly_model, meta = load_models()
meta_text = f"Trained at: {meta.get('trained_at', 'N/A')} | MAE: {meta.get('mae', 'N/A')}"
st.caption(meta_text)

train_col1, train_col2 = st.columns([1, 3])
with train_col1:
    if st.button("Train / Retrain", type="primary"):
        try:
            report = train_models(df)
            st.success(f"Trained on {report.rows_used} rows | MAE={report.mae:.2f}")
        except Exception as e:
            st.error(f"Training failed: {e}")

with train_col2:
    st.write(
        "Forecast: RandomForestRegressor (iterative 1-hour).  "
        "Anomaly: IsolationForest over sensor + AQI features."
    )


pred_col1, pred_col2 = st.columns([1.2, 1])

with pred_col1:
    st.subheader("Next 1 Hour AQI Prediction")
    if forecast_model is None:
        st.info("No forecast model saved yet. Click Train / Retrain.")
    else:
        try:
            pred = predict_next_hour(df, step_seconds=5, horizon_minutes=60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred["ts"], y=pred["pred_aqi"], name="Pred AQI", opacity=0.5))
            fig.add_trace(go.Scatter(x=pred["ts"], y=pred["pred_aqi_smooth"], name="Smoothed", line=dict(width=3)))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with pred_col2:
    st.subheader("Anomaly Warnings")
    if anomaly_model is None:
        st.info("No anomaly model saved yet. Click Train / Retrain.")
    else:
        try:
            labeled = detect_anomalies(df.tail(1000))
            recent_anoms = labeled[labeled["anomaly"]].tail(10)
            latest_is_anom = bool(labeled.iloc[-1]["anomaly"])
            if latest_is_anom:
                st.error("Latest reading looks anomalous.")
            else:
                st.success("Latest reading looks normal.")
            if not recent_anoms.empty:
                st.write("Recent anomalies:")
                st.dataframe(
                    recent_anoms[["ts_iso", "temperature_c", "humidity_rh", "co2_ppm", "pm25_ug_m3", "aqi", "anomaly_score"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No recent anomalies detected.")
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")
