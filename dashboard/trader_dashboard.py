import streamlit as st
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Intraday Energy Desk")

st.title("⚡ Nordic Intraday Energy Trading Desk")

# Sidebar
st.sidebar.header("Controls")
zone = st.sidebar.selectbox("Select Bidding Zone", ["DK1", "DK2", "SE3", "NO2"])
st.sidebar.metric("Current NOP (Net Open Position)", "45.5 MW", delta="-10 MW")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Live Orderbook & Imbalance: {zone}")
    times = [datetime.now() + timedelta(minutes=15*i) for i in range(20)]
    prices = np.random.normal(50, 15, 20)
    st.line_chart({"Time": times, "VWAP Price (EUR/MWh)": prices}, x="Time", y="VWAP Price (EUR/MWh)")

with col2:
    st.subheader("Wind Forecast vs Realized Generation")
    wind_forecast = np.random.normal(1200, 200, 20)
    wind_actual = wind_forecast + np.random.normal(0, 50, 20)
    
    st.line_chart({
        "Time": times,
        "Forecast (MW)": wind_forecast,
        "Actual (MW)": wind_actual
    }, x="Time", y=["Forecast (MW)", "Actual (MW)"])

st.subheader("System Alerts")
st.error("🚨 [WIND SHOCK DETECTED] ECMWF Revised DK1 Wind down by 650MW for 14:00 Delivery. Expected Spread widening vs DE.")
st.warning("⚠️ [GRID ALERT] Energinet reporting reduced ATC on DK1-DE interconnector.")
