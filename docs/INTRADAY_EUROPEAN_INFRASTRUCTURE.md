# Intraday European Energy Trading Infrastructure Design

## Executive Summary
This document outlines the design, critique, and iterative improvement of a production-grade data and research pipeline supporting an intraday European energy trading desk (focusing on Nordic markets like DK1/DK2, Nord Pool, EEX, TTF, and NBP). This system is designed for a multi-manager hedge fund requiring robust, low-latency, and highly scalable infrastructure.

---

## STEP 1: First Full Architecture Design

### 1. Data Ingestion Architecture
- **APIs & Schedulers:** 
  - **Streaming:** WebSocket/FIX connections for market data (Nord Pool, EEX).
  - **Batch:** REST APIs/FTP for weather (ECMWF, NOAA) and grid data (ENTSO-E Transparency Platform, Energinet), scheduled via **Apache Airflow**.
- **Message Broker:** **Apache Kafka** is used to ingest and decouple incoming streams. Dedicated topics are set up for `market_data_tick`, `weather_updates`, and `grid_status`.
- **Fault Tolerance:** Kafka handles retries. Missing batch data triggers Airflow sensor alerts.

### 2. Data Storage Layer
- **Real-Time/Tick Data:** **ClickHouse** (columnar) for blazing-fast aggregations of order book data and trades.
- **Relational/Metadata:** **PostgreSQL** for instrument reference data, trading schedules, and mapping (e.g., matching a weather station to a price zone).
- **Historical/Research Storage:** **AWS S3 / MinIO** using **Parquet/Arrow** formats, partitioned by `year=YYYY/month=MM/zone=ZONE`.

### 3. Data Processing and Feature Engineering
- **Engine:** **Python (Polars/Pandas)** running on Kubernetes.
- **Transformations:**
  - **Alignment:** Weather forecasts (often hourly or 3-hourly) are interpolated/forward-filled to match 15-min or 5-min intraday trading intervals.
  - **Features:** Generation of rolling volatility, bid/ask imbalance, VWAP, and lag features (e.g., T-1 hour wind error).
  - **Grid Indicators:** Calculating theoretical vs. actual cross-border flows.

### 4. Modeling Layer
- **Models Used:** 
  - **Gradient Boosting (LightGBM/XGBoost):** Primary workhorse for power price forecasting and zone spread prediction due to non-linear relationships in energy grids.
  - **ARIMA/SARIMAX:** Baseline for load/demand forecasting.
  - **LSTMs:** Explored for very short-term imbalance prediction capturing temporal dependencies in order book flow.
- **Deployment:** Models are serialized (e.g., ONNX) and served via FastAPI or directly embedded in the trading engine for low-latency scoring.

### 5. Backtesting Engine
- **Core:** Event-driven architecture (already present in our repo) that simulates tick-by-tick execution.
- **Features:** 
  - **Slippage Modeling:** Modeled as a function of order size relative to top-of-book volume.
  - **Transaction Costs:** Explicit exchange fees (e.g., Nord Pool clearing fees) incorporated into PnL.
  - **Position Management:** Strict limits on net open position per zone (DK1/DK2) to avoid physical delivery defaults.

### 6. Visualization and Trader Tools
- **Framework:** **Dash/Plotly** web application.
- **Dashboards:**
  - **Forecast vs. Realized:** Overlay of ECMWF wind forecast vs. Energinet real-time production.
  - **Spread Relationships:** Real-time charting of DK1-DK2 and DK1-DE (Germany) spreads.
  - **Order Book Heatmap:** Depth visualization for Nord Pool intraday.

### 7. Alerting and Monitoring
- **Tools:** **Prometheus** for metrics, **Grafana** for dashboards, and **PagerDuty/Slack** for alerts.
- **Triggers:**
  - Data staleness (e.g., ENTSO-E API delay > 5 mins).
  - Sudden wind forecast revisions (> 500MW change in a single run).
  - Abnormal spread widening beyond historical 2-sigma bounds.

### 8. Performance Considerations
- **Latency:** Target < 50ms for market data to signal generation.
- **Memory Efficiency:** Polars used for out-of-core processing; ClickHouse handles heavy aggregations.
- **Scaling:** Kubernetes auto-scaling for feature engineering workers during market open / auction times.

### 9. Technology Stack
- **Languages:** Python (Research/Logic), C++ or Rust (Execution - future scope).
- **Data:** Kafka, ClickHouse, Postgres, S3, Parquet/Arrow.
- **Compute/Orchestration:** Kubernetes, Docker, Apache Airflow.

### 10. Example Intern Project: "Intraday Wind Shock Detection System"
- **Goal:** Build an automated pipeline that detects sudden revisions in weather forecasts.
- **Steps:**
  1. Ingest continuous ECMWF/GFS updates.
  2. Diff current forecast vs. T-1 forecast for Nordic wind parks.
  3. Map MW changes to expected price impact using a historical linear regression model.
  4. Push alerts to a dedicated Slack channel for the intraday desk.
  5. Visualize the delta on a Dash web app.

---

## STEP 2: Critique of First Architecture

**Weaknesses Identified:**
1. **Latency in Feature Engineering:** Relying on Python (Polars/Pandas) running inside Kubernetes pods reading from Kafka introduces serialization/deserialization overhead. For true intraday trading where milliseconds matter, Python is too slow for real-time order book feature generation.
2. **Data Alignment Issues:** Forward-filling weather data is dangerous. A forecast published at 10:00 AM predicting a drop at 11:00 AM cannot simply be interpolated. "As-of" time mapping (bitemporal modeling) is completely missing.
3. **Backtester Unrealistic Assumptions:** The backtester does not account for the "gate closure" times of physical power markets. In European intraday, you cannot trade physical power right up to the second of delivery; there are strict transmission system operator (TSO) cut-offs.
4. **ClickHouse Overload:** Pushing raw order book updates directly into ClickHouse for real-time querying by trading models will cause high query latency under load.

---

## STEP 3: Improved Architecture (Iteration 2)

**Modifications:**
1. **Bitemporal Data Architecture:** 
   - Introduced a strict `knowledge_time` (when did we know this?) and `event_time` (what time does this forecast apply to?) schema in ClickHouse and Parquet. This guarantees zero look-ahead bias in the backtester.
2. **C++ / Rust Feature Engine:** 
   - Moved critical path feature engineering (order book imbalance, rolling VWAP) from Python to C++. Python is retained strictly for model training and daily batch jobs. 
   - Real-time signals are written directly to shared memory or a low-latency bus (e.g., Aeron) rather than Kafka.
3. **Market Microstructure Fidelity:**
   - The backtester now explicitly models **Gate Closure**. Orders are forcibly rejected if submitted after the local TSO cut-off (e.g., 60 mins before delivery in DK1).
4. **Separation of Storage Concerns:**
   - **Redis (In-Memory):** Placed in front of ClickHouse to serve the latest state (e.g., current order book snapshot, latest wind forecast) to the live trading engine. ClickHouse is relegated to historical research and end-of-day analytics.

---

## STEP 4: Critique of Iteration 2

**Weaknesses Identified:**
1. **Complexity overhead:** Introducing Aeron and C++ alongside Python, Kafka, and Redis creates a massive operational burden for a small quant desk. 
2. **Imbalance Market Blindspot:** The system predicts spreads and prices, but European trading profitability heavily relies on the balancing market (the cost of being physically short/long after gate closure). The architecture doesn't explicitly handle TSO regulation states (up-regulating vs. down-regulating).
3. **Cross-Border Flow Constraints:** DK1 and DK2 are heavily influenced by cables to Norway, Sweden, and Germany. The model treats these as static features, but Available Transmission Capacity (ATC) is a dynamic constraint published by Nord Pool.

---

## STEP 5: Final Production-Grade Architecture (Iteration 3)

This represents the realistic equilibrium found at top-tier commodity hedge funds.

### The Final Pipeline
1. **Data Ingestion (The "Collector"):**
   - **Market Data:** Rust-based collectors attach to Nord Pool and EEX APIs, normalizing data into a proprietary binary format and publishing to **Kafka**.
   - **Fundamental/Weather:** Python workers orchestrated by **Prefect** (modern alternative to Airflow) pull ENTSO-E (ATC, balancing states) and ECMWF data.
2. **Bitemporal Storage Layer:**
   - **ArcticDB / kdb+:** Implemented for native bitemporal timeseries storage. This handles the complex versioning of weather forecasts inherently, replacing the clunky S3/Parquet custom setup.
3. **Feature Engineering & Signal Generation:**
   - We utilize a **hybrid approach**. 
   - *Fast Path:* Order book features (microstructure) are computed in Rust/C++ in memory. 
   - *Slow Path:* Fundamental features (wind forecasts, ATC constraints, balancing state probabilities) are computed in Python/Polars and pushed to **Redis** to be read by the execution engine.
4. **Trading Engine & Execution:**
   - Incorporates a **Constraint Optimizer**. Before generating an order, the engine checks ATC (transmission limits) and gate closure times. 
   - Explicitly models the penalty of the **Imbalance Price**. If the system forecasts DK1 to be in an "Up-regulation" state, it biases the market-making algorithms to hold a long physical position into delivery, as the TSO will pay a premium for excess power.
5. **Backtesting:**
   - Uses an event-driven framework that replays the exact Kafka logs (Market Data + Fundamentals) exactly as they appeared at `knowledge_time`. 
   - Includes market-impact models specific to power (e.g., clearing the book on illiquid hours).
6. **Monitoring:**
   - **Datadog** for system health. 
   - **Custom Streamlit Dashboards** for traders, featuring live PnL, current Net Open Position (NOP) per bidding zone, and the Intern's "Wind Shock" alerts directly integrated into the UI.
