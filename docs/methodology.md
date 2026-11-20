# Technical Methodology: Spatio-Temporal Retail & Linebuster Analytics

## 1. Data Ingestion, Normalization & Privacy Controls

### 1.1 Read-Only Isolation & Table Allowlisting (`src/db_loader.py`)
- All SQLite connections open strictly with URI mode `?mode=ro`, preventing accidental writes or schema mutations.
- Queries are gated by an explicit `MASTER_ALLOWLIST` (`Summary`, `Items`, `Hourly_Sales`, `Master_Financials`, `Contextual_Split`, `Sales_Channels`, `Day_of_Week_Sales`) and `KIOSK_ALLOWLIST` (`active_branches` restricted to non-sensitive metadata columns).
- Sensitive/PII/financial tables (`client_vault`, `billing_logs`, `Team_Directory`, `Active_Team`, `Horarios`, `Google_Reviews`, `Bank_Transactions`, `Expenses`, `Loans_Schedule`, `Needs_Review`, `Tips_Log`, `App_Clients`) are hard-blocked via `TABLE_DENYLIST` and `RestrictedTableError`.

### 1.2 Multi-Period Schema Coalescing & Sanitation
- **Branch Coalescing:** Rows tagged `DEFAULT` and `TC-001` are merged per date/key by preferring `TC-001` and coalescing blank (`""`/`NULL`) fields from `DEFAULT`.
- **Operating Hour Clamp (`06:00–21:00`):** `Hourly_Sales` entries are clamped to `range(6, 22)` (`6 <= hour_int <= 21`), eliminating off-hours export shift artifacts while reindexing active dates across all 16 operating slots (`fill_value=0.0`) before computing hourly profiles.
- **Category & SKU Normalization:** Historical `Items` (coverage through `2026-08-20`) undergo UTF-8/latin-1 mojibake repair and deterministic 4-way classification (`Beverages`, `Food_Pastries`, `Combos`, `Retail`). Post-August category shares combine `Master_Financials` (`food`, `drink`, `sales_food`, `sales_hot_drinks`, `sales_cold_drinks`) with a cumulative-anomaly guard (`value <= 1.05 * gross_sales_local`).

---

## 2. Geo-Fenced OSINT & Spatial Lag Modeling (`src/osint_collector.py`, `src/spatial_models.py`)

### 2.1 Event Classification & Geocoding
- Local RSS/Atom feeds and CSV/JSON logs are classified into four operational disruption categories using accent-folded Spanish/English keyword lexicons: `traffic_accident`, `road_closure`, `public_safety`, and `weather_disruption`.
- Coordinates resolve first via explicit GeoRSS/CSV coordinates, second via an offline Silao, Guanajuato corridor gazetteer (`SILAO_GAZETTEER`), and optionally via Nominatim when explicitly opted in.

### 2.2 Haversine Geofence Decay & Distributed Lag Regression
- Store-to-incident distance uses great-circle Haversine distance (`R = 6,371,000 m`), mapped to concentric geofence rings (`0–500 m`, `500–1500 m`, `1500–5000 m`, `>5000 m`) and exponential distance-decay weights:
  $$w(d) = \exp\!\left(-\frac{\ln 2 \cdot d}{\lambda_{1/2}}\right), \quad \lambda_{1/2} = 1500\text{ m}$$
- Daily and hourly baseline sales are residualized via OLS controlling for linear trend and day-of-week seasonality (plus hour-of-day dummies for hourly series):
  $$r_t = \frac{y_t - \hat{y}_t}{\hat{y}_t}$$
- Chronological calendar-grid lags ($t$, $t+1$, $t+2$) test acute and carry-over street shock effects via joint OLS:
  $$r_t = \alpha + \beta_0 E_t + \beta_1 E_{t-1} + \beta_2 E_{t-2} + \varepsilon_t$$

---

## 3. Prospective (Ex-Ante) Linebuster Business Case (`src/causal_eval.py`)

Because `LINEBUSTER_START_DATE = None` prior to pilot launch, the evaluation models expected return ex ante rather than misinterpreting historical reporting changes:

1. **Peak Bottleneck Profiling:** Quantifies peak-hour (`08:00–09:59`, `18:00–20:59`) revenue share and empirical `P90` / `P95` daily peak throughput ceilings from observed `Hourly_Sales` volumes.
2. **OSINT-Normalized Demand Baseline:** Excludes days affected by nearby Silao external incidents (`<= 1500 m`) so capacity and ticket baselines represent normal storefront demand.
3. **Channel Segmentation Diagnostic (`2026-09-05`):** Audits the appearance of `"Tablet"` in `Sales_Channels` / `Contextual_Split` strictly according to `TABLET_CHANNEL_INTERPRETATION` (`"pos_channel_change"` by default), never as a linebuster rollout.
4. **3-Scenario Peak-Only Lift & Financial Sensitivity:** Applies walk-away queue recovery (`+5%`, `+10%`, `+15%`) and optional attach-rate basket lift (`+0%`, `+2%`, `+4%`) **strictly to peak-window revenue**, keeping off-peak revenue unchanged. Evaluates break-even incremental tickets/day, monthly net incremental profit, payback period in months, and a `gross_margin_pct × peak_lift_pct` sensitivity matrix.

---

## 4. Post-Launch Pilot Measurement Plan (DiD & ITS)

Once `LINEBUSTER_START_DATE` is set for a live store pilot, `src/causal_eval.py` provides two pre-validated causal estimators:
- **Within-Day Difference-in-Differences (`estimate_did`):** Compares peak treatment hours vs. same-day off-peak control hours before and after launch with date-clustered standard errors:
  $$y_{dt} = \alpha + \gamma\,\text{Peak}_h + \delta\,\text{Post}_d + \tau\,(\text{Peak}_h \times \text{Post}_d) + X_d'\beta + u_{dh}$$
- **Interrupted Time Series (`estimate_interrupted_time_series`):** Evaluates daily level shift and post-launch slope change using Newey-West HAC standard errors.
