# Executive Business Case: Prospective Peak-Hour Linebuster Deployment

**Store / Franchise Unit:** {{store_name}} (`{{branch_id}}`, {{city}})  
**Analysis Mode:** `{{data_mode}}` | **Observation Window:** {{date_range}}  
**Coordinates Status:** {{coords_status}}  
**Linebuster Deployment Status:** Prospective / Ex-Ante (`LINEBUSTER_START_DATE = None`)

---

## 1. Executive Summary

- **OSINT-Normalized Baseline:** Across {{total_days}} observed days, {{incident_days_excluded}} external-disruption day(s) (`<= 1500 m`) were excluded to isolate clean underlying demand across {{clean_days}} clean days (**{{clean_mean_daily_revenue}} {{currency}}/day**, avg ticket **{{clean_mean_avg_ticket}} {{currency}}**, **{{clean_mean_tickets_per_day}}** tickets/day).
- **Peak Congestion Bottleneck:** Peak windows (**{{peak_hours_label}}**) concentrate **{{peak_revenue_share_pct}}** of daily sales (**{{clean_mean_daily_peak_revenue}} {{currency}}/day**). Empirical top-decile throughput proves headroom up to **{{peak_p90_ceiling}} {{currency}}/day** (P90) and **{{peak_p95_ceiling}} {{currency}}/day** (P95) during peak windows.
- **Base-Case ROI:** Under the Base scenario (+10% peak queue recovery, +2% attach lift), projected net incremental profit is **{{base_monthly_net_profit}} {{currency}}/month** with a **{{base_payback_months}}-month** hardware payback and a break-even hurdle of **{{base_breakeven_tickets_per_day}}** incremental tickets/day.

---

## 2. Diagnostic Guardrails & Segmentation Audit

- **2026-09-05 Channel Audit (`{{tablet_start_date}}`):** Treated strictly as `{{tablet_interpretation}}` (POS reporting segmentation check, **not** a linebuster deployment).
- **Attach-Rate Proxies:** Historical food revenue share is **{{food_revenue_share_pct}}**; pre-cutoff (`<= 2026-08-20`) combo unit share is **{{combo_unit_share_pct}}**.
- **Cost & Margin Assumptions (`{{currency}}`):** Hardware CapEx **{{hardware_capex}}**, Monthly SaaS **{{monthly_saas_fee}}**, Peak Labor **{{peak_labor_cost_per_hr}}/hr**, Blended Gross Margin **{{gross_margin_pct}}**.

---

## 3. Prospective 3-Scenario Simulation (Peak Hours Only)

{{scenarios_table_md}}

---

## 4. Sensitivity Matrix: Monthly Net Profit ({{currency}}) & Payback (Months)

{{sensitivity_table_md}}

---

## 5. Post-Approval Pilot Measurement Plan (60-Day Trial)

1. **Set `LINEBUSTER_START_DATE`:** Record exact pilot go-live date in `src/config.py` or `data/raw/local_assumptions.json`.
2. **Primary Causal Estimator (Within-Day Difference-in-Differences / DiD):** Run `estimate_did()` comparing peak treatment hours (`{{peak_hours_label}}`) against same-day off-peak hours with date-clustered standard errors and OSINT incident controls.
3. **Secondary Trend Estimator (Interrupted Time Series / ITS):** Run `estimate_interrupted_time_series()` with HAC standard errors to separate immediate queue-speed step lift from sustained adoption slope.
4. **Go / No-Go Gate (Day 30 & Day 60):** Expand to additional franchise units if observed peak-hour ATT exceeds break-even (**{{base_breakeven_tickets_per_day}}** tickets/day equivalent) at `p < 0.05` and realized payback remains under 6 months.
