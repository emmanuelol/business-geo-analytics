"""Unit tests for src.notebook_utils shared notebook analysis and rendering helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import StoreConfig
from src.notebook_utils import (
    attach_dilution_stress_test,
    audit_intraday_incident_hours,
    compute_weighted_dow_summary,
    date_overlap_audit,
    decompose_daily_baseline,
    load_incidents_for_mode,
    pilot_mde_table,
    placebo_lead_test,
    quadrant_summary,
    regime_and_trend_audit,
    resolve_incidents_path,
    ring_confidence_intervals,
    split_weekday_weekend_hourly,
    stratify_residuals_by_category,
    weekday_weekend_peak_table,
)


def test_resolve_incidents_path_isolates_raw_and_sample(tmp_path: Path) -> None:
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "sample").mkdir(parents=True)
    sample_inc = tmp_path / "data" / "sample" / "incidents.csv"
    sample_inc.write_text("incident_id,date,title,category\n1,2026-06-01,Choque,traffic_accident\n", encoding="utf-8")

    cfg = StoreConfig(repo_root=tmp_path)

    # raw mode must never fall back to sample/incidents.csv
    with pytest.raises(FileNotFoundError, match="incidents_raw.csv"):
        resolve_incidents_path(cfg, "raw")

    # sample mode falls back to data/sample/incidents.csv when incidents_sample.csv is absent
    assert resolve_incidents_path(cfg, "sample") == sample_inc

    # when processed incidents exist, both modes pick their own processed file
    proc_sample = tmp_path / "data" / "processed" / "incidents_sample.csv"
    proc_sample.write_text("incident_id,date,title,category\n2,2026-06-02,Cierre,road_closure\n", encoding="utf-8")
    assert resolve_incidents_path(cfg, "sample") == proc_sample

    proc_raw = tmp_path / "data" / "processed" / "incidents_raw.csv"
    proc_raw.write_text("incident_id,date,title,category,lat,lon\n3,2026-06-03,Lluvia,weather_disruption,20.95,-101.42\n", encoding="utf-8")
    df_raw, used_path = load_incidents_for_mode(cfg, "raw", relevant_only=True)
    assert used_path == proc_raw
    assert len(df_raw) == 1


def test_weighted_dow_and_decomposition_and_quincena() -> None:
    dates = pd.date_range("2026-05-01", periods=35, freq="D")
    # Alternate low-ticket/low-rev and high-ticket/high-rev days on the same DOW
    revs = [1000.0 if i % 2 == 0 else 5000.0 for i in range(len(dates))]
    tickets = [10 if i % 2 == 0 else 25 for i in range(len(dates))]
    daily = pd.DataFrame({"date": dates, "revenue": revs, "tickets": tickets})

    dow_tbl = compute_weighted_dow_summary(daily)
    total_weighted = daily["revenue"].sum() / daily["tickets"].sum()
    overall_from_tbl = dow_tbl["total_revenue"].sum() / dow_tbl["total_tickets"].sum()
    assert overall_from_tbl == pytest.approx(total_weighted, rel=1e-6)
    # Unweighted mean of daily avg_ticket would be (100+200)/2=150; weighted is 6000/35=171.43
    assert overall_from_tbl > 165.0

    decomp_df, pay_tbl = decompose_daily_baseline(daily)
    assert {"rolling_14d", "trend", "dow_effect", "is_quincena", "residual_pct"}.issubset(
        decomp_df.columns
    )
    assert set(pay_tbl["pay_cycle"]) == {"Quincena", "Non-Quincena"}


def test_date_overlap_placebo_category_and_intraday() -> None:
    dates = pd.date_range("2026-06-01", periods=20, freq="D")
    joined = pd.DataFrame(
        {
            "date": dates,
            "residual_pct": np.where(dates.day == 10, -0.25, 0.01),
            "has_incident_lag0": dates.day == 10,
        }
    )
    inc = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-10", "2026-06-12"]),
            "hour_int": [9, 18],
            "category": ["traffic_accident", "road_closure"],
            "lat": [20.952, 20.948],
            "lon": [-101.422, -101.418],
        }
    )

    ov = date_overlap_audit(joined, inc)
    assert int(ov.loc[0, "overlap_days"]) == 3
    assert int(ov.loc[0, "incident_days_in_window"]) == 2

    no_ov = date_overlap_audit(
        joined, pd.DataFrame({"date": pd.to_datetime(["2025-01-01"])})
    )
    assert int(no_ov.loc[0, "incident_days_in_window"]) == 0
    assert "WARNING" in str(no_ov.loc[0, "status"])

    placebo = placebo_lead_test(joined)
    assert int(placebo.loc[0, "lag"]) == -1
    assert {"coef", "std_err", "ci_low", "ci_high", "flag_pretrend"}.issubset(placebo.columns)

    cat_tbl = stratify_residuals_by_category(joined, inc)
    assert {"traffic_accident", "no_incident"}.issubset(set(cat_tbl["category"]))

    hourly = pd.DataFrame(
        [
            {"date": d, "hour_int": h, "volume": 300.0 if h in (8, 9, 18) else 150.0}
            for d in dates
            for h in range(6, 22)
        ]
    )
    intraday = audit_intraday_incident_hours(hourly, inc)
    assert not intraday.empty
    assert {"hour_int", "affected_mean_volume", "baseline_mean_volume"}.issubset(
        intraday.columns
    )


def test_spatial_ci_quadrants_and_linebuster_stress_mde() -> None:
    joined = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=10, freq="D"),
            "nearest_ring_lag0": ["0-500m"] * 3 + ["no_incident"] * 7,
            "residual_pct": [-0.12, -0.08, -0.10] + [0.01] * 7,
        }
    )
    ring_summary = pd.DataFrame(
        {
            "ring": ["0-500m", "no_incident"],
            "n_obs": [3, 7],
            "mean_residual_pct": [-0.10, 0.01],
            "std_residual_pct": [0.02, 0.01],
        }
    )
    with_ci = ring_confidence_intervals(joined, ring_summary)
    assert (with_ci["ci_high"] > with_ci["mean_residual_pct"]).all()
    assert (with_ci["ci_low"] < with_ci["mean_residual_pct"]).all()

    enriched = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
            "latitude": [20.96, 20.96, 20.94],
            "longitude": [-101.41, -101.43, -101.43],
        }
    )
    quads = quadrant_summary(enriched, joined, store_lat=20.95, store_lon=-101.42)
    assert {"NE", "NW", "SW"}.issubset(set(quads["quadrant"]))

    hourly = pd.DataFrame(
        [
            {"date": d, "hour_int": h, "volume": 500.0 if h in (8, 9, 18, 19, 20) else 150.0}
            for d in pd.date_range("2026-07-01", periods=20, freq="D")
            for h in range(6, 22)
        ]
    )
    ww = split_weekday_weekend_hourly(hourly, peak_hours=(8, 9, 18, 19, 20))
    assert set(ww["day_type"]) == {"Weekday", "Weekend"}
    pk_tbl = weekday_weekend_peak_table(hourly, peak_hours=(8, 9, 18, 19, 20))
    assert set(pk_tbl["day_type"]) == {"Weekday", "Weekend"}
    assert set(pk_tbl["hour_int"]) == {8, 9, 18, 19, 20}

    cfg = StoreConfig()
    baseline = {
        "clean_mean_daily_revenue": 4500.0,
        "clean_mean_daily_peak_revenue": 2500.0,
        "clean_mean_daily_offpeak_revenue": 2000.0,
        "clean_mean_avg_ticket": 150.0,
        "clean_mean_tickets_per_day": 30.0,
        "peak_p95_daily_ceiling": 3100.0,
    }
    stress = attach_dilution_stress_test(baseline=baseline, cfg=cfg)
    assert list(stress["attach_lift_pct"]) == pytest.approx([-0.05, 0.0, 0.05])
    assert stress["monthly_net_incremental_profit"].is_monotonic_increasing

    clean_daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=20, freq="D"),
            "revenue": np.linspace(3000.0, 3600.0, 20),
        }
    )
    mde = pilot_mde_table(clean_daily, hourly, peak_hours=(8, 9, 18, 19, 20))
    assert list(mde["pilot_days"]) == [14, 28, 42, 56, 84]
    assert mde["daily_mde_pct"].is_monotonic_decreasing
    assert mde["peak_mde_pct"].is_monotonic_decreasing

    mde_mf = pilot_mde_table(
        clean_daily, hourly, peak_hours=(7, 8, 9), weekdays_only=True
    )
    assert list(mde_mf["pilot_days"]) == [14, 28, 42, 56, 84]
    assert (mde_mf["schedule_scope"] == "Mon-Fri").all()
    assert mde_mf["peak_mde_pct"].is_monotonic_decreasing


def test_regime_and_trend_audit_separates_pos_shift_and_weather() -> None:
    dates = pd.date_range("2026-08-20", periods=30, freq="D")
    joined = pd.DataFrame(
        {
            "date": dates,
            "revenue": np.where(dates >= pd.Timestamp("2026-09-05"), 3200.0, 4000.0),
            "residual_pct": np.where(dates >= pd.Timestamp("2026-09-05"), -0.08, 0.02),
            "has_incident_lag0": dates.day.isin([25, 10]),
            "primary_category_lag0": np.where(
                dates.day == 25,
                "weather_disruption",
                np.where(dates.day == 10, "traffic_accident", None),
            ),
        }
    )
    audit = regime_and_trend_audit(joined, tablet_start_date="2026-09-05")
    reg = audit["regime_summary"]
    ws = audit["weather_vs_street"]
    assert set(reg["regime"]) == {"Pre POS Change", "Post POS Change"}
    assert set(ws["disruption_group"]) == {"Weather", "Street / Traffic / Safety", "No Incident"}
