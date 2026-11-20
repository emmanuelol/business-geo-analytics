"""Tests for src.causal_eval (prospective simulator + synthetic DiD/ITS recovery)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_eval import (
    analyze_attach_and_segmentation,
    build_osint_normalized_baseline,
    build_sensitivity_matrix,
    estimate_did,
    estimate_hourly_bar_throughput,
    estimate_interrupted_time_series,
    profile_hourly_bottlenecks,
    simulate_linebuster_scenarios,
)
from src.config import StoreConfig


def test_bottleneck_profiling_and_osint_normalization() -> None:
    dates = pd.date_range("2026-08-01", periods=20, freq="D")
    inc_dates = {pd.Timestamp("2026-08-05"), pd.Timestamp("2026-08-12")}

    daily_rows = []
    hourly_rows = []
    for d in dates:
        is_inc = d in inc_dates
        mult = 0.65 if is_inc else 1.0
        day_total = 0.0
        for h in range(6, 22):
            base = 600.0 if h in (8, 9, 18, 19, 20) else 200.0
            # Make one non-incident day extra busy to test P90/P95 above mean
            busy = 1.25 if d == pd.Timestamp("2026-08-15") else 1.0
            vol = round(base * mult * busy, 2)
            day_total += vol
            hourly_rows.append(
                {
                    "date": d,
                    "hour_int": h,
                    "hour_label": f"{h:02d}:00",
                    "volume": vol,
                    "is_peak": h in (8, 9, 18, 19, 20),
                }
            )
        daily_rows.append(
            {
                "date": d,
                "revenue": day_total,
                "tickets": int(round(day_total / 150.0)),
                "avg_ticket": 150.0,
                "upt": 2.2,
                "food_sales": day_total * 0.25,
                "drink_sales": day_total * 0.72,
                "shop_sales": day_total * 0.03,
                "is_post_tablet_change": False,
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    hourly_df = pd.DataFrame(hourly_rows)

    profile = profile_hourly_bottlenecks(hourly_df, peak_hours=(8, 9, 18, 19, 20))
    assert len(profile["hourly_table"]) == 16
    assert profile["peak_revenue_share"] > 0.55
    assert profile["peak_p95_daily_ceiling"] > profile["mean_daily_peak_revenue"]

    baseline = build_osint_normalized_baseline(
        daily_df=daily_df,
        hourly_df=hourly_df,
        incident_dates=list(inc_dates),
        peak_hours=(8, 9, 18, 19, 20),
    )
    assert baseline["incident_days_excluded"] == 2
    assert baseline["clean_days"] == 18
    assert baseline["clean_mean_daily_revenue"] > baseline["raw_mean_daily_revenue"]
    # On clean days, peak hours = 5 * 600 = 3000 (plus one 1.25 day -> ~3041.67)
    assert baseline["clean_mean_daily_peak_revenue"] == pytest.approx(3041.67, rel=0.01)


def test_prospective_scenarios_and_sensitivity_math() -> None:
    cfg = StoreConfig(
        hardware_capex=12000.0,
        monthly_saas_fee=1200.0,
        peak_labor_cost_per_hr=60.0,
        gross_margin_pct=0.65,
        peak_hours=(8, 9, 18, 19, 20),
    )
    baseline = {
        "clean_mean_daily_revenue": 4500.0,
        "clean_mean_daily_peak_revenue": 2500.0,
        "clean_mean_daily_offpeak_revenue": 2000.0,
        "clean_mean_avg_ticket": 150.0,
        "clean_mean_tickets_per_day": 30.0,
        "peak_p90_daily_ceiling": 2900.0,
        "peak_p95_daily_ceiling": 3100.0,
        "peak_revenue_share": 2500.0 / 4500.0,
    }

    # Test with 0 incremental labor hrs (reallocated cashier) so math is closed-form
    sim = simulate_linebuster_scenarios(
        baseline=baseline,
        config=cfg,
        labor_hours_per_day=0.0,
        operating_days_per_month=30.0,
    )
    assert list(sim["scenario"]) == ["Conservative", "Base", "Optimistic"]

    # Base scenario: +10% peak lift + 2% attach on peak rev (2500 * 0.12 = 300/day)
    base = sim.loc[sim["scenario"] == "Base"].iloc[0]
    assert base["incremental_peak_rev_day"] == pytest.approx(250.0)
    assert base["incremental_attach_rev_day"] == pytest.approx(50.0)
    assert base["total_incremental_rev_day"] == pytest.approx(300.0)
    # Off-peak untouched -> total daily rev = 4500 + 300 = 4800
    assert base["projected_daily_revenue"] == pytest.approx(4800.0)
    # Monthly gross profit = 300 * 30 * 0.65 = 5850; monthly opex = 1200 -> net = 4650
    assert base["monthly_net_incremental_profit"] == pytest.approx(4650.0)
    assert base["payback_months"] == pytest.approx(12000.0 / 4650.0, rel=1e-3)
    # Breakeven rev/day = (1200/30) / 0.65 = 61.538; tickets/day = 61.538 / 150 = 0.4103
    assert base["breakeven_tickets_per_day"] == pytest.approx((40.0 / 0.65) / 150.0, rel=1e-3)

    matrix = build_sensitivity_matrix(
        baseline=baseline,
        config=cfg,
        labor_hours_per_day=0.0,
    )
    assert {"peak_lift_pct", "gross_margin_pct", "monthly_net_incremental_profit", "payback_months"}.issubset(matrix.columns)
    assert len(matrix) == 24


def test_tablet_segmentation_is_diagnostic_only() -> None:
    dates = pd.date_range("2026-08-25", periods=20, freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "revenue": 4000.0,
            "tickets": 25.0,
            "avg_ticket": 160.0,
            "upt": 2.2,
            "food_sales": 1000.0,
            "drink_sales": 2800.0,
            "shop_sales": 200.0,
            "is_post_tablet_change": dates >= pd.Timestamp("2026-09-05"),
        }
    )
    mix = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-18", "2026-08-19"]),
            "product_name": ["Americano", "Iced Latte + Croissant"],
            "category": ["Beverages", "Combos"],
            "quantity_sold": [10, 2],
        }
    )
    seg = analyze_attach_and_segmentation(
        daily_df=daily,
        product_mix_df=mix,
        tablet_start_date="2026-09-05",
        tablet_interpretation="pos_channel_change",
    )
    assert seg["tablet_interpretation"] == "pos_channel_change"
    assert seg["food_revenue_share"] == pytest.approx(0.25)
    assert seg["combo_unit_share_pre_cutoff"] == pytest.approx(2.0 / 12.0)


def test_did_recovers_known_synthetic_effect() -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2026-07-01", periods=80, freq="D")
    post_start = pd.Timestamp("2026-08-10")
    true_att = 120.0

    rows = []
    for d in dates:
        day_shock = rng.normal(0.0, 12.0)
        dow_effect = 25.0 if d.day_of_week >= 4 else 0.0
        is_post = int(d >= post_start)
        for h in (8, 9, 11, 13, 18, 19):
            is_peak = int(h in (8, 9, 18, 19))
            y = (
                220.0
                + 150.0 * is_peak
                + 15.0 * is_post
                + true_att * is_peak * is_post
                + dow_effect
                + day_shock
                + rng.normal(0.0, 15.0)
            )
            rows.append(
                {
                    "date": d,
                    "hour_int": h,
                    "is_peak": is_peak,
                    "is_post_linebuster": is_post,
                    "is_weekend": int(d.day_of_week >= 4),
                    "volume": y,
                }
            )
    panel = pd.DataFrame(rows)
    res = estimate_did(
        panel,
        outcome_col="volume",
        treat_col="is_peak",
        post_col="is_post_linebuster",
        date_col="date",
        control_cols=("is_weekend",),
    )
    assert res["att"] == pytest.approx(true_att, abs=15.0)
    assert res["p_value"] < 0.001
    assert res["n_clusters"] == 80


def test_its_recovers_known_synthetic_level_and_slope() -> None:
    rng = np.random.default_rng(456)
    dates = pd.date_range("2026-06-01", periods=100, freq="D")
    intervention = pd.Timestamp("2026-07-25")

    t = np.arange(len(dates), dtype=float)
    post = (dates >= intervention).astype(float)
    t_since = np.where(dates >= intervention, (dates - intervention).days.to_numpy(), 0.0)

    true_level = 450.0
    true_slope = 12.0
    y = 3000.0 + 5.0 * t + true_level * post + true_slope * t_since + rng.normal(0.0, 35.0, size=len(dates))

    df = pd.DataFrame({"date": dates, "revenue": y})
    res = estimate_interrupted_time_series(
        df,
        outcome_col="revenue",
        date_col="date",
        intervention_date=intervention,
    )
    assert res["level_change"] == pytest.approx(true_level, abs=60.0)
    assert res["slope_change"] == pytest.approx(true_slope, abs=3.0)
    assert res["level_p_value"] < 0.001
    assert res["slope_p_value"] < 0.001


def test_weekdays_only_mon_fri_7_to_10_and_bar_throughput() -> None:
    dates = pd.date_range("2026-08-03", periods=28, freq="D")  # 4 full Mon-Sun weeks
    daily_rows = []
    hourly_rows = []
    for d in dates:
        is_wknd = d.day_of_week >= 5
        day_total = 0.0
        for h in range(6, 22):
            # High morning traffic on Mon-Fri 7..9; low morning / high afternoon on weekends
            if not is_wknd and h in (7, 8, 9):
                vol = 600.0
            elif is_wknd and h in (18, 19, 20):
                vol = 900.0
            else:
                vol = 150.0
            day_total += vol
            hourly_rows.append({"date": d, "hour_int": h, "volume": vol})
        daily_rows.append(
            {
                "date": d,
                "revenue": day_total,
                "tickets": int(round(day_total / 150.0)),
                "avg_ticket": 150.0,
                "upt": 2.0,
                "drink_sales": day_total * 0.75,
                "food_sales": day_total * 0.25,
                "shop_sales": 0.0,
            }
        )
    daily_df = pd.DataFrame(daily_rows)
    hourly_df = pd.DataFrame(hourly_rows)

    scoped = build_osint_normalized_baseline(
        daily_df=daily_df,
        hourly_df=hourly_df,
        peak_hours=(7, 8, 9),
        weekdays_only=True,
    )
    assert scoped["weekdays_only"] is True
    assert scoped["target_peak_hours"] == (7, 8, 9)
    assert scoped["clean_days"] == 20  # 4 * 5 Mon-Fri weekdays
    assert scoped["clean_mean_daily_peak_revenue"] == pytest.approx(1800.0)

    cfg = StoreConfig(
        hardware_capex=12000.0,
        monthly_saas_fee=1200.0,
        peak_labor_cost_per_hr=80.0,
        gross_margin_pct=0.65,
    )
    # Auto-uses 3 peak hrs (7,8,9) * 80 = 240/day and 21.7 weekdays/month (~65.1 hrs/mo)
    sim = simulate_linebuster_scenarios(baseline=scoped, config=cfg)
    base = sim.loc[sim["scenario"] == "Base"].iloc[0]
    expected_monthly_opex = 1200.0 + (3.0 * 80.0 * 21.7)
    assert base["daily_incremental_opex"] * 21.7 == pytest.approx(expected_monthly_opex, rel=1e-4)

    throughput = estimate_hourly_bar_throughput(
        hourly_df=hourly_df,
        daily_df=daily_df,
        seg={"drink_revenue_share": 0.75, "product_mix_unit_shares": {"Beverages": 0.70, "Combos": 0.10}},
        peak_hours=(7, 8, 9),
        weekdays_only=True,
    )
    hr8 = throughput.loc[throughput["hour_int"] == 8].iloc[0]
    assert hr8["implied_mean_tickets_hr"] == pytest.approx(600.0 / 150.0)
    # 4 tickets/hr * 2.0 upt * (0.70 + 0.10 drinks-in-combos) = 6.4 drinks/hr
    assert hr8["implied_mean_drinks_hr"] == pytest.approx(4.0 * 2.0 * 0.80)
