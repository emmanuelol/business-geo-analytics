"""Prospective linebuster scenario simulator and post-launch causal inference helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    ITEMS_COVERAGE_END_DATE,
    OPERATING_HOUR_MAX,
    OPERATING_HOUR_MIN,
    PEAK_HOURS_DEFAULT,
    TABLET_CHANNEL_INTERPRETATION,
    TABLET_CHANNEL_START_DATE,
    StoreConfig,
    load_config,
)
from src.spatial_models import enrich_incidents_spatial


def _complete_hourly_panel(
    hourly_df: pd.DataFrame,
    operating_hours: tuple[int, int] = (OPERATING_HOUR_MIN, OPERATING_HOUR_MAX),
) -> pd.DataFrame:
    df = hourly_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour_int"] = pd.to_numeric(df["hour_int"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "hour_int"]).copy()
    df["hour_int"] = df["hour_int"].astype(int)

    min_h, max_h = operating_hours
    df = df[(df["hour_int"] >= min_h) & (df["hour_int"] <= max_h)]
    if df.empty:
        return pd.DataFrame(columns=["date", "hour_int", "volume"])

    dates = sorted(df["date"].unique())
    hours = list(range(min_h, max_h + 1))
    idx = pd.MultiIndex.from_product([dates, hours], names=["date", "hour_int"])
    panel = (
        df.groupby(["date", "hour_int"], as_index=True)["volume"]
        .sum()
        .reindex(idx, fill_value=0.0)
        .reset_index()
    )
    return panel


def profile_hourly_bottlenecks(
    hourly_df: pd.DataFrame,
    peak_hours: Sequence[int] = PEAK_HOURS_DEFAULT,
    operating_hours: tuple[int, int] = (OPERATING_HOUR_MIN, OPERATING_HOUR_MAX),
) -> dict[str, Any]:
    """Profile hour-of-day revenue concentration and P90/P95 capacity ceilings."""
    panel = _complete_hourly_panel(hourly_df, operating_hours=operating_hours)
    peak_set = tuple(sorted(int(h) for h in peak_hours))

    if panel.empty:
        empty_tbl = pd.DataFrame(
            columns=[
                "hour_int",
                "hour_label",
                "is_peak",
                "n_days",
                "mean_volume",
                "median_volume",
                "p90_volume",
                "p95_volume",
                "max_volume",
                "share_of_daily_revenue",
                "headroom_to_p95_pct",
            ]
        )
        return {
            "hourly_table": empty_tbl,
            "peak_hours": peak_set,
            "n_days": 0,
            "peak_revenue_share": 0.0,
            "mean_daily_total_revenue": 0.0,
            "mean_daily_peak_revenue": 0.0,
            "mean_daily_offpeak_revenue": 0.0,
            "peak_p90_daily_ceiling": 0.0,
            "peak_p95_daily_ceiling": 0.0,
        }

    panel["is_peak"] = panel["hour_int"].isin(peak_set)
    grouped = panel.groupby("hour_int", as_index=False)["volume"].agg(
        n_days="count",
        mean_volume="mean",
        median_volume="median",
        p90_volume=lambda s: float(np.quantile(s, 0.90)),
        p95_volume=lambda s: float(np.quantile(s, 0.95)),
        max_volume="max",
    )
    total_mean_daily = float(grouped["mean_volume"].sum())
    grouped["hour_label"] = grouped["hour_int"].map(lambda h: f"{int(h):02d}:00")
    grouped["is_peak"] = grouped["hour_int"].isin(peak_set)
    grouped["share_of_daily_revenue"] = (
        grouped["mean_volume"] / total_mean_daily if total_mean_daily > 0 else 0.0
    )
    grouped["headroom_to_p95_pct"] = np.where(
        grouped["mean_volume"] > 0,
        (grouped["p95_volume"] - grouped["mean_volume"]) / grouped["mean_volume"],
        0.0,
    )

    cols = [
        "hour_int",
        "hour_label",
        "is_peak",
        "n_days",
        "mean_volume",
        "median_volume",
        "p90_volume",
        "p95_volume",
        "max_volume",
        "share_of_daily_revenue",
        "headroom_to_p95_pct",
    ]
    hourly_table = grouped[cols].sort_values("hour_int").reset_index(drop=True)

    daily_totals = panel.groupby("date")["volume"].sum()
    daily_peak = panel[panel["is_peak"]].groupby("date")["volume"].sum().reindex(
        daily_totals.index, fill_value=0.0
    )
    daily_offpeak = daily_totals - daily_peak

    mean_daily_peak = float(daily_peak.mean())
    mean_daily_total = float(daily_totals.mean())
    return {
        "hourly_table": hourly_table,
        "peak_hours": peak_set,
        "n_days": int(len(daily_totals)),
        "peak_revenue_share": (mean_daily_peak / mean_daily_total) if mean_daily_total > 0 else 0.0,
        "mean_daily_total_revenue": mean_daily_total,
        "mean_daily_peak_revenue": mean_daily_peak,
        "mean_daily_offpeak_revenue": float(daily_offpeak.mean()),
        "peak_p90_daily_ceiling": float(np.quantile(daily_peak, 0.90)),
        "peak_p95_daily_ceiling": float(np.quantile(daily_peak, 0.95)),
    }


def build_osint_normalized_baseline(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    incident_dates: Iterable[Any] | None = None,
    incidents_df: pd.DataFrame | None = None,
    store_lat: float | None = None,
    store_lon: float | None = None,
    max_distance_m: float = 1500.0,
    peak_hours: Sequence[int] = PEAK_HOURS_DEFAULT,
    operating_hours: tuple[int, int] = (OPERATING_HOUR_MIN, OPERATING_HOUR_MAX),
) -> dict[str, Any]:
    """Build clean demand baseline excluding OSINT external-incident shock days."""
    daily = daily_df.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).copy()

    hourly = hourly_df.copy()
    hourly["date"] = pd.to_datetime(hourly["date"], errors="coerce")
    hourly = hourly.dropna(subset=["date"]).copy()

    exclude_set: set[pd.Timestamp] = set()
    if incident_dates is not None:
        exclude_set.update(pd.to_datetime(list(incident_dates)).normalize())

    if incidents_df is not None and not incidents_df.empty:
        if store_lat is not None and store_lon is not None:
            enriched = enrich_incidents_spatial(
                incidents_df,
                store_lat=store_lat,
                store_lon=store_lon,
                max_distance_m=max_distance_m,
            )
            nearby = enriched[enriched["distance_m"] <= float(max_distance_m)]
        else:
            nearby = incidents_df.copy()
            nearby["date"] = pd.to_datetime(nearby["date"], errors="coerce")
        exclude_set.update(nearby["date"].dropna().dt.normalize().tolist())

    if "has_incident_lag0" in daily.columns:
        flagged = daily.loc[daily["has_incident_lag0"].astype(bool), "date"].dt.normalize()
        exclude_set.update(flagged.tolist())

    daily_norm_dates = daily["date"].dt.normalize()
    clean_daily = daily[~daily_norm_dates.isin(exclude_set)].copy()
    if clean_daily.empty:
        clean_daily = daily.copy()

    clean_hourly = hourly[~hourly["date"].dt.normalize().isin(exclude_set)].copy()
    if clean_hourly.empty:
        clean_hourly = hourly.copy()

    bottleneck = profile_hourly_bottlenecks(
        clean_hourly, peak_hours=peak_hours, operating_hours=operating_hours
    )

    raw_mean_rev = float(daily["revenue"].mean()) if not daily.empty else 0.0
    clean_mean_rev = float(clean_daily["revenue"].mean()) if not clean_daily.empty else 0.0
    clean_tickets = (
        float(clean_daily["tickets"].mean())
        if "tickets" in clean_daily.columns and clean_daily["tickets"].notna().any()
        else 0.0
    )
    clean_avg_ticket = (
        float(clean_daily["avg_ticket"].mean())
        if "avg_ticket" in clean_daily.columns and clean_daily["avg_ticket"].notna().any()
        else (clean_mean_rev / clean_tickets if clean_tickets > 0 else 0.0)
    )
    clean_upt = (
        float(clean_daily["upt"].mean())
        if "upt" in clean_daily.columns and clean_daily["upt"].notna().any()
        else 0.0
    )

    # Anchor daily peak revenue to clean_mean_daily_revenue * peak_share when hourly sample covers fewer days
    peak_share = float(bottleneck["peak_revenue_share"])
    clean_peak_rev = (
        float(bottleneck["mean_daily_peak_revenue"])
        if bottleneck["n_days"] > 0
        else clean_mean_rev * peak_share
    )
    clean_offpeak_rev = (
        float(bottleneck["mean_daily_offpeak_revenue"])
        if bottleneck["n_days"] > 0
        else clean_mean_rev * (1.0 - peak_share)
    )

    excluded_count = int(daily_norm_dates.isin(exclude_set).sum())
    uplift_adj = (
        (clean_mean_rev - raw_mean_rev) / raw_mean_rev if raw_mean_rev > 0 else 0.0
    )

    return {
        "total_days": int(len(daily)),
        "incident_days_excluded": excluded_count,
        "clean_days": int(len(clean_daily)),
        "raw_mean_daily_revenue": raw_mean_rev,
        "clean_mean_daily_revenue": clean_mean_rev,
        "osint_uplift_adjustment_pct": uplift_adj,
        "clean_mean_tickets_per_day": clean_tickets,
        "clean_mean_avg_ticket": clean_avg_ticket,
        "clean_mean_upt": clean_upt,
        "peak_revenue_share": peak_share,
        "clean_mean_daily_peak_revenue": clean_peak_rev,
        "clean_mean_daily_offpeak_revenue": clean_offpeak_rev,
        "peak_p90_daily_ceiling": float(bottleneck["peak_p90_daily_ceiling"]),
        "peak_p95_daily_ceiling": float(bottleneck["peak_p95_daily_ceiling"]),
        "clean_bottleneck": bottleneck,
        "clean_daily_df": clean_daily,
        "clean_hourly_df": clean_hourly,
    }


def analyze_attach_and_segmentation(
    daily_df: pd.DataFrame,
    product_mix_df: pd.DataFrame | None = None,
    contextual_split_df: pd.DataFrame | None = None,
    sales_channels_df: pd.DataFrame | None = None,
    tablet_start_date: str = TABLET_CHANNEL_START_DATE,
    tablet_interpretation: str = TABLET_CHANNEL_INTERPRETATION,
    items_end_date: str = ITEMS_COVERAGE_END_DATE,
) -> dict[str, Any]:
    """Compute food/combo attach baselines and audit Tablet channel segmentation."""
    daily = daily_df.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).copy()

    food_share = np.nan
    drink_share = np.nan
    shop_share = np.nan
    if {"food_sales", "drink_sales"}.issubset(daily.columns):
        valid_cat = daily.dropna(subset=["food_sales", "drink_sales"])
        if not valid_cat.empty:
            shop_vals = (
                valid_cat["shop_sales"].fillna(0.0)
                if "shop_sales" in valid_cat.columns
                else 0.0
            )
            cat_total = (valid_cat["food_sales"] + valid_cat["drink_sales"] + shop_vals).sum()
            if cat_total > 0:
                food_share = float(valid_cat["food_sales"].sum() / cat_total)
                drink_share = float(valid_cat["drink_sales"].sum() / cat_total)
                shop_share = float(np.sum(shop_vals) / cat_total)

    mix_shares: dict[str, float] = {}
    combo_unit_share = np.nan
    food_plus_combo_share = np.nan
    if product_mix_df is not None and not product_mix_df.empty:
        mix = product_mix_df.copy()
        mix["date"] = pd.to_datetime(mix["date"], errors="coerce")
        mix = mix[mix["date"] <= pd.Timestamp(items_end_date)]
        total_qty = float(mix["quantity_sold"].sum())
        if total_qty > 0:
            by_cat = mix.groupby("category")["quantity_sold"].sum() / total_qty
            mix_shares = {str(k): float(v) for k, v in by_cat.items()}
            combo_unit_share = float(mix_shares.get("Combos", 0.0))
            food_plus_combo_share = float(
                mix_shares.get("Food_Pastries", 0.0) + mix_shares.get("Combos", 0.0)
            )

    ctx_shares: dict[str, float] = {}
    if contextual_split_df is not None and not contextual_split_df.empty:
        ctx = contextual_split_df.copy()
        ctx_total = float(ctx["revenue"].sum())
        if ctx_total > 0:
            ctx_by_cat = ctx.groupby("category")["revenue"].sum() / ctx_total
            ctx_shares = {str(k): float(v) for k, v in ctx_by_cat.items()}

    cutoff = pd.Timestamp(tablet_start_date)
    pre = daily[daily["date"] < cutoff]
    post = daily[daily["date"] >= cutoff]

    channel_shares: dict[str, float] = {}
    if sales_channels_df is not None and not sales_channels_df.empty:
        ch_total = float(sales_channels_df["revenue"].sum())
        if ch_total > 0:
            ch_by = sales_channels_df.groupby("channel")["revenue"].sum() / ch_total
            channel_shares = {str(k): float(v) for k, v in ch_by.items()}

    return {
        "tablet_start_date": tablet_start_date,
        "tablet_interpretation": tablet_interpretation,
        "items_coverage_end_date": items_end_date,
        "food_revenue_share": food_share,
        "drink_revenue_share": drink_share,
        "shop_revenue_share": shop_share,
        "product_mix_unit_shares": mix_shares,
        "combo_unit_share_pre_cutoff": combo_unit_share,
        "food_plus_combo_unit_share_pre_cutoff": food_plus_combo_share,
        "contextual_split_shares": ctx_shares,
        "sales_channel_shares": channel_shares,
        "pre_tablet_days": int(len(pre)),
        "post_tablet_days": int(len(post)),
        "pre_tablet_mean_revenue": float(pre["revenue"].mean()) if not pre.empty else np.nan,
        "post_tablet_mean_revenue": float(post["revenue"].mean()) if not post.empty else np.nan,
        "pre_tablet_mean_avg_ticket": float(pre["avg_ticket"].mean()) if not pre.empty else np.nan,
        "post_tablet_mean_avg_ticket": float(post["avg_ticket"].mean()) if not post.empty else np.nan,
        "pre_tablet_mean_upt": float(pre["upt"].mean()) if not pre.empty else np.nan,
        "post_tablet_mean_upt": float(post["upt"].mean()) if not post.empty else np.nan,
    }


def _evaluate_scenario_row(
    scenario_name: str,
    peak_lift_pct: float,
    attach_lift_pct: float,
    baseline: dict[str, Any],
    cfg: StoreConfig,
    labor_hours_per_day: float,
    operating_days_per_month: float,
) -> dict[str, Any]:
    daily_peak_rev = float(baseline["clean_mean_daily_peak_revenue"])
    daily_total_rev = float(baseline["clean_mean_daily_revenue"])
    avg_ticket = max(float(baseline["clean_mean_avg_ticket"]), 1e-6)
    p95_peak_ceiling = float(baseline.get("peak_p95_daily_ceiling", 0.0))

    inc_peak_rev_day = daily_peak_rev * float(peak_lift_pct)
    inc_attach_rev_day = daily_peak_rev * float(attach_lift_pct)
    total_inc_rev_day = inc_peak_rev_day + inc_attach_rev_day
    projected_peak_rev_day = daily_peak_rev + total_inc_rev_day
    projected_daily_rev = daily_total_rev + total_inc_rev_day

    daily_saas = cfg.monthly_saas_fee / operating_days_per_month
    daily_labor = cfg.peak_labor_cost_per_hr * labor_hours_per_day
    daily_opex = daily_saas + daily_labor

    inc_gross_profit_day = total_inc_rev_day * cfg.gross_margin_pct
    net_profit_day = inc_gross_profit_day - daily_opex

    monthly_inc_rev = total_inc_rev_day * operating_days_per_month
    monthly_gross_profit = inc_gross_profit_day * operating_days_per_month
    monthly_opex = daily_opex * operating_days_per_month
    monthly_net_profit = monthly_gross_profit - monthly_opex

    breakeven_rev_day = daily_opex / cfg.gross_margin_pct
    breakeven_tickets_day = breakeven_rev_day / avg_ticket
    payback_months = (
        cfg.hardware_capex / monthly_net_profit
        if monthly_net_profit > 0
        else np.inf
    )
    annual_net_profit = 12.0 * monthly_net_profit - cfg.hardware_capex
    roi_12m_pct = (
        (annual_net_profit / cfg.hardware_capex) * 100.0
        if cfg.hardware_capex > 0
        else np.inf
    )

    return {
        "scenario": scenario_name,
        "peak_lift_pct": float(peak_lift_pct),
        "attach_lift_pct": float(attach_lift_pct),
        "gross_margin_pct": float(cfg.gross_margin_pct),
        "incremental_peak_rev_day": round(inc_peak_rev_day, 2),
        "incremental_attach_rev_day": round(inc_attach_rev_day, 2),
        "total_incremental_rev_day": round(total_inc_rev_day, 2),
        "projected_peak_rev_day": round(projected_peak_rev_day, 2),
        "projected_daily_revenue": round(projected_daily_rev, 2),
        "pct_of_p95_peak_ceiling": (
            round(projected_peak_rev_day / p95_peak_ceiling, 4)
            if p95_peak_ceiling > 0
            else np.nan
        ),
        "daily_incremental_gross_profit": round(inc_gross_profit_day, 2),
        "daily_incremental_opex": round(daily_opex, 2),
        "daily_net_incremental_profit": round(net_profit_day, 2),
        "monthly_incremental_revenue": round(monthly_inc_rev, 2),
        "monthly_incremental_gross_profit": round(monthly_gross_profit, 2),
        "monthly_incremental_opex": round(monthly_opex, 2),
        "monthly_net_incremental_profit": round(monthly_net_profit, 2),
        "breakeven_revenue_per_day": round(breakeven_rev_day, 2),
        "breakeven_tickets_per_day": round(breakeven_tickets_day, 2),
        "payback_months": round(payback_months, 2) if np.isfinite(payback_months) else np.inf,
        "roi_12m_pct": round(roi_12m_pct, 2) if np.isfinite(roi_12m_pct) else np.inf,
    }


def simulate_linebuster_scenarios(
    baseline: dict[str, Any],
    config: StoreConfig | None = None,
    scenarios: Sequence[tuple[str, float, float]] = (
        ("Conservative", 0.05, 0.00),
        ("Base", 0.10, 0.02),
        ("Optimistic", 0.15, 0.04),
    ),
    labor_hours_per_day: float | None = None,
    operating_days_per_month: float = 30.0,
) -> pd.DataFrame:
    """Simulate 3-scenario ex-ante peak-hour revenue lift, break-even, and ROI payback."""
    cfg = config or load_config()
    eff_labor_hrs = (
        float(len(cfg.peak_hours))
        if labor_hours_per_day is None
        else float(labor_hours_per_day)
    )
    rows = [
        _evaluate_scenario_row(
            scenario_name=name,
            peak_lift_pct=p_lift,
            attach_lift_pct=a_lift,
            baseline=baseline,
            cfg=cfg,
            labor_hours_per_day=eff_labor_hrs,
            operating_days_per_month=operating_days_per_month,
        )
        for name, p_lift, a_lift in scenarios
    ]
    return pd.DataFrame(rows)


def build_sensitivity_matrix(
    baseline: dict[str, Any],
    config: StoreConfig | None = None,
    lift_grid: Sequence[float] = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20),
    margin_grid: Sequence[float] = (0.55, 0.60, 0.65, 0.70),
    attach_lift_pct: float = 0.0,
    labor_hours_per_day: float | None = None,
    operating_days_per_month: float = 30.0,
) -> pd.DataFrame:
    """Evaluate monthly profit, payback, and break-even across lift x margin combinations."""
    cfg = config or load_config()
    eff_labor_hrs = (
        float(len(cfg.peak_hours))
        if labor_hours_per_day is None
        else float(labor_hours_per_day)
    )
    rows: list[dict[str, Any]] = []
    for margin in margin_grid:
        tmp_cfg = StoreConfig(
            **{**cfg.to_dict(include_paths=False), "gross_margin_pct": float(margin)},
            repo_root=cfg.repo_root,
        )
        for lift in lift_grid:
            rows.append(
                _evaluate_scenario_row(
                    scenario_name=f"lift_{int(round(lift * 100))}pct_margin_{int(round(margin * 100))}pct",
                    peak_lift_pct=float(lift),
                    attach_lift_pct=float(attach_lift_pct),
                    baseline=baseline,
                    cfg=tmp_cfg,
                    labor_hours_per_day=eff_labor_hrs,
                    operating_days_per_month=operating_days_per_month,
                )
            )
    return pd.DataFrame(rows)


def estimate_did(
    df: pd.DataFrame,
    outcome_col: str = "volume",
    treat_col: str = "is_peak",
    post_col: str = "is_post_linebuster",
    date_col: str = "date",
    control_cols: Sequence[str] = (),
) -> dict[str, Any]:
    """Post-launch Difference-in-Differences OLS with date-clustered standard errors."""
    req = [outcome_col, treat_col, post_col, date_col, *control_cols]
    work = df[req].dropna().copy()
    work["_treat"] = work[treat_col].astype(float)
    work["_post"] = work[post_col].astype(float)
    work["_did"] = work["_treat"] * work["_post"]

    x_cols = ["_treat", "_post", "_did"] + list(control_cols)
    X = sm.add_constant(work[x_cols].astype(float), has_constant="add")
    y = work[outcome_col].astype(float)

    groups = pd.factorize(pd.to_datetime(work[date_col]))[0]
    n_clusters = int(len(np.unique(groups)))
    model = sm.OLS(y, X)
    if n_clusters >= 2:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        fit = model.fit()

    ci = fit.conf_int()
    return {
        "att": float(fit.params["_did"]),
        "std_err": float(fit.bse["_did"]),
        "t_stat": float(fit.tvalues["_did"]),
        "p_value": float(fit.pvalues["_did"]),
        "ci_low": float(ci.loc["_did", 0]),
        "ci_high": float(ci.loc["_did", 1]),
        "n_obs": int(fit.nobs),
        "n_clusters": n_clusters,
        "r_squared": float(fit.rsquared),
    }


def estimate_interrupted_time_series(
    df: pd.DataFrame,
    outcome_col: str = "revenue",
    date_col: str = "date",
    intervention_date: str | pd.Timestamp = TABLET_CHANNEL_START_DATE,
    control_cols: Sequence[str] = (),
    hac_maxlags: int = 3,
) -> dict[str, Any]:
    """Post-launch Interrupted Time Series segmented regression with HAC standard errors."""
    work = df[[date_col, outcome_col, *control_cols]].dropna().copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values(date_col).reset_index(drop=True)

    cutoff = pd.Timestamp(intervention_date)
    work["_t"] = (work[date_col] - work[date_col].min()).dt.total_seconds() / 86400.0
    work["_post"] = (work[date_col] >= cutoff).astype(float)
    work["_t_since"] = np.where(
        work[date_col] >= cutoff,
        (work[date_col] - cutoff).dt.total_seconds() / 86400.0,
        0.0,
    )

    x_cols = ["_t", "_post", "_t_since"] + list(control_cols)
    X = sm.add_constant(work[x_cols].astype(float), has_constant="add")
    y = work[outcome_col].astype(float)

    model = sm.OLS(y, X)
    if len(work) > hac_maxlags + len(x_cols) + 2:
        fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
    else:
        fit = model.fit()

    return {
        "intervention_date": str(cutoff.date()),
        "level_change": float(fit.params["_post"]),
        "level_std_err": float(fit.bse["_post"]),
        "level_p_value": float(fit.pvalues["_post"]),
        "slope_change": float(fit.params["_t_since"]),
        "slope_std_err": float(fit.bse["_t_since"]),
        "slope_p_value": float(fit.pvalues["_t_since"]),
        "n_obs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
    }
