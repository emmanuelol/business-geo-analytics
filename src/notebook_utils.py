"""Shared rendering, mode-isolation, and diagnostic helpers for notebooks 01-05."""

from __future__ import annotations

import html
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.causal_eval import simulate_linebuster_scenarios
from src.config import StoreConfig
from src.osint_collector import load_incidents_file

_QUINCENA_DAYS: frozenset[int] = frozenset({1, 14, 15, 16, 28, 29, 30, 31})


def configure_mpl_env() -> Any:
    """Configure PID-isolated MPLCONFIGDIR and conditional Agg backend for notebooks."""
    mpl_dir = Path(f"/tmp/bga-matplotlib-{os.getpid()}")
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def show_fig(fig: Any) -> None:
    """Apply tight layout, display figure inline inside IPython kernels, and close it."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


def render_table(
    df: pd.DataFrame,
    title: str | None = None,
    fmt: Mapping[str, str] | None = None,
    max_rows: int = 30,
) -> None:
    """Display a formatted HTML DataFrame in Jupyter with plain-text fallback."""
    view = df.head(max_rows).copy()
    if fmt:
        applicable = {col: spec for col, spec in fmt.items() if col in view.columns}
        table_html = view.style.format(applicable, na_rep="N/A").hide(axis="index").to_html()
    else:
        table_html = view.to_html(index=False, border=0)

    title_html = f"<h4>{html.escape(title)}</h4>" if title else ""
    try:
        from IPython.display import HTML, display

        display(HTML(f"{title_html}{table_html}"))
    except ImportError:
        if title:
            print(title)
        print(view)


def processed_incidents_path(cfg: StoreConfig, mode: str) -> Path:
    """Return canonical processed OSINT incident CSV path for sample or raw mode."""
    norm_mode = mode.strip().lower()
    if norm_mode not in {"sample", "raw"}:
        raise ValueError(f"Unsupported mode {mode!r}; expected 'sample' or 'raw'")
    return cfg.processed_dir / f"incidents_{norm_mode}.csv"


def resolve_incidents_path(cfg: StoreConfig, mode: str) -> Path:
    """Resolve mode-isolated incident CSV path without leaking sample data into raw."""
    norm_mode = mode.strip().lower()
    proc_path = processed_incidents_path(cfg, norm_mode)
    if norm_mode == "raw":
        if not proc_path.is_file():
            raise FileNotFoundError(
                f"Missing {proc_path}. Run notebooks/02_silao_osint_pipeline.ipynb "
                "with BGA_DATA_MODE=raw first."
            )
        return proc_path
    if proc_path.is_file():
        return proc_path
    return cfg.sample_dir / "incidents.csv"


def load_incidents_for_mode(
    cfg: StoreConfig, mode: str, relevant_only: bool = True
) -> tuple[pd.DataFrame, Path]:
    """Load incidents for the active DATA_MODE and return (DataFrame, resolved_path)."""
    path = resolve_incidents_path(cfg, mode)
    df = load_incidents_file(path, relevant_only=relevant_only)
    return df, path


def date_overlap_audit(
    sales_df: pd.DataFrame, incidents_df: pd.DataFrame
) -> pd.DataFrame:
    """Audit calendar window overlap between sales dates and OSINT incident dates."""
    s_dates = pd.to_datetime(sales_df["date"], errors="coerce").dropna().dt.normalize()
    i_dates = (
        pd.to_datetime(incidents_df["date"], errors="coerce").dropna().dt.normalize()
        if not incidents_df.empty and "date" in incidents_df.columns
        else pd.Series([], dtype="datetime64[ns]")
    )

    if s_dates.empty or i_dates.empty:
        return pd.DataFrame(
            [
                {
                    "sales_min": str(s_dates.min().date()) if not s_dates.empty else "N/A",
                    "sales_max": str(s_dates.max().date()) if not s_dates.empty else "N/A",
                    "inc_min": str(i_dates.min().date()) if not i_dates.empty else "N/A",
                    "inc_max": str(i_dates.max().date()) if not i_dates.empty else "N/A",
                    "overlap_days": 0,
                    "incident_days_in_window": 0,
                    "status": "WARNING: zero_overlap",
                }
            ]
        )

    s_min, s_max = s_dates.min(), s_dates.max()
    i_min, i_max = i_dates.min(), i_dates.max()
    ov_start = max(s_min, i_min)
    ov_end = min(s_max, i_max)
    overlap_days = int((ov_end - ov_start).days + 1) if ov_end >= ov_start else 0
    in_window = int(i_dates[(i_dates >= s_min) & (i_dates <= s_max)].nunique())
    status = "OK" if in_window > 0 else "WARNING: zero_overlap"

    return pd.DataFrame(
        [
            {
                "sales_min": str(s_min.date()),
                "sales_max": str(s_max.date()),
                "inc_min": str(i_min.date()),
                "inc_max": str(i_max.date()),
                "overlap_days": overlap_days,
                "incident_days_in_window": in_window,
                "status": status,
            }
        ]
    )


def compute_weighted_dow_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute day-of-week totals and volume-weighted average ticket."""
    work = daily_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).copy()
    work["dow_num"] = work["date"].dt.day_of_week
    work["dow"] = work["date"].dt.day_name()
    work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce").fillna(0.0)
    work["tickets"] = pd.to_numeric(work["tickets"], errors="coerce").fillna(0.0)

    grouped = work.groupby("dow_num", as_index=False).agg(
        dow=("dow", "first"),
        days=("date", "count"),
        mean_revenue=("revenue", "mean"),
        total_revenue=("revenue", "sum"),
        mean_tickets=("tickets", "mean"),
        total_tickets=("tickets", "sum"),
    )
    grouped["weighted_avg_ticket"] = np.where(
        grouped["total_tickets"] > 0,
        grouped["total_revenue"] / grouped["total_tickets"],
        np.nan,
    )
    return grouped.sort_values("dow_num").reset_index(drop=True)


def decompose_daily_baseline(
    daily_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform additive trend + DOW + quincena decomposition and pay-cycle comparison."""
    df = daily_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["tickets"] = (
        pd.to_numeric(df["tickets"], errors="coerce").fillna(0.0)
        if "tickets" in df.columns
        else 0.0
    )

    df["rolling_14d"] = df["revenue"].rolling(14, min_periods=3).mean()
    trend = df["revenue"].rolling(14, center=True, min_periods=3).mean()
    df["trend"] = trend.bfill().ffill().fillna(df["revenue"].mean())
    df["dow_num"] = df["date"].dt.day_of_week

    detrended = df["revenue"] - df["trend"]
    dow_map = detrended.groupby(df["dow_num"]).mean().to_dict()
    df["dow_effect"] = df["dow_num"].map(dow_map).fillna(0.0)

    df["is_quincena"] = df["date"].dt.day.isin(_QUINCENA_DAYS)
    post_dow = detrended - df["dow_effect"]
    q_map = post_dow.groupby(df["is_quincena"]).mean().to_dict()
    df["quincena_effect"] = df["is_quincena"].map(q_map).fillna(0.0)

    df["expected_decomp"] = np.maximum(
        df["trend"] + df["dow_effect"] + df["quincena_effect"], 1e-6
    )
    df["residual_value"] = df["revenue"] - df["expected_decomp"]
    df["residual_pct"] = df["residual_value"] / df["expected_decomp"]

    df["pay_cycle"] = np.where(df["is_quincena"], "Quincena", "Non-Quincena")
    pay_tbl = df.groupby("pay_cycle", as_index=False).agg(
        days=("date", "count"),
        mean_revenue=("revenue", "mean"),
        total_revenue=("revenue", "sum"),
        mean_tickets=("tickets", "mean"),
        total_tickets=("tickets", "sum"),
    )
    pay_tbl["weighted_avg_ticket"] = np.where(
        pay_tbl["total_tickets"] > 0,
        pay_tbl["total_revenue"] / pay_tbl["total_tickets"],
        np.nan,
    )
    non_q = pay_tbl.loc[pay_tbl["pay_cycle"] == "Non-Quincena", "mean_revenue"]
    ref = float(non_q.iloc[0]) if not non_q.empty and non_q.iloc[0] > 0 else np.nan
    pay_tbl["pct_diff_vs_non_quincena"] = (
        (pay_tbl["mean_revenue"] - ref) / ref if np.isfinite(ref) else np.nan
    )
    return df, pay_tbl


def split_weekday_weekend_hourly(
    hourly_df: pd.DataFrame, peak_hours: Sequence[int]
) -> pd.DataFrame:
    """Aggregate hourly volume (and tickets if present) by Weekday vs Weekend."""
    work = hourly_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["hour_int"] = pd.to_numeric(work["hour_int"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["date", "hour_int"]).copy()
    work["hour_int"] = work["hour_int"].astype(int)
    work["day_type"] = np.where(work["date"].dt.day_of_week < 5, "Weekday", "Weekend")

    agg_spec: dict[str, Any] = {
        "n_days": ("date", "nunique"),
        "mean_volume": ("volume", "mean"),
        "p90_volume": ("volume", lambda s: float(np.quantile(s, 0.90))),
        "p95_volume": ("volume", lambda s: float(np.quantile(s, 0.95))),
    }
    if "tickets" in work.columns:
        work["tickets"] = pd.to_numeric(work["tickets"], errors="coerce")
        agg_spec["mean_tickets"] = ("tickets", "mean")

    out = work.groupby(["day_type", "hour_int"], as_index=False).agg(**agg_spec)
    out["hour_label"] = out["hour_int"].map(lambda h: f"{int(h):02d}:00")
    out["is_peak"] = out["hour_int"].isin(set(int(h) for h in peak_hours))
    return out.sort_values(["day_type", "hour_int"]).reset_index(drop=True)


def placebo_lead_test(
    joined_df: pd.DataFrame, outcome_col: str = "residual_pct"
) -> pd.DataFrame:
    """Estimate t-1 placebo lead effect on sales residuals prior to incident day."""
    work = joined_df.sort_values("date").copy()
    work["_lead_inc"] = work["has_incident_lag0"].shift(-1).astype(float)
    work = work[[outcome_col, "_lead_inc"]].dropna()
    n_obs = len(work)
    if n_obs <= 3 or work["_lead_inc"].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "lag": -1,
                    "label": "t-1 (Placebo Lead)",
                    "coef": np.nan,
                    "std_err": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_obs": n_obs,
                    "flag_pretrend": False,
                }
            ]
        )

    y = work[outcome_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n_obs, dtype=float), work["_lead_inc"].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df_resid = max(n_obs - 2, 1)
    sigma2 = float(np.sum(resid**2)) / df_resid
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    coef = float(beta[1])
    t_val = coef / se if se > 0 else np.nan
    p_val = float(2.0 * stats.t.sf(abs(t_val), df=df_resid)) if np.isfinite(t_val) else np.nan
    t_crit = float(stats.t.ppf(0.975, df=df_resid))
    ci_low = coef - t_crit * se
    ci_high = coef + t_crit * se

    return pd.DataFrame(
        [
            {
                "lag": -1,
                "label": "t-1 (Placebo Lead)",
                "coef": round(coef, 6),
                "std_err": round(se, 6),
                "t_stat": round(t_val, 4) if np.isfinite(t_val) else np.nan,
                "p_value": round(p_val, 6) if np.isfinite(p_val) else np.nan,
                "ci_low": round(ci_low, 6),
                "ci_high": round(ci_high, 6),
                "n_obs": n_obs,
                "flag_pretrend": bool(ci_low > 0 or ci_high < 0),
            }
        ]
    )


def stratify_residuals_by_category(
    joined_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
    outcome_col: str = "residual_pct",
) -> pd.DataFrame:
    """Summarize daily sales residuals stratified by OSINT incident category vs clean days."""
    base = joined_df[["date", outcome_col]].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base = base.dropna(subset=["date", outcome_col])

    if incidents_df.empty or "category" not in incidents_df.columns:
        return pd.DataFrame(
            [
                {
                    "category": "no_incident",
                    "n_days": int(len(base)),
                    "mean_residual_pct": float(base[outcome_col].mean()) if len(base) else np.nan,
                    "std_residual_pct": float(base[outcome_col].std(ddof=1)) if len(base) > 1 else np.nan,
                }
            ]
        )

    inc = incidents_df[["date", "category"]].copy()
    inc["date"] = pd.to_datetime(inc["date"], errors="coerce").dt.normalize()
    inc = inc.dropna(subset=["date", "category"]).drop_duplicates()

    merged = base.merge(inc, on="date", how="inner")
    inc_dates = set(inc["date"])
    clean = base[~base["date"].isin(inc_dates)].copy()
    clean["category"] = "no_incident"

    combined = pd.concat([merged, clean], ignore_index=True)
    grouped = (
        combined.groupby("category", as_index=False)[outcome_col]
        .agg(
            n_days="count",
            mean_residual_pct="mean",
            std_residual_pct=lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        )
        .sort_values(["mean_residual_pct", "category"])
        .reset_index(drop=True)
    )
    return grouped


def audit_intraday_incident_hours(
    hourly_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
    window_hours: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """Compare hourly revenue during incident (h, h+1) windows vs same-hour baseline."""
    hr = hourly_df.copy()
    hr["date"] = pd.to_datetime(hr["date"], errors="coerce").dt.normalize()
    hr["hour_int"] = pd.to_numeric(hr["hour_int"], errors="coerce")
    hr["volume"] = pd.to_numeric(hr["volume"], errors="coerce").fillna(0.0)
    hr = hr.dropna(subset=["date", "hour_int"]).copy()
    hr["hour_int"] = hr["hour_int"].astype(int)

    affected_slots: set[tuple[pd.Timestamp, int]] = set()
    if not incidents_df.empty and {"date", "hour_int"}.issubset(incidents_df.columns):
        inc = incidents_df.copy()
        inc["date"] = pd.to_datetime(inc["date"], errors="coerce").dt.normalize()
        inc["hour_int"] = pd.to_numeric(inc["hour_int"], errors="coerce")
        for _, row in inc.dropna(subset=["date", "hour_int"]).iterrows():
            for off in window_hours:
                affected_slots.add((row["date"], int(row["hour_int"]) + int(off)))

    hr["is_affected"] = [
        (d, h) in affected_slots for d, h in zip(hr["date"], hr["hour_int"])
    ]
    base = hr[~hr["is_affected"]].groupby("hour_int")["volume"].mean().rename("baseline_mean_volume")
    aff = (
        hr[hr["is_affected"]]
        .groupby("hour_int")
        .agg(affected_obs=("volume", "count"), affected_mean_volume=("volume", "mean"))
    )
    joined = pd.concat([base, aff], axis=1).reset_index()
    joined["affected_obs"] = joined["affected_obs"].fillna(0).astype(int)
    joined["diff_pct"] = np.where(
        (joined["baseline_mean_volume"] > 0) & joined["affected_mean_volume"].notna(),
        (joined["affected_mean_volume"] - joined["baseline_mean_volume"])
        / joined["baseline_mean_volume"],
        np.nan,
    )
    return joined.sort_values("hour_int").reset_index(drop=True)


def ring_confidence_intervals(
    joined_df: pd.DataFrame,
    ring_summary: pd.DataFrame,
    ring_col: str = "nearest_ring_lag0",
    outcome_col: str = "residual_pct",
) -> pd.DataFrame:
    """Attach 95% normal-approximation confidence intervals per geofence ring."""
    out = ring_summary.copy()
    stats_map: dict[str, tuple[int, float, float]] = {}
    for ring, grp in joined_df.groupby(ring_col):
        vals = grp[outcome_col].dropna()
        n = int(len(vals))
        mean_v = float(vals.mean()) if n else np.nan
        half = 1.96 * float(vals.std(ddof=1)) / math.sqrt(n) if n >= 2 else np.nan
        stats_map[str(ring)] = (n, mean_v, half)

    ci_half: list[float] = []
    for _, row in out.iterrows():
        n_obs = int(row.get("n_obs", 0))
        if n_obs >= 2 and "std_residual_pct" in out.columns and pd.notna(row["std_residual_pct"]):
            ci_half.append(1.96 * float(row["std_residual_pct"]) / math.sqrt(n_obs))
        else:
            ci_half.append(stats_map.get(str(row["ring"]), (0, np.nan, np.nan))[2])

    out["ci_half_width"] = ci_half
    out["ci_low"] = out["mean_residual_pct"] - out["ci_half_width"]
    out["ci_high"] = out["mean_residual_pct"] + out["ci_half_width"]
    return out


def _resolve_lat_lon_cols(df: pd.DataFrame) -> tuple[str, str]:
    if {"lat", "lon"}.issubset(df.columns):
        return "lat", "lon"
    if {"latitude", "longitude"}.issubset(df.columns):
        return "latitude", "longitude"
    raise KeyError(f"Expected ('lat','lon') or ('latitude','longitude') in {list(df.columns)}")


def quadrant_summary(
    enriched_inc: pd.DataFrame,
    joined_df: pd.DataFrame,
    store_lat: float,
    store_lon: float,
    outcome_col: str = "residual_pct",
) -> pd.DataFrame:
    """Summarize incident counts and sales residuals across NW/NE/SW/SE quadrants."""
    if enriched_inc.empty:
        return pd.DataFrame(
            columns=["quadrant", "incident_count", "affected_days", "mean_residual_pct", "std_residual_pct"]
        )

    lat_col, lon_col = _resolve_lat_lon_cols(enriched_inc)
    inc = enriched_inc.copy()
    inc["date"] = pd.to_datetime(inc["date"], errors="coerce").dt.normalize()
    inc_lat = pd.to_numeric(inc[lat_col], errors="coerce")
    inc_lon = pd.to_numeric(inc[lon_col], errors="coerce")
    dlat = inc_lat - float(store_lat)
    dlon = (inc_lon - float(store_lon)) * math.cos(math.radians(float(store_lat)))
    bearing_rad = np.arctan2(dlat, dlon)

    ns = np.where(np.sin(bearing_rad) >= 0, "N", "S")
    ew = np.where(np.cos(bearing_rad) >= 0, "E", "W")
    inc["quadrant"] = [f"{a}{b}" for a, b in zip(ns, ew)]

    sales = joined_df[["date", outcome_col]].copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce").dt.normalize()

    counts = inc.groupby("quadrant", as_index=False).size().rename(columns={"size": "incident_count"})
    date_quads = inc[["date", "quadrant"]].dropna().drop_duplicates()
    merged = date_quads.merge(sales, on="date", how="inner")
    res_tbl = merged.groupby("quadrant", as_index=False)[outcome_col].agg(
        affected_days="count",
        mean_residual_pct="mean",
        std_residual_pct=lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan,
    )
    out = counts.merge(res_tbl, on="quadrant", how="left").sort_values("quadrant").reset_index(drop=True)
    return out


def weekday_weekend_peak_table(
    hourly_df: pd.DataFrame, peak_hours: Sequence[int]
) -> pd.DataFrame:
    """Filter hourly Weekday vs Weekend profile strictly to configured peak_hours."""
    ww = split_weekday_weekend_hourly(hourly_df, peak_hours=peak_hours)
    return ww[ww["is_peak"]].reset_index(drop=True)


def attach_dilution_stress_test(
    baseline: dict[str, Any],
    cfg: StoreConfig,
    attach_lifts: Sequence[float] = (-0.05, 0.0, 0.05),
    base_peak_lift_pct: float = 0.10,
    labor_hours_per_day: float | None = None,
) -> pd.DataFrame:
    """Stress-test Base peak-hour lift against negative/zero/positive attach-rate shifts."""
    scenarios = [
        (f"Attach {a:+.0%}", float(base_peak_lift_pct), float(a))
        for a in attach_lifts
    ]
    return simulate_linebuster_scenarios(
        baseline=baseline,
        config=cfg,
        scenarios=scenarios,
        labor_hours_per_day=labor_hours_per_day,
    )


def pilot_mde_table(
    clean_daily_df: pd.DataFrame,
    clean_hourly_df: pd.DataFrame,
    peak_hours: Sequence[int],
    horizons_days: Sequence[int] = (14, 28, 42, 56, 84),
    weekdays_only: bool = False,
) -> pd.DataFrame:
    """Compute two-sided alpha=0.05, 80% power MDE% for daily and peak-hour revenue."""
    daily_work = clean_daily_df.copy()
    daily_work["date"] = pd.to_datetime(daily_work["date"], errors="coerce")
    daily_work = daily_work.dropna(subset=["date"])

    hr = clean_hourly_df.copy()
    hr["date"] = pd.to_datetime(hr["date"], errors="coerce").dt.normalize()
    hr["hour_int"] = pd.to_numeric(hr["hour_int"], errors="coerce")
    hr["volume"] = pd.to_numeric(hr["volume"], errors="coerce").fillna(0.0)
    hr = hr.dropna(subset=["date"])

    if weekdays_only:
        daily_work = daily_work[daily_work["date"].dt.day_of_week < 5]
        hr = hr[hr["date"].dt.day_of_week < 5]

    daily_rev = pd.to_numeric(daily_work["revenue"], errors="coerce").dropna()
    mu_daily = float(daily_rev.mean()) if not daily_rev.empty else np.nan
    sd_daily = float(daily_rev.std(ddof=1)) if len(daily_rev) > 1 else np.nan

    peak_daily = (
        hr[hr["hour_int"].isin(set(int(h) for h in peak_hours))]
        .groupby("date")["volume"]
        .sum()
    )
    mu_peak = float(peak_daily.mean()) if not peak_daily.empty else np.nan
    sd_peak = float(peak_daily.std(ddof=1)) if len(peak_daily) > 1 else np.nan

    z_sum = 1.96 + 0.84
    scope_label = "Mon-Fri" if weekdays_only else "Mon-Sun"
    rows: list[dict[str, float | int | str]] = []
    for days in horizons_days:
        t_val = int(days)
        factor = z_sum * math.sqrt(2.0 / t_val)
        daily_abs = factor * sd_daily if np.isfinite(sd_daily) else np.nan
        peak_abs = factor * sd_peak if np.isfinite(sd_peak) else np.nan
        rows.append(
            {
                "schedule_scope": scope_label,
                "pilot_days": t_val,
                "calendar_weeks": round(t_val / 5.0 if weekdays_only else t_val / 7.0, 1),
                "daily_mde_abs": round(daily_abs, 2) if np.isfinite(daily_abs) else np.nan,
                "daily_mde_pct": round(daily_abs / mu_daily, 4)
                if np.isfinite(daily_abs) and mu_daily > 0
                else np.nan,
                "peak_mde_abs": round(peak_abs, 2) if np.isfinite(peak_abs) else np.nan,
                "peak_mde_pct": round(peak_abs / mu_peak, 4)
                if np.isfinite(peak_abs) and mu_peak > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def regime_and_trend_audit(
    joined_df: pd.DataFrame, tablet_start_date: str
) -> dict[str, pd.DataFrame]:
    """Separate post-POS regime shifts from acute weather vs street-incident shocks."""
    work = joined_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).copy()
    cutoff = pd.Timestamp(tablet_start_date)
    work["regime"] = np.where(
        work["date"] >= cutoff, "Post POS Change", "Pre POS Change"
    )
    has_inc = (
        work["has_incident_lag0"].fillna(False).astype(bool)
        if "has_incident_lag0" in work.columns
        else pd.Series(False, index=work.index)
    )
    work["_inc"] = has_inc.astype(int)

    reg_agg: dict[str, Any] = {
        "days": ("date", "count"),
        "mean_residual_pct": ("residual_pct", "mean"),
        "incident_days": ("_inc", "sum"),
    }
    if "revenue" in work.columns:
        reg_agg["mean_revenue"] = ("revenue", "mean")
    regime_tbl = work.groupby("regime", as_index=False).agg(**reg_agg)
    regime_tbl["incident_share"] = np.where(
        regime_tbl["days"] > 0,
        regime_tbl["incident_days"] / regime_tbl["days"],
        0.0,
    )

    cat_col = (
        work["primary_category_lag0"].fillna("none").astype(str)
        if "primary_category_lag0" in work.columns
        else pd.Series("none", index=work.index)
    )
    work["disruption_group"] = np.where(
        ~has_inc,
        "No Incident",
        np.where(cat_col == "weather_disruption", "Weather", "Street / Traffic / Safety"),
    )
    ws_tbl = (
        work.groupby("disruption_group", as_index=False)["residual_pct"]
        .agg(
            days="count",
            mean_residual_pct="mean",
            std_residual_pct=lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        )
        .sort_values("mean_residual_pct")
        .reset_index(drop=True)
    )
    return {"regime_summary": regime_tbl, "weather_vs_street": ws_tbl}
