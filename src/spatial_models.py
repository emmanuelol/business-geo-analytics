"""Spatial distance-decay models, geofence buffers, residualization, and lag joins."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.config import GEOFENCE_RADII_M

_EARTH_RADIUS_M = 6_371_000.0
_QUINCENA_DAYS: frozenset[int] = frozenset({1, 14, 15, 16, 28, 29, 30, 31})
MX_FIXED_HOLIDAYS: frozenset[tuple[int, int]] = frozenset(
    {(1, 1), (2, 5), (3, 21), (5, 1), (9, 16), (11, 20), (12, 25)}
)


def haversine_meters(
    lat1: float | np.ndarray | pd.Series,
    lon1: float | np.ndarray | pd.Series,
    lat2: float | np.ndarray | pd.Series,
    lon2: float | np.ndarray | pd.Series,
) -> float | np.ndarray:
    """Compute great-circle Haversine distance in meters (scalar or vectorized)."""
    is_scalar = all(np.isscalar(v) for v in (lat1, lon1, lat2, lon2))
    phi1 = np.radians(np.asarray(lat1, dtype=float))
    lam1 = np.radians(np.asarray(lon1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    lam2 = np.radians(np.asarray(lon2, dtype=float))

    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    dist = 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(dist) if is_scalar else dist


def assign_geofence_ring(
    distance_m: float | np.ndarray | pd.Series,
    radii_m: Sequence[int] = GEOFENCE_RADII_M,
) -> str | pd.Series:
    """Assign each distance (m) to mutually exclusive rings like '0-500m', '500-1500m'."""
    sorted_radii = sorted(int(r) for r in radii_m)
    if not sorted_radii or any(r <= 0 for r in sorted_radii):
        raise ValueError(f"radii_m must contain positive integers, got {radii_m}")

    labels: list[str] = []
    prev = 0
    for r in sorted_radii:
        labels.append(f"{prev}-{r}m")
        prev = r
    labels.append(f">{sorted_radii[-1]}m")

    def _classify_one(val: float) -> str:
        if np.isnan(val):
            return "unknown"
        for idx, r in enumerate(sorted_radii):
            if val <= r:
                return labels[idx]
        return labels[-1]

    if np.isscalar(distance_m):
        return _classify_one(float(distance_m))

    series = pd.Series(distance_m, dtype=float)
    return series.map(_classify_one)


def distance_decay_weight(
    distance_m: float | np.ndarray | pd.Series,
    half_life_m: float = 1500.0,
    kernel: str = "exponential",
    max_distance_m: float | None = None,
) -> float | np.ndarray:
    """Compute distance-decay weights in [0, 1] with optional max cutoff."""
    if half_life_m <= 0:
        raise ValueError(f"half_life_m must be positive, got {half_life_m}")
    is_scalar = np.isscalar(distance_m)
    d = np.maximum(0.0, np.asarray(distance_m, dtype=float))

    if kernel == "exponential":
        w = np.exp(-math.log(2.0) * d / float(half_life_m))
    elif kernel == "inverse":
        w = 1.0 / (1.0 + d / float(half_life_m))
    else:
        raise ValueError(f"Unsupported decay kernel {kernel!r}; use 'exponential' or 'inverse'")

    if max_distance_m is not None:
        w = np.where(d <= float(max_distance_m), w, 0.0)
    return float(w) if is_scalar else w


def enrich_incidents_spatial(
    incidents_df: pd.DataFrame,
    store_lat: float,
    store_lon: float,
    radii_m: Sequence[int] = GEOFENCE_RADII_M,
    half_life_m: float = 1500.0,
    kernel: str = "exponential",
    max_distance_m: float | None = None,
) -> pd.DataFrame:
    """Attach store distance, geofence ring, buffer flags, and exposure weight to incidents."""
    df = incidents_df.copy()
    if df.empty:
        df["distance_m"] = pd.Series(dtype=float)
        df["geofence_ring"] = pd.Series(dtype=str)
        df["decay_weight"] = pd.Series(dtype=float)
        df["exposure_score"] = pd.Series(dtype=float)
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    severity = (
        pd.to_numeric(df["severity"], errors="coerce").fillna(1.0)
        if "severity" in df.columns
        else pd.Series(1.0, index=df.index)
    )

    dists = haversine_meters(store_lat, store_lon, df["lat"].to_numpy(), df["lon"].to_numpy())
    df["distance_m"] = np.round(dists, 2)
    df["geofence_ring"] = assign_geofence_ring(df["distance_m"], radii_m=radii_m).to_numpy()
    for r in sorted(int(x) for x in radii_m):
        df[f"within_{r}m"] = df["distance_m"] <= r

    weights = distance_decay_weight(
        df["distance_m"].to_numpy(),
        half_life_m=half_life_m,
        kernel=kernel,
        max_distance_m=max_distance_m,
    )
    df["decay_weight"] = np.round(weights, 6)
    df["exposure_score"] = np.round(severity.to_numpy() * df["decay_weight"].to_numpy(), 6)
    return df


def _append_calendar_controls(
    df: pd.DataFrame,
    date_col: str,
    design_parts: list[pd.DataFrame],
    regime_dates: Iterable[str | pd.Timestamp] | None,
    include_quincena: bool,
    holiday_dates: Iterable[str | pd.Timestamp] | str | None,
    control_cols: Sequence[str] | None,
) -> None:
    if regime_dates:
        for idx, rd in enumerate(regime_dates):
            cutoff = pd.Timestamp(rd)
            design_parts.append(
                pd.DataFrame(
                    {f"_regime_{idx}": (df[date_col] >= cutoff).astype(float)},
                    index=df.index,
                )
            )
    if include_quincena:
        q_flag = df[date_col].dt.day.isin(_QUINCENA_DAYS).astype(float)
        if "is_quincena" not in df.columns:
            df["is_quincena"] = q_flag.astype(bool)
        design_parts.append(pd.DataFrame({"_quincena": q_flag}, index=df.index))

    if holiday_dates is not None:
        if isinstance(holiday_dates, str) and holiday_dates.lower() == "mx":
            h_flag = pd.Series(
                [
                    (int(m), int(d)) in MX_FIXED_HOLIDAYS
                    for m, d in zip(df[date_col].dt.month, df[date_col].dt.day)
                ],
                index=df.index,
                dtype=float,
            )
        else:
            hol_set = {pd.Timestamp(h).normalize() for h in holiday_dates}
            h_flag = df[date_col].dt.normalize().isin(hol_set).astype(float)
        if "is_holiday" not in df.columns:
            df["is_holiday"] = h_flag.astype(bool)
        design_parts.append(pd.DataFrame({"_holiday": h_flag}, index=df.index))

    if control_cols:
        for col in control_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
                design_parts.append(pd.DataFrame({f"_ctrl_{col}": vals}, index=df.index))


def compute_sales_residuals(
    sales_df: pd.DataFrame,
    value_col: str = "revenue",
    date_col: str = "date",
    extra_group_cols: Sequence[str] | None = None,
    regime_dates: Iterable[str | pd.Timestamp] | None = None,
    include_quincena: bool = False,
    holiday_dates: Iterable[str | pd.Timestamp] | str | None = None,
    control_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Estimate baseline expected sales via OLS (trend + DOW + optional controls) and residuals."""
    df = sales_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)

    exp_col = f"expected_{value_col}"
    res_col = f"residual_{value_col}"
    if valid.sum() < 4:
        df[exp_col] = np.nanmean(y) if valid.any() else 0.0
        df[res_col] = y - df[exp_col]
        df["residual_pct"] = np.where(df[exp_col] > 0, df[res_col] / df[exp_col], 0.0)
        return df

    t_days = (df[date_col] - df[date_col].min()).dt.total_seconds() / 86400.0
    t_scale = max(float(t_days.max()), 1.0)
    design_parts: list[pd.DataFrame] = [
        pd.DataFrame(
            {
                "intercept": 1.0,
                "trend": t_days / t_scale,
            },
            index=df.index,
        ),
        pd.get_dummies(df[date_col].dt.day_of_week, prefix="dow", drop_first=True, dtype=float),
    ]
    if extra_group_cols:
        for col in extra_group_cols:
            design_parts.append(
                pd.get_dummies(df[col], prefix=str(col), drop_first=True, dtype=float)
            )
    _append_calendar_controls(
        df=df,
        date_col=date_col,
        design_parts=design_parts,
        regime_dates=regime_dates,
        include_quincena=include_quincena,
        holiday_dates=holiday_dates,
        control_cols=control_cols,
    )

    X = pd.concat(design_parts, axis=1).to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
    expected = np.maximum(X @ beta, 1e-6)
    residual = y - expected

    df[exp_col] = np.round(expected, 4)
    df[res_col] = np.round(residual, 4)
    df["residual_pct"] = np.round(residual / expected, 6)
    return df


def _aggregate_exposure_frame(
    enriched: pd.DataFrame,
    key_cols: list[str],
    max_distance_m: float | None,
) -> pd.DataFrame:
    scoped = enriched.dropna(subset=key_cols).copy()
    if max_distance_m is not None:
        scoped = scoped[scoped["distance_m"] <= float(max_distance_m)]

    if scoped.empty:
        return pd.DataFrame(
            columns=key_cols
            + [
                "incident_count",
                "incident_exposure",
                "nearest_incident_m",
                "nearest_ring",
                "primary_category",
            ]
        )

    scoped = scoped.sort_values(key_cols + ["distance_m"])
    agg = (
        scoped.groupby(key_cols, as_index=False)
        .agg(
            incident_count=("exposure_score", "size"),
            incident_exposure=("exposure_score", "sum"),
            nearest_incident_m=("distance_m", "min"),
            nearest_ring=("geofence_ring", "first"),
            primary_category=("category", "first")
            if "category" in scoped.columns
            else ("geofence_ring", "first"),
        )
    )
    return agg


def join_sales_with_incidents(
    sales_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
    store_lat: float,
    store_lon: float,
    freq: str = "D",
    radii_m: Sequence[int] = GEOFENCE_RADII_M,
    half_life_m: float = 1500.0,
    max_distance_m: float | None = 5000.0,
    value_col: str = "revenue",
    lags: Sequence[int] = (0, 1, 2),
    regime_dates: Iterable[str | pd.Timestamp] | None = None,
    include_quincena: bool = False,
    holiday_dates: Iterable[str | pd.Timestamp] | str | None = None,
    residual_control_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join sales residuals with geo-fenced incident exposure and chronological lag t..t+k."""
    freq_norm = freq.upper()
    if freq_norm not in {"D", "H"}:
        raise ValueError(f"freq must be 'D' or 'H', got {freq!r}")

    extra_cols = ["hour_int"] if freq_norm == "H" and "hour_int" in sales_df.columns else None
    sales = compute_sales_residuals(
        sales_df,
        value_col=value_col,
        date_col="date",
        extra_group_cols=extra_cols,
        regime_dates=regime_dates,
        include_quincena=include_quincena,
        holiday_dates=holiday_dates,
        control_cols=residual_control_cols,
    )

    enriched = enrich_incidents_spatial(
        incidents_df,
        store_lat=store_lat,
        store_lon=store_lon,
        radii_m=radii_m,
        half_life_m=half_life_m,
        max_distance_m=max_distance_m,
    )

    key_cols = ["date"] if freq_norm == "D" else ["date", "hour_int"]
    if freq_norm == "H" and "hour_int" in enriched.columns:
        enriched["hour_int"] = pd.to_numeric(enriched["hour_int"], errors="coerce")

    agg = _aggregate_exposure_frame(enriched, key_cols=key_cols, max_distance_m=max_distance_m)

    if freq_norm == "D":
        min_d = sales["date"].min()
        max_d = sales["date"].max()
        grid = pd.DataFrame({"date": pd.date_range(min_d, max_d, freq="D")})
    else:
        dates = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
        hours = sorted(sales["hour_int"].dropna().astype(int).unique())
        idx = pd.MultiIndex.from_product([dates, hours], names=["date", "hour_int"])
        grid = pd.DataFrame(index=idx).reset_index()

    grid = pd.merge(grid, agg, on=key_cols, how="left")
    grid["incident_count"] = grid["incident_count"].fillna(0).astype(int)
    grid["incident_exposure"] = grid["incident_exposure"].fillna(0.0).astype(float)
    grid["nearest_ring"] = grid["nearest_ring"].fillna("no_incident")

    for lag in lags:
        lag_i = int(lag)
        grid[f"incident_exposure_lag{lag_i}"] = (
            grid["incident_exposure"].shift(lag_i).fillna(0.0)
        )
        grid[f"has_incident_lag{lag_i}"] = (
            grid["incident_count"].shift(lag_i).fillna(0) > 0
        )

    grid = grid.rename(
        columns={
            "incident_count": "incident_count_lag0",
            "nearest_incident_m": "nearest_incident_m_lag0",
            "nearest_ring": "nearest_ring_lag0",
            "primary_category": "primary_category_lag0",
        }
    )

    keep_cols = key_cols + [
        "incident_count_lag0",
        "nearest_incident_m_lag0",
        "nearest_ring_lag0",
        "primary_category_lag0",
    ] + [
        c
        for lag in lags
        for c in (f"incident_exposure_lag{int(lag)}", f"has_incident_lag{int(lag)}")
    ]
    return pd.merge(sales, grid[keep_cols], on=key_cols, how="left")


def estimate_lag_effects(
    joined_df: pd.DataFrame,
    outcome_col: str = "residual_pct",
    exposure_prefix: str = "incident_exposure_lag",
    lags: Sequence[int] = (0, 1, 2),
    control_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Estimate joint OLS impact of t, t+1, t+2 incident exposure on sales residuals."""
    lag_list = [int(k) for k in lags]
    cols = [f"{exposure_prefix}{k}" for k in lag_list]
    ctrl_list = [c for c in control_cols if c in joined_df.columns]
    work = joined_df[[outcome_col] + cols + ctrl_list].dropna().copy()

    n_obs = len(work)
    n_params = 1 + len(lag_list) + len(ctrl_list)
    if n_obs <= n_params:
        return pd.DataFrame(
            [
                {
                    "lag": k,
                    "coef": np.nan,
                    "std_err": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_obs": n_obs,
                    "r_squared": np.nan,
                }
                for k in lag_list
            ]
        )

    y = work[outcome_col].to_numpy(dtype=float)
    X_parts = [np.ones(n_obs, dtype=float), work[cols].to_numpy(dtype=float)]
    if ctrl_list:
        X_parts.append(work[ctrl_list].astype(float).to_numpy(dtype=float))
    X = np.column_stack(X_parts)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df_resid = n_obs - X.shape[1]
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0

    sigma2 = rss / max(df_resid, 1)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_crit = float(stats.t.ppf(0.975, df=max(df_resid, 1)))

    rows: list[dict[str, float | int]] = []
    for idx, k in enumerate(lag_list, start=1):
        coef = float(beta[idx])
        err = float(se[idx])
        t_val = coef / err if err > 0 else np.nan
        p_val = (
            float(2.0 * stats.t.sf(abs(t_val), df=max(df_resid, 1)))
            if np.isfinite(t_val)
            else np.nan
        )
        rows.append(
            {
                "lag": k,
                "coef": round(coef, 6),
                "std_err": round(err, 6),
                "t_stat": round(t_val, 4) if np.isfinite(t_val) else np.nan,
                "p_value": round(p_val, 6) if np.isfinite(p_val) else np.nan,
                "ci_low": round(coef - t_crit * err, 6) if np.isfinite(err) else np.nan,
                "ci_high": round(coef + t_crit * err, 6) if np.isfinite(err) else np.nan,
                "n_obs": n_obs,
                "r_squared": round(r2, 6),
            }
        )
    return pd.DataFrame(rows)


def summarize_geofence_sensitivity(
    joined_df: pd.DataFrame,
    ring_col: str = "nearest_ring_lag0",
    outcome_col: str = "residual_pct",
) -> pd.DataFrame:
    """Compare residual sales anomalies across geofence rings vs no-incident baseline."""
    baseline = joined_df.loc[joined_df[ring_col] == "no_incident", outcome_col].dropna()
    order = ["0-500m", "500-1500m", "1500-5000m", ">5000m", "no_incident"]
    present = list(joined_df[ring_col].dropna().unique())
    rings = [r for r in order if r in present] + [r for r in present if r not in order]

    rows: list[dict[str, object]] = []
    for ring in rings:
        vals = joined_df.loc[joined_df[ring_col] == ring, outcome_col].dropna()
        if ring != "no_incident" and len(vals) >= 2 and len(baseline) >= 2:
            _, p_val = stats.ttest_ind(vals, baseline, equal_var=False)
            p_out = round(float(p_val), 6) if np.isfinite(p_val) else np.nan
        else:
            p_out = np.nan
        rows.append(
            {
                "ring": ring,
                "n_obs": int(len(vals)),
                "mean_residual_pct": round(float(vals.mean()), 6) if len(vals) else np.nan,
                "median_residual_pct": round(float(vals.median()), 6) if len(vals) else np.nan,
                "std_residual_pct": round(float(vals.std(ddof=1)), 6) if len(vals) > 1 else np.nan,
                "p_value_vs_baseline": p_out,
            }
        )
    return pd.DataFrame(rows)
