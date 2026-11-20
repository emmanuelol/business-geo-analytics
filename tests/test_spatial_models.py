"""Tests for src.spatial_models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spatial_models import (
    assign_geofence_ring,
    compute_sales_residuals,
    distance_decay_weight,
    enrich_incidents_spatial,
    estimate_lag_effects,
    haversine_meters,
    join_sales_with_incidents,
    summarize_geofence_sensitivity,
)


def test_haversine_and_geofence_rings() -> None:
    # Identical points -> 0 m
    assert haversine_meters(20.95, -101.42, 20.95, -101.42) == pytest.approx(0.0, abs=1e-6)

    # 1 degree latitude ~ 111.195 km
    one_deg = haversine_meters(20.0, -101.0, 21.0, -101.0)
    assert 111_000.0 <= one_deg <= 111_400.0

    # Vectorized array support + ring assignment
    dists = np.array([150.0, 500.0, 1200.0, 3200.0, 7500.0])
    rings = list(assign_geofence_ring(dists, radii_m=(500, 1500, 5000)))
    assert rings == ["0-500m", "0-500m", "500-1500m", "1500-5000m", ">5000m"]


def test_distance_decay_monotonic_and_cutoff() -> None:
    w0 = distance_decay_weight(0.0, half_life_m=1500.0)
    w_half = distance_decay_weight(1500.0, half_life_m=1500.0)
    w_far = distance_decay_weight(6000.0, half_life_m=1500.0, max_distance_m=5000.0)

    assert w0 == pytest.approx(1.0)
    assert w_half == pytest.approx(0.5, rel=1e-6)
    assert w_far == pytest.approx(0.0)


def test_join_calendar_lags_and_recover_synthetic_shock() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-06-01", periods=120, freq="D")
    base = 4000.0 + 300.0 * np.sin(2 * np.pi * dates.day_of_week.to_numpy() / 7.0)

    # Place 10 high-severity nearby incidents at ~200m
    inc_dates = dates[[10, 22, 35, 48, 60, 72, 84, 95, 105, 112]]
    incidents = pd.DataFrame(
        {
            "incident_id": [f"I-{i}" for i in range(len(inc_dates))],
            "date": inc_dates.strftime("%Y-%m-%d"),
            "hour_int": 9,
            "category": "road_closure",
            "severity": 1.0,
            "lat": 20.9518,   # ~200m north of (20.95, -101.42)
            "lon": -101.42,
        }
    )

    # Inject true shock: -20% on day t, -10% on day t+1, 0% on day t+2
    shock = np.zeros(len(dates))
    for idx in [10, 22, 35, 48, 60, 72, 84, 95, 105, 112]:
        shock[idx] -= 0.20
        shock[idx + 1] -= 0.10

    revenue = base * (1.0 + shock) + rng.normal(0.0, 25.0, size=len(dates))
    sales = pd.DataFrame({"date": dates, "revenue": revenue})

    joined = join_sales_with_incidents(
        sales_df=sales,
        incidents_df=incidents,
        store_lat=20.95,
        store_lon=-101.42,
        max_distance_m=5000.0,
    )

    assert "residual_pct" in joined.columns
    assert "incident_exposure_lag0" in joined.columns
    assert "incident_exposure_lag1" in joined.columns
    assert "incident_exposure_lag2" in joined.columns

    # Verify calendar lag alignment: day after inc_dates[0] must have lag1 > 0
    day_t = joined.loc[joined["date"] == inc_dates[0]].iloc[0]
    day_t1 = joined.loc[joined["date"] == inc_dates[0] + pd.Timedelta(days=1)].iloc[0]
    day_t2 = joined.loc[joined["date"] == inc_dates[0] + pd.Timedelta(days=2)].iloc[0]
    assert day_t["incident_exposure_lag0"] > 0.8
    assert day_t1["incident_exposure_lag1"] > 0.8
    assert day_t2["incident_exposure_lag2"] > 0.8

    lags_df = estimate_lag_effects(joined, outcome_col="residual_pct", lags=(0, 1, 2))
    lag0 = lags_df.loc[lags_df["lag"] == 0].iloc[0]
    lag1 = lags_df.loc[lags_df["lag"] == 1].iloc[0]
    lag2 = lags_df.loc[lags_df["lag"] == 2].iloc[0]

    assert lag0["coef"] < -0.10
    assert lag0["p_value"] < 0.01
    assert lag1["coef"] < -0.05
    assert lag1["p_value"] < 0.05
    assert abs(lag2["coef"]) < 0.04


def test_hourly_join_and_geofence_sensitivity() -> None:
    dates = pd.date_range("2026-07-01", periods=14, freq="D")
    rows = []
    for d in dates:
        for h in (8, 9, 18):
            vol = 500.0 if h in (8, 18) else 420.0
            if d == pd.Timestamp("2026-07-05") and h == 8:
                vol = 250.0  # -50% drop during nearby incident
            rows.append({"date": d, "hour_int": h, "volume": vol})
    hourly_df = pd.DataFrame(rows)

    incidents = pd.DataFrame(
        {
            "incident_id": ["H-1", "H-2"],
            "date": ["2026-07-05", "2026-07-09"],
            "hour_int": [8, 18],
            "category": ["traffic_accident", "weather_disruption"],
            "severity": [0.9, 0.7],
            "lat": [20.952, 20.980],  # ~220m vs ~3.3km
            "lon": [-101.42, -101.42],
        }
    )

    enriched = enrich_incidents_spatial(incidents, 20.95, -101.42)
    assert list(enriched["geofence_ring"]) == ["0-500m", "1500-5000m"]

    joined_h = join_sales_with_incidents(
        sales_df=hourly_df,
        incidents_df=incidents,
        store_lat=20.95,
        store_lon=-101.42,
        freq="H",
        value_col="volume",
    )
    assert len(joined_h) == len(hourly_df)
    hit = joined_h[(joined_h["date"] == "2026-07-05") & (joined_h["hour_int"] == 8)].iloc[0]
    assert hit["has_incident_lag0"]
    assert hit["residual_pct"] < -0.25

    res_summary = summarize_geofence_sensitivity(joined_h)
    assert "0-500m" in set(res_summary["ring"])
    assert "no_incident" in set(res_summary["ring"])

    res_only = compute_sales_residuals(
        hourly_df, value_col="volume", extra_group_cols=["hour_int"]
    )
    assert "residual_volume" in res_only.columns
