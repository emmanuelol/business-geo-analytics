"""Fully synthetic sample dataset generator using hard-coded generic retail priors."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_LAT,
    DEFAULT_LON,
    PEAK_HOURS_DEFAULT,
    TABLET_CHANNEL_START_DATE,
)

EXPECTED_FILES: tuple[str, ...] = (
    "daily_sales.csv",
    "hourly_sales.csv",
    "product_mix.csv",
    "incidents.csv",
)

_SYNTHETIC_BRANCH_ID = "SAMPLE-001"

# Hourly demand profile weights for operating hours 6..21 (peaks at 8-9 and 18-20)
_HOUR_WEIGHTS: dict[int, float] = {
    6: 0.015,
    7: 0.055,
    8: 0.095,
    9: 0.088,
    10: 0.062,
    11: 0.058,
    12: 0.052,
    13: 0.050,
    14: 0.048,
    15: 0.055,
    16: 0.064,
    17: 0.070,
    18: 0.092,
    19: 0.096,
    20: 0.080,
    21: 0.015,
}

_MENU_CATALOG: tuple[tuple[str, str, float], ...] = (
    ("Americano", "Beverages", 0.16),
    ("Espresso", "Beverages", 0.07),
    ("Iced Latte", "Beverages", 0.17),
    ("Matcha Latte", "Beverages", 0.09),
    ("Cold Brew", "Beverages", 0.07),
    ("Chai Latte", "Beverages", 0.07),
    ("Butter Croissant", "Food_Pastries", 0.10),
    ("Ham & Cheese Panini", "Food_Pastries", 0.08),
    ("Chocolate Brownie", "Food_Pastries", 0.06),
    ("Lemon Cake", "Food_Pastries", 0.04),
    ("Iced Latte + Butter Croissant", "Combos", 0.04),
    ("Americano + Chocolate Brownie", "Combos", 0.03),
    ("House Blend 250g", "Retail", 0.012),
    ("Ceramic Mug", "Retail", 0.008),
)

_INCIDENT_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "category": "traffic_accident",
        "language": "es",
        "title": "Accidente vial bloquea carril en Libramiento Norte",
        "summary": "Choque multiple genera congestion severa en cruce cercano.",
        "source": "SyntheticLocalFeed",
        "location_name": "Libramiento Norte, Silao",
    },
    {
        "category": "road_closure",
        "language": "es",
        "title": "Cierre parcial por obras de pavimentacion en boulevard",
        "summary": "Desvio vehicular durante horas pico por mantenimiento urbano.",
        "source": "SyntheticMunicipio",
        "location_name": "Blvd. Raul Bailleres, Silao",
    },
    {
        "category": "public_safety",
        "language": "es",
        "title": "Operativo de seguridad y cierre preventivo en zona centro",
        "summary": "Despliegue policial reduce flujo peatonal durante la tarde.",
        "source": "SyntheticPress",
        "location_name": "Zona Centro, Silao",
    },
    {
        "category": "weather_disruption",
        "language": "en",
        "title": "Heavy thunderstorm causes street flooding on main corridor",
        "summary": "Intense rainfall slows evening commute and storefront walk-ins.",
        "source": "SyntheticWeather",
        "location_name": "Corredor Industrial, Silao",
    },
)


def _offset_coords(
    lat: float, lon: float, distance_m: float, bearing_deg: float
) -> tuple[float, float]:
    earth_r_m = 6_371_000.0
    bearing = math.radians(bearing_deg)
    d_lat = (distance_m * math.cos(bearing)) / earth_r_m
    d_lon = (distance_m * math.sin(bearing)) / (earth_r_m * math.cos(math.radians(lat)))
    return round(lat + math.degrees(d_lat), 6), round(lon + math.degrees(d_lon), 6)


def _build_synthetic_incidents(
    dates: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    chosen_indices = sorted(rng.choice(np.arange(10, len(dates) - 5), size=24, replace=False))
    distances_m = [250.0, 420.0, 850.0, 1300.0, 2600.0, 4400.0, 6500.0]
    rows: list[dict[str, object]] = []

    for i, idx in enumerate(chosen_indices):
        tpl = _INCIDENT_TEMPLATES[i % len(_INCIDENT_TEMPLATES)]
        dist_m = float(distances_m[i % len(distances_m)])
        bearing = float(rng.uniform(0.0, 360.0))
        inc_lat, inc_lon = _offset_coords(DEFAULT_LAT, DEFAULT_LON, dist_m, bearing)
        hour_int = int(rng.choice([8, 9, 13, 17, 18, 19]))
        dt = dates[idx]
        severity = round(float(rng.uniform(0.45, 0.95)), 2)
        rows.append(
            {
                "incident_id": f"INC-{i + 1:03d}",
                "timestamp": f"{dt.strftime('%Y-%m-%d')}T{hour_int:02d}:15:00",
                "date": dt.strftime("%Y-%m-%d"),
                "hour_int": hour_int,
                "title": tpl["title"],
                "summary": tpl["summary"],
                "source": tpl["source"],
                "category": tpl["category"],
                "severity": severity,
                "lat": inc_lat,
                "lon": inc_lon,
                "location_name": tpl["location_name"],
                "language": tpl["language"],
            }
        )

    return pd.DataFrame(rows)


def _compute_incident_shock_series(
    dates: pd.DatetimeIndex, incidents: pd.DataFrame
) -> np.ndarray:
    """Inject realistic negative foot-traffic shocks for nearby synthetic incidents."""
    shocks = np.zeros(len(dates), dtype=float)
    date_to_pos = {d.strftime("%Y-%m-%d"): idx for idx, d in enumerate(dates)}

    for _, row in incidents.iterrows():
        pos = date_to_pos.get(str(row["date"]))
        if pos is None:
            continue
        # Approximate planar distance from store centroid for shock injection
        dlat = (float(row["lat"]) - DEFAULT_LAT) * 111_000.0
        dlon = (float(row["lon"]) - DEFAULT_LON) * 111_000.0 * math.cos(math.radians(DEFAULT_LAT))
        dist_m = math.hypot(dlat, dlon)
        proximity = math.exp(-dist_m / 1800.0)
        magnitude = 0.22 * float(row["severity"]) * proximity
        shocks[pos] += magnitude
        if pos + 1 < len(shocks):
            shocks[pos + 1] += 0.4 * magnitude
    return np.clip(shocks, 0.0, 0.45)


def generate_sample_datasets(
    output_dir: Path | str | None = None,
    days: int = 180,
    start_date: str = "2026-05-01",
    seed: int = 42,
) -> dict[str, Path]:
    """Generate 4 fully synthetic sample CSV files from generic retail priors."""
    rng = np.random.default_rng(seed)
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent.parent / "data" / "sample"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start=start_date, periods=days, freq="D")
    incidents_df = _build_synthetic_incidents(dates, rng)
    shock_drag = _compute_incident_shock_series(dates, incidents_df)

    # Generic specialty-coffee daily revenue baseline (unrelated to private store figures)
    dow_mult = np.array([0.88, 0.92, 0.96, 1.00, 1.12, 1.24, 1.16])
    day_idx = np.arange(days, dtype=float)
    trend = 1.0 + 0.08 * (day_idx / max(days - 1, 1))
    noise = rng.normal(1.0, 0.08, size=days)
    base_rev = 3200.0 * dow_mult[dates.day_of_week.to_numpy()] * trend * noise
    daily_rev = np.maximum(900.0, np.round(base_rev * (1.0 - shock_drag), 2))

    avg_ticket = np.round(rng.normal(128.0, 9.0, size=days).clip(95.0, 165.0), 2)
    tickets = np.maximum(8, np.round(daily_rev / avg_ticket).astype(int))
    daily_rev = np.round(tickets * avg_ticket, 2)
    net_sales = np.round(daily_rev / 1.16, 2)
    iva = np.round(daily_rev - net_sales, 2)
    upt = np.round(rng.normal(2.15, 0.22, size=days).clip(1.4, 3.2), 2)

    card_share = rng.uniform(0.62, 0.78, size=days)
    app_share = rng.uniform(0.06, 0.14, size=days)
    cash_share = np.maximum(0.05, 1.0 - card_share - app_share)
    food_share = rng.uniform(0.22, 0.30, size=days)
    shop_share = rng.uniform(0.01, 0.04, size=days)
    drink_share = 1.0 - food_share - shop_share

    tablet_cutoff = pd.Timestamp(TABLET_CHANNEL_START_DATE)
    daily_df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "branch_id": _SYNTHETIC_BRANCH_ID,
            "revenue": daily_rev,
            "net_sales": net_sales,
            "iva": iva,
            "tickets": tickets,
            "avg_ticket": avg_ticket,
            "upt": upt,
            "gross_sales_local": daily_rev,
            "net_revenue_local": net_sales,
            "card_sales": np.round(daily_rev * card_share, 2),
            "delivery_sales": np.round(daily_rev * rng.uniform(0.0, 0.04, size=days), 2),
            "app_sales": np.round(daily_rev * app_share, 2),
            "gross_cash_sales": np.round(daily_rev * cash_share, 2),
            "total_tips": np.round(daily_rev * rng.uniform(0.04, 0.07, size=days), 2),
            "net_cash_deposit": np.round(daily_rev * cash_share * 0.92, 2),
            "food_sales": np.round(daily_rev * food_share, 2),
            "drink_sales": np.round(daily_rev * drink_share, 2),
            "shop_sales": np.round(daily_rev * shop_share, 2),
            "is_post_tablet_change": dates >= tablet_cutoff,
        }
    )

    # Hourly sales across operating hours 6..21
    hours = sorted(_HOUR_WEIGHTS.keys())
    base_weights = np.array([_HOUR_WEIGHTS[h] for h in hours], dtype=float)
    base_weights = base_weights / base_weights.sum()
    hourly_rows: list[dict[str, object]] = []
    for d_pos, dt in enumerate(dates):
        day_weights = base_weights * rng.uniform(0.85, 1.15, size=len(hours))
        day_weights = day_weights / day_weights.sum()
        vols = np.round(daily_rev[d_pos] * day_weights, 2)
        for h, vol in zip(hours, vols):
            hourly_rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "branch_id": _SYNTHETIC_BRANCH_ID,
                    "hour_int": int(h),
                    "hour_label": f"{int(h):02d}:00",
                    "volume": float(vol),
                    "is_peak": bool(h in PEAK_HOURS_DEFAULT),
                }
            )
    hourly_df = pd.DataFrame(hourly_rows)

    # Product mix across all 4 categories
    sku_probs = np.array([item[2] for item in _MENU_CATALOG], dtype=float)
    sku_probs = sku_probs / sku_probs.sum()
    mix_rows: list[dict[str, object]] = []
    for d_pos, dt in enumerate(dates):
        total_units = max(12, int(round(tickets[d_pos] * upt[d_pos])))
        counts = rng.multinomial(total_units, sku_probs)
        for (sku, cat, _), qty in zip(_MENU_CATALOG, counts):
            if qty > 0:
                mix_rows.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "branch_id": _SYNTHETIC_BRANCH_ID,
                        "product_name": sku,
                        "category": cat,
                        "quantity_sold": int(qty),
                    }
                )
    mix_df = pd.DataFrame(mix_rows)

    outputs = {
        "daily_sales.csv": target_dir / "daily_sales.csv",
        "hourly_sales.csv": target_dir / "hourly_sales.csv",
        "product_mix.csv": target_dir / "product_mix.csv",
        "incidents.csv": target_dir / "incidents.csv",
    }
    daily_df.to_csv(outputs["daily_sales.csv"], index=False)
    hourly_df.to_csv(outputs["hourly_sales.csv"], index=False)
    mix_df.to_csv(outputs["product_mix.csv"], index=False)
    incidents_df.to_csv(outputs["incidents.csv"], index=False)
    return outputs


if __name__ == "__main__":
    written = generate_sample_datasets()
    for name, path in written.items():
        print(f"Generated {name}: {path}")
