"""Tests for src.config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import (
    COORDS_VERIFIED_DEFAULT,
    DEFAULT_LAT,
    DEFAULT_LON,
    LINEBUSTER_START_DATE,
    OPERATING_HOURS_RANGE,
    TABLET_CHANNEL_INTERPRETATION,
    TABLET_CHANNEL_START_DATE,
    StoreConfig,
    load_config,
)


def test_defaults_match_confirmed_values(tmp_path: Path) -> None:
    cfg = load_config(assumptions_path=tmp_path / "missing.json", env={})

    assert isinstance(cfg, StoreConfig)
    assert cfg.branch_id == "TC-001"
    assert cfg.lat == pytest.approx(20.95)
    assert cfg.lon == pytest.approx(-101.42)
    assert DEFAULT_LAT == pytest.approx(20.95)
    assert DEFAULT_LON == pytest.approx(-101.42)
    assert cfg.coords_verified is False
    assert COORDS_VERIFIED_DEFAULT is False
    assert cfg.geofence_radii_m == (500, 1500, 5000)
    assert cfg.operating_hours == (6, 21)
    assert list(cfg.operating_hour_range()) == list(range(6, 22))
    assert list(OPERATING_HOURS_RANGE) == list(range(6, 22))
    assert cfg.peak_hours == (8, 9, 18, 19, 20)
    assert cfg.linebuster_start_date is None
    assert LINEBUSTER_START_DATE is None
    assert cfg.tablet_channel_start_date == "2026-09-05"
    assert TABLET_CHANNEL_START_DATE == "2026-09-05"
    assert cfg.tablet_channel_interpretation == "pos_channel_change"
    assert TABLET_CHANNEL_INTERPRETATION == "pos_channel_change"
    assert cfg.hardware_capex > 0
    assert cfg.monthly_saas_fee > 0
    assert cfg.peak_labor_cost_per_hr > 0
    assert 0 < cfg.gross_margin_pct <= 1.0


def test_override_precedence_defaults_lt_json_lt_env(tmp_path: Path) -> None:
    assumptions_file = tmp_path / "local_assumptions.json"
    assumptions_file.write_text(
        json.dumps(
            {
                "lat": 20.945123,
                "lon": -101.428765,
                "COORDS_VERIFIED": True,
                "hardware_capex": 18500.0,
                "monthly_saas_fee": 1500.0,
                "peak_labor_cost_per_hr": 95.0,
                "gross_margin_pct": 0.68,
                "PEAK_HOURS": [7, 8, 9, 18, 19],
            }
        ),
        encoding="utf-8",
    )

    env = {
        "BGA_LAT": "20.949999",
        "BGA_HARDWARE_CAPEX": "21000",
        "BGA_PEAK_HOURS": "8,9,19,20",
        "BGA_LINEBUSTER_START_DATE": "2026-12-01",
        "BGA_TABLET_CHANNEL_INTERPRETATION": "self_service_tablet_rollout",
    }

    cfg = load_config(assumptions_path=assumptions_file, env=env)

    # Env overrides JSON and defaults
    assert cfg.lat == pytest.approx(20.949999)
    assert cfg.hardware_capex == pytest.approx(21000.0)
    assert cfg.peak_hours == (8, 9, 19, 20)
    assert cfg.linebuster_start_date == "2026-12-01"
    assert cfg.tablet_channel_interpretation == "self_service_tablet_rollout"

    # JSON overrides defaults where env not set
    assert cfg.lon == pytest.approx(-101.428765)
    assert cfg.coords_verified is True
    assert cfg.monthly_saas_fee == pytest.approx(1500.0)
    assert cfg.peak_labor_cost_per_hr == pytest.approx(95.0)
    assert cfg.gross_margin_pct == pytest.approx(0.68)


def test_env_none_clears_linebuster_date(tmp_path: Path) -> None:
    assumptions_file = tmp_path / "local_assumptions.json"
    assumptions_file.write_text(
        json.dumps({"linebuster_start_date": "2026-10-01"}),
        encoding="utf-8",
    )
    cfg = load_config(
        assumptions_path=assumptions_file,
        env={"BGA_LINEBUSTER_START_DATE": "none"},
    )
    assert cfg.linebuster_start_date is None


def test_validation_rejects_invalid_values(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="tablet_channel_interpretation"):
        load_config(
            assumptions_path=missing,
            env={"BGA_TABLET_CHANNEL_INTERPRETATION": "invalid_mode"},
        )

    with pytest.raises(ValueError, match="gross_margin_pct"):
        load_config(
            assumptions_path=missing,
            env={"BGA_GROSS_MARGIN_PCT": "1.5"},
        )

    with pytest.raises(ValueError, match="peak_hours"):
        load_config(
            assumptions_path=missing,
            env={"BGA_PEAK_HOURS": "4,8,19"},
        )

    with pytest.raises(ValueError, match="lat"):
        load_config(
            assumptions_path=missing,
            env={"BGA_LAT": "95.0"},
        )
