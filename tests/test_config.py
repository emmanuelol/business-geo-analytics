"""Tests for src.config."""

from __future__ import annotations

import json
import tomllib
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IN_MINIMAL_CONTAINER = not (_REPO_ROOT / ".git").exists() and not (
    _REPO_ROOT / "Dockerfile"
).exists()


def test_defaults_match_confirmed_values(tmp_path: Path) -> None:
    cfg = load_config(
        config_path=tmp_path / "missing.toml",
        assumptions_path=tmp_path / "missing.json",
        env={},
    )

    assert isinstance(cfg, StoreConfig)
    assert cfg.data_mode == "sample"
    assert cfg.fetch_live is False
    assert cfg.osint_start is None
    assert cfg.osint_end is None
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

    cfg = load_config(
        config_path=tmp_path / "missing.toml",
        assumptions_path=assumptions_file,
        env=env,
    )

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
        config_path=tmp_path / "missing.toml",
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
            config_path=tmp_path / "missing.toml",
            assumptions_path=missing,
            env={"BGA_LAT": "95.0"},
        )

    with pytest.raises(ValueError, match="data_mode"):
        load_config(
            config_path=tmp_path / "missing.toml",
            assumptions_path=missing,
            env={"BGA_DATA_MODE": "prod"},
        )


def test_layered_toml_precedence_and_empty_env_ignored(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)

    (cfg_dir / "default.toml").write_text(
        '[runtime]\ndata_mode = "sample"\nfetch_live = false\n\n'
        '[branch]\nlat = 20.95\nlon = -101.42\ncoords_verified = false\n\n'
        '[roi]\nhardware_capex = 12000.0\ngross_margin_pct = 0.65\n',
        encoding="utf-8",
    )
    (tmp_path / "data" / "raw" / "local_assumptions.json").write_text(
        json.dumps({"monthly_saas_fee": 900.0, "hardware_capex": 9999.0}),
        encoding="utf-8",
    )
    local_toml = cfg_dir / "local.toml"
    local_toml.write_text(
        '[runtime]\ndata_mode = "raw"\nfetch_live = true\n'
        'osint_start = "2025-10-16"\nosint_end = "2026-11-19"\n\n'
        '[branch]\nlat = 20.941111\ncoords_verified = true\n\n'
        '[roi]\nhardware_capex = 3500.0\npeak_labor_cost_per_hr = 0.0\n',
        encoding="utf-8",
    )

    cfg = load_config(
        repo_root=tmp_path,
        env={
            "BGA_DATA_MODE": "",
            "BGA_HARDWARE_CAPEX": "4200.0",
        },
    )
    assert cfg.data_mode == "raw"
    assert cfg.fetch_live is True
    assert cfg.osint_start == "2025-10-16"
    assert cfg.osint_end == "2026-11-19"
    assert cfg.lat == pytest.approx(20.941111)
    assert cfg.coords_verified is True
    assert cfg.monthly_saas_fee == pytest.approx(900.0)
    assert cfg.peak_labor_cost_per_hr == pytest.approx(0.0)
    assert cfg.hardware_capex == pytest.approx(4200.0)

    alt_toml = tmp_path / "custom.toml"
    alt_toml.write_text('[runtime]\ndata_mode = "sample"\n', encoding="utf-8")
    cfg_alt = load_config(
        repo_root=tmp_path,
        env={"BGA_CONFIG_PATH": str(alt_toml)},
    )
    assert cfg_alt.data_mode == "sample"


@pytest.mark.skipif(
    _IN_MINIMAL_CONTAINER,
    reason="Example templates are intentionally omitted from the minimal runtime container image",
)
def test_committed_example_templates_parse_cleanly(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_toml = repo_root / "config" / "default.toml"
    local_example_toml = repo_root / "config" / "local.example.toml"
    env_example = repo_root / ".env.example"
    assumptions_example = repo_root / "local_assumptions.example.json"

    assert default_toml.is_file(), "Missing config/default.toml"
    assert local_example_toml.is_file(), "Missing config/local.example.toml"
    assert env_example.is_file(), "Missing .env.example"
    assert assumptions_example.is_file(), "Missing local_assumptions.example.json"

    default_parsed = tomllib.loads(default_toml.read_text(encoding="utf-8"))
    assert default_parsed["runtime"]["data_mode"] == "sample"
    assert default_parsed["runtime"]["fetch_live"] is False
    cfg_default = load_config(
        config_path=tmp_path / "missing.toml",
        assumptions_path=tmp_path / "missing.json",
        env={},
    )
    assert cfg_default.data_mode == "sample"
    assert cfg_default.coords_verified is False

    local_parsed = tomllib.loads(local_example_toml.read_text(encoding="utf-8"))
    assert "runtime" in local_parsed and "branch" in local_parsed and "roi" in local_parsed
    cfg_example = load_config(
        config_path=local_example_toml,
        assumptions_path=tmp_path / "missing.json",
        env={},
    )
    assert cfg_example.branch_id == "TC-001"

    parsed_env: dict[str, str] = {}
    for line in env_example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        parsed_env[k.strip()] = v.strip()

    for required_key in (
        "BGA_DATA_MODE",
        "BGA_LAT",
        "BGA_LON",
        "BGA_COORDS_VERIFIED",
        "BGA_HARDWARE_CAPEX",
        "BGA_MONTHLY_SAAS_FEE",
        "BGA_PEAK_LABOR_COST_PER_HR",
        "BGA_GROSS_MARGIN_PCT",
        "BGA_LINEBUSTER_START_DATE",
    ):
        assert required_key in parsed_env

    cfg_from_env = load_config(assumptions_path=tmp_path / "missing.json", env=parsed_env)
    assert cfg_from_env.coords_verified is False
    assert cfg_from_env.linebuster_start_date is None

    payload = json.loads(assumptions_example.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["coords_verified"] is False
    assert payload["linebuster_start_date"] is None
    cfg_from_json = load_config(assumptions_path=assumptions_example, env={})
    assert cfg_from_json.hardware_capex == pytest.approx(float(payload["hardware_capex"]))
    assert cfg_from_json.gross_margin_pct == pytest.approx(float(payload["gross_margin_pct"]))
