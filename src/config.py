"""Store configuration, spatial geofences, operating hours, and ROI assumptions."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DATA_MODE = "sample"
VALID_DATA_MODES = ("sample", "raw")
DEFAULT_BRANCH_ID = "TC-001"
# Approximate Silao centroid; replace locally via local_assumptions.json or BGA_LAT/BGA_LON
DEFAULT_LAT = 20.95
DEFAULT_LON = -101.42
COORDS_VERIFIED_DEFAULT = False

GEOFENCE_RADII_M: tuple[int, ...] = (500, 1500, 5000)
OPERATING_HOUR_MIN = 6
OPERATING_HOUR_MAX = 21
OPERATING_HOURS_RANGE = range(OPERATING_HOUR_MIN, OPERATING_HOUR_MAX + 1)
PEAK_HOURS_DEFAULT: tuple[int, ...] = (8, 9, 18, 19, 20)

LINEBUSTER_START_DATE: str | None = None
TABLET_CHANNEL_START_DATE = "2026-09-05"
TABLET_CHANNEL_INTERPRETATION = "pos_channel_change"
VALID_TABLET_INTERPRETATIONS = (
    "pos_channel_change",
    "self_service_tablet_rollout",
)
ITEMS_COVERAGE_END_DATE = "2026-08-20"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y"}:
            return True
        if norm in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean for {field_name}: {value!r}")


def _parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def _parse_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        items = [int(x) for x in value]
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError(f"Invalid sequence for {field_name}: {value!r}")
            items = [int(x) for x in parsed]
        else:
            items = [int(part.strip()) for part in stripped.split(",") if part.strip()]
    else:
        raise ValueError(f"Invalid sequence for {field_name}: {value!r}")
    return tuple(items)


@dataclass
class StoreConfig:
    """Configuration for a retail storefront, spatial buffers, and ROI simulator."""

    data_mode: str = DEFAULT_DATA_MODE
    fetch_live: bool = False
    osint_start: str | None = None
    osint_end: str | None = None

    branch_id: str = DEFAULT_BRANCH_ID
    store_name: str = "The Coffee - Paseo MS"
    city: str = "Silao, Guanajuato, Mexico"
    address: str = "Libramiento Nte. 3365, El Progreso, Silao, Guanajuato"
    lat: float = DEFAULT_LAT
    lon: float = DEFAULT_LON
    coords_verified: bool = COORDS_VERIFIED_DEFAULT
    geofence_radii_m: tuple[int, ...] = GEOFENCE_RADII_M
    operating_hours: tuple[int, int] = (OPERATING_HOUR_MIN, OPERATING_HOUR_MAX)
    peak_hours: tuple[int, ...] = PEAK_HOURS_DEFAULT
    linebuster_start_date: str | None = LINEBUSTER_START_DATE
    tablet_channel_start_date: str = TABLET_CHANNEL_START_DATE
    tablet_channel_interpretation: str = TABLET_CHANNEL_INTERPRETATION
    items_coverage_end_date: str = ITEMS_COVERAGE_END_DATE

    # Configurable ROI placeholder assumptions (override via local_assumptions.json or BGA_*)
    hardware_capex: float = 12000.0
    monthly_saas_fee: float = 1200.0
    peak_labor_cost_per_hr: float = 85.0
    gross_margin_pct: float = 0.65
    currency: str = "MXN"

    repo_root: Path = field(default_factory=_default_repo_root)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        self.data_mode = str(self.data_mode).strip().lower()
        if self.data_mode not in VALID_DATA_MODES:
            raise ValueError(
                f"data_mode must be one of {VALID_DATA_MODES}, got {self.data_mode!r}"
            )
        self.fetch_live = _parse_bool(self.fetch_live, "fetch_live")
        self.osint_start = _parse_optional_str(self.osint_start)
        self.osint_end = _parse_optional_str(self.osint_end)
        self.lat = float(self.lat)
        self.lon = float(self.lon)
        self.coords_verified = bool(self.coords_verified)
        self.hardware_capex = float(self.hardware_capex)
        self.monthly_saas_fee = float(self.monthly_saas_fee)
        self.peak_labor_cost_per_hr = float(self.peak_labor_cost_per_hr)
        self.gross_margin_pct = float(self.gross_margin_pct)

        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"lat must be within [-90, 90], got {self.lat}")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"lon must be within [-180, 180], got {self.lon}")

        if len(self.operating_hours) != 2:
            raise ValueError(f"operating_hours must be (start, end), got {self.operating_hours}")
        op_start, op_end = int(self.operating_hours[0]), int(self.operating_hours[1])
        if not (0 <= op_start <= op_end <= 23):
            raise ValueError(f"operating_hours out of bounds 0..23: {(op_start, op_end)}")
        self.operating_hours = (op_start, op_end)

        if not self.geofence_radii_m:
            raise ValueError("geofence_radii_m cannot be empty")
        radii = tuple(sorted({int(r) for r in self.geofence_radii_m}))
        if any(r <= 0 for r in radii):
            raise ValueError(f"geofence_radii_m must be positive, got {radii}")
        self.geofence_radii_m = radii

        if not self.peak_hours:
            raise ValueError("peak_hours cannot be empty")
        peaks = tuple(sorted({int(h) for h in self.peak_hours}))
        if any(h < op_start or h > op_end for h in peaks):
            raise ValueError(
                f"peak_hours {peaks} must fall within operating_hours {self.operating_hours}"
            )
        self.peak_hours = peaks

        if self.tablet_channel_interpretation not in VALID_TABLET_INTERPRETATIONS:
            raise ValueError(
                "tablet_channel_interpretation must be one of "
                f"{VALID_TABLET_INTERPRETATIONS}, got {self.tablet_channel_interpretation!r}"
            )

        if self.hardware_capex < 0 or self.monthly_saas_fee < 0 or self.peak_labor_cost_per_hr < 0:
            raise ValueError("ROI cost parameters must be non-negative")
        if not (0.0 < self.gross_margin_pct <= 1.0):
            raise ValueError(
                f"gross_margin_pct must be in (0, 1], got {self.gross_margin_pct}"
            )

    @property
    def raw_dir(self) -> Path:
        return self.repo_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.repo_root / "data" / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.repo_root / "data" / "sample"

    @property
    def master_sales_db_path(self) -> Path:
        return self.raw_dir / "master_sales.db"

    @property
    def kiosk_db_path(self) -> Path:
        return self.raw_dir / "kiosk.db"

    @property
    def local_assumptions_path(self) -> Path:
        return self.raw_dir / "local_assumptions.json"

    @property
    def default_config_path(self) -> Path:
        return self.repo_root / "config" / "default.toml"

    @property
    def local_config_path(self) -> Path:
        return self.repo_root / "config" / "local.toml"

    def operating_hour_range(self) -> range:
        """Return inclusive integer hour range (e.g. 6..21 -> range(6, 22))."""
        return range(self.operating_hours[0], self.operating_hours[1] + 1)

    def to_dict(self, include_paths: bool = False) -> dict[str, Any]:
        """Serialize configuration to plain dict for reporting and diagnostics."""
        data = asdict(self)
        if include_paths:
            data["repo_root"] = str(self.repo_root)
            data["raw_dir"] = str(self.raw_dir)
            data["processed_dir"] = str(self.processed_dir)
            data["sample_dir"] = str(self.sample_dir)
        else:
            data.pop("repo_root", None)
        return data


_FIELD_PARSERS: dict[str, Any] = {
    "data_mode": lambda v: str(v).strip().lower(),
    "fetch_live": lambda v: _parse_bool(v, "fetch_live"),
    "osint_start": _parse_optional_str,
    "osint_end": _parse_optional_str,
    "branch_id": str,
    "store_name": str,
    "city": str,
    "address": str,
    "lat": float,
    "lon": float,
    "coords_verified": lambda v: _parse_bool(v, "coords_verified"),
    "geofence_radii_m": lambda v: _parse_int_tuple(v, "geofence_radii_m"),
    "operating_hours": lambda v: _parse_int_tuple(v, "operating_hours"),
    "peak_hours": lambda v: _parse_int_tuple(v, "peak_hours"),
    "linebuster_start_date": _parse_optional_str,
    "tablet_channel_start_date": str,
    "tablet_channel_interpretation": str,
    "items_coverage_end_date": str,
    "hardware_capex": float,
    "monthly_saas_fee": float,
    "peak_labor_cost_per_hr": float,
    "gross_margin_pct": float,
    "currency": str,
}


def _apply_overrides(values: dict[str, Any], raw_overrides: Mapping[str, Any]) -> None:
    for key, raw_val in raw_overrides.items():
        norm_key = key.strip().lower()
        parser = _FIELD_PARSERS.get(norm_key)
        if parser is not None:
            values[norm_key] = parser(raw_val)


def _load_toml_flat(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    flat: dict[str, Any] = {}
    for key, val in payload.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val
    return flat


def load_config(
    assumptions_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    repo_root: Path | str | None = None,
    config_path: Path | str | None = None,
) -> StoreConfig:
    """Load StoreConfig: defaults < default.toml < local_assumptions.json < local.toml < env BGA_*."""
    env_map = os.environ if env is None else env
    resolved_root = (
        Path(repo_root)
        if repo_root is not None
        else Path(env_map["BGA_REPO_ROOT"])
        if env_map.get("BGA_REPO_ROOT", "").strip()
        else _default_repo_root()
    )

    values = StoreConfig(repo_root=resolved_root).to_dict(include_paths=False)

    default_toml = resolved_root / "config" / "default.toml"
    if default_toml.is_file():
        _apply_overrides(values, _load_toml_flat(default_toml))

    json_path = (
        Path(assumptions_path)
        if assumptions_path is not None
        else resolved_root / "data" / "raw" / "local_assumptions.json"
    )
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {json_path}")
        _apply_overrides(values, payload)

    env_cfg = env_map.get("BGA_CONFIG_PATH", "").strip()
    local_toml = (
        Path(config_path)
        if config_path is not None
        else Path(env_cfg)
        if env_cfg
        else resolved_root / "config" / "local.toml"
    )
    if local_toml.is_file():
        _apply_overrides(values, _load_toml_flat(local_toml))

    env_overrides = {
        k[len("BGA_") :].lower(): v
        for k, v in env_map.items()
        if k.startswith("BGA_")
        and k not in {"BGA_REPO_ROOT", "BGA_CONFIG_PATH"}
        and str(v).strip() != ""
    }
    _apply_overrides(values, env_overrides)

    return StoreConfig(repo_root=resolved_root, **values)
