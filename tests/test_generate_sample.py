"""Tests for src.generate_sample (fully synthetic generator, zero raw reads)."""

from __future__ import annotations

import builtins
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.generate_sample import EXPECTED_FILES, generate_sample_datasets


def test_generator_never_opens_raw_data_or_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_text = Path("src/generate_sample.py").read_text(encoding="utf-8")
    assert "sqlite3" not in src_text
    assert "db_loader" not in src_text
    assert "master_sales.db" not in src_text
    assert "kiosk.db" not in src_text
    assert "data/raw" not in src_text

    orig_open = builtins.open
    orig_path_open = Path.open

    def guarded_open(file, *args, **kwargs):
        path_str = str(file)
        if "data/raw" in path_str or path_str.endswith(".db"):
            raise AssertionError(f"Generator attempted forbidden raw read: {path_str}")
        return orig_open(file, *args, **kwargs)

    def guarded_path_open(self: Path, *args, **kwargs):
        path_str = str(self)
        if "data/raw" in path_str or path_str.endswith(".db"):
            raise AssertionError(f"Generator attempted forbidden raw Path.open: {path_str}")
        return orig_path_open(self, *args, **kwargs)

    def forbidden_sqlite(*args, **kwargs):
        raise AssertionError("Generator must never call sqlite3.connect")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(sqlite3, "connect", forbidden_sqlite)

    written = generate_sample_datasets(output_dir=tmp_path, seed=42)
    assert set(written.keys()) == set(EXPECTED_FILES)


def test_generated_schemas_and_determinism(tmp_path: Path) -> None:
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    generate_sample_datasets(output_dir=dir_a, seed=42)
    generate_sample_datasets(output_dir=dir_b, seed=42)

    for fname in EXPECTED_FILES:
        df_a = pd.read_csv(dir_a / fname)
        df_b = pd.read_csv(dir_b / fname)
        pd.testing.assert_frame_equal(df_a, df_b)
        assert not df_a.empty

    daily = pd.read_csv(dir_a / "daily_sales.csv")
    assert {
        "date",
        "branch_id",
        "revenue",
        "net_sales",
        "iva",
        "tickets",
        "avg_ticket",
        "upt",
        "food_sales",
        "drink_sales",
        "shop_sales",
        "is_post_tablet_change",
    }.issubset(daily.columns)
    assert (daily["branch_id"] == "SAMPLE-001").all()
    assert (daily["revenue"] > 0).all()

    hourly = pd.read_csv(dir_a / "hourly_sales.csv")
    assert set(hourly["hour_int"].unique()) == set(range(6, 22))
    assert (hourly["volume"] >= 0).all()
    assert hourly["is_peak"].dtype == bool

    mix = pd.read_csv(dir_a / "product_mix.csv")
    assert set(mix["category"].unique()) == {
        "Beverages",
        "Food_Pastries",
        "Combos",
        "Retail",
    }

    incidents = pd.read_csv(dir_a / "incidents.csv")
    assert set(incidents["category"].unique()) == {
        "traffic_accident",
        "road_closure",
        "public_safety",
        "weather_disruption",
    }
    assert incidents["lat"].between(20.85, 21.05).all()
    assert incidents["lon"].between(-101.55, -101.30).all()
    assert incidents["severity"].between(0.0, 1.0).all()


def test_committed_sample_files_present_and_valid() -> None:
    sample_dir = Path("data/sample")
    for fname in EXPECTED_FILES:
        path = sample_dir / fname
        assert path.is_file(), f"Missing committed sample CSV: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0
