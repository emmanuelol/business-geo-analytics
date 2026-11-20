"""Repository verification script: gitignore safety, RO schemas, sample parity & smoke tests."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.causal_eval import (
    build_osint_normalized_baseline,
    profile_hourly_bottlenecks,
    simulate_linebuster_scenarios,
)
from src.config import load_config
from src.db_loader import (
    TABLE_DENYLIST,
    RestrictedTableError,
    inspect_schema,
    load_active_branches,
    load_daily_sales,
    load_hourly_sales,
    load_product_mix,
    read_allowlisted_table,
)
from src.generate_sample import EXPECTED_FILES, generate_sample_datasets
from src.osint_collector import load_incidents_file
from src.spatial_models import estimate_lag_effects, join_sales_with_incidents


def _check_gitignore_and_no_db_leak(repo_root: Path) -> None:
    if not (repo_root / ".git").exists():
        image_dirs = ["src", "notebooks", "docs", "scripts", "tests", "data/sample"]
        exts = {".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm"}
        leaked_files = [
            str(p.relative_to(repo_root))
            for sub in image_dirs
            if (repo_root / sub).exists()
            for p in (repo_root / sub).rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ]
        if leaked_files:
            raise RuntimeError(f"Database file leaked into container image: {leaked_files}")
        print("[OK] Container mode (no .git): zero .db/.sqlite files in image paths.")
        return

    targets = [
        "data/raw/kiosk.db",
        "data/raw/master_sales.db",
        "data/raw/local_assumptions.json",
        "config/local.toml",
        "notebooks/01_sales_baseline.html",
        ".env",
    ]
    res = subprocess.run(
        ["git", "check-ignore", "-v", *targets],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    print("[OK] git check-ignore verified:")
    for line in res.stdout.strip().splitlines():
        print(f"  {line}")

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = [
        ln
        for ln in status.stdout.splitlines()
        if any(
            ln.lower().endswith(ext)
            for ext in (".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm")
        )
    ]
    if leaked:
        raise RuntimeError(f"Database file visible in git status: {leaked}")
    print("[OK] Zero .db/.sqlite files visible in git status.")


def _check_config_templates(repo_root: Path) -> None:
    if not (repo_root / ".git").exists():
        default_toml = repo_root / "config" / "default.toml"
        assert default_toml.is_file(), f"Missing {default_toml}"
        load_config(config_path=default_toml, repo_root=repo_root, env={})
        print("[OK] Container config/default.toml verified.")
        return
    default_toml = repo_root / "config" / "default.toml"
    local_toml_tpl = repo_root / "config" / "local.example.toml"
    env_tpl = repo_root / ".env.example"
    json_tpl = repo_root / "local_assumptions.example.json"
    assert default_toml.is_file(), f"Missing {default_toml}"
    assert local_toml_tpl.is_file(), f"Missing {local_toml_tpl}"
    assert env_tpl.is_file(), f"Missing {env_tpl}"
    assert json_tpl.is_file(), f"Missing {json_tpl}"
    load_config(config_path=local_toml_tpl, assumptions_path=json_tpl, env={})
    print("[OK] Config templates (config/default.toml, config/local.example.toml, .env.example, local_assumptions.example.json) verified.")


def _check_notebooks_contract(repo_root: Path) -> None:
    nb_dir = repo_root / "notebooks"
    notebooks = sorted(nb_dir.glob("*.ipynb"))
    assert len(notebooks) == 5, f"Expected 5 notebooks in {nb_dir}, found {len(notebooks)}"
    for nb_path in notebooks:
        nb = nbformat.read(nb_path, as_version=4)
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            assert idx > 0 and nb.cells[idx - 1].cell_type == "markdown", (
                f"{nb_path.name} cell {cell.get('id')} missing preceding Markdown"
            )
            for line in cell.source.splitlines():
                assert not re.match(r"^\s*matplotlib\.use\(", line), (
                    f"{nb_path.name} has unguarded matplotlib.use()"
                )
                assert ".to_string(" not in line, f"{nb_path.name} uses .to_string()"
            if nb_path.name[:2] in {"03", "04", "05"}:
                assert 'sample_dir / "incidents.csv"' not in cell.source, (
                    f"{nb_path.name} hardcodes sample incidents.csv"
                )
    print("[OK] Notebooks 01-05 structural, backend, and mode-isolation contracts verified.")


def _check_readonly_databases(cfg) -> None:
    if cfg.master_sales_db_path.is_file():
        ms_schema = inspect_schema(cfg.master_sales_db_path, db_kind="master_sales")
        print("[OK] master_sales.db allowlisted schema (denylist excluded):")
        for table, meta in ms_schema.items():
            assert table not in TABLE_DENYLIST
            print(f"  - {table}: rows={meta['row_count']}, cols={len(meta['columns'])}")

        daily = load_daily_sales(config=cfg)
        hourly = load_hourly_sales(config=cfg)
        mix = load_product_mix(config=cfg)
        print(
            f"  [loader] daily={len(daily)} ({daily['date'].min().date()}..{daily['date'].max().date()})"
        )
        print(
            f"  [loader] hourly={len(hourly)} ({hourly['date'].min().date()}..{hourly['date'].max().date()}), "
            f"hours={int(hourly['hour_int'].min())}..{int(hourly['hour_int'].max())}"
        )
        print(
            f"  [loader] product_mix={len(mix)} ({mix['date'].min().date()}..{mix['date'].max().date()})"
        )
        _run_smoke_pipeline(daily, hourly, cfg, label="raw")
    else:
        print("[SKIP] data/raw/master_sales.db absent; skipping raw DB inspection.")

    if cfg.kiosk_db_path.is_file():
        ks_schema = inspect_schema(cfg.kiosk_db_path, db_kind="kiosk")
        print("[OK] kiosk.db allowlisted schema (client_vault excluded):")
        for table, meta in ks_schema.items():
            assert table == "active_branches"
            print(f"  - {table}: rows={meta['row_count']}, cols={meta['columns']}")

        branches = load_active_branches(config=cfg)
        print(f"  [loader] active_branches rows={len(branches)}")

        try:
            read_allowlisted_table(cfg.kiosk_db_path, "client_vault", db_kind="kiosk")
        except RestrictedTableError:
            print("[OK] RestrictedTableError confirmed for client_vault.")
        else:
            raise AssertionError("client_vault access was not blocked!")
    else:
        print("[SKIP] data/raw/kiosk.db absent; skipping kiosk DB inspection.")


def _check_sample_generation(cfg) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        written = generate_sample_datasets(output_dir=tmp_dir, seed=42)
        for fname in EXPECTED_FILES:
            gen_df = pd.read_csv(written[fname])
            committed_path = cfg.sample_dir / fname
            assert committed_path.is_file(), f"Missing {committed_path}"
            committed_df = pd.read_csv(committed_path)
            pd.testing.assert_frame_equal(gen_df, committed_df)
    print(f"[OK] Deterministic sample generation verified against {cfg.sample_dir}.")


def _run_smoke_pipeline(daily: pd.DataFrame, hourly: pd.DataFrame, cfg, label: str) -> None:
    incidents = load_incidents_file(cfg.sample_dir / "incidents.csv", relevant_only=True)
    joined = join_sales_with_incidents(
        sales_df=daily,
        incidents_df=incidents,
        store_lat=cfg.lat,
        store_lon=cfg.lon,
        max_distance_m=float(max(cfg.geofence_radii_m)),
    )
    lags = estimate_lag_effects(joined, lags=(0, 1, 2))
    assert len(lags) == 3

    bottleneck = profile_hourly_bottlenecks(
        hourly, peak_hours=cfg.peak_hours, operating_hours=cfg.operating_hours
    )
    assert len(bottleneck["hourly_table"]) == 16

    baseline = build_osint_normalized_baseline(
        daily_df=daily,
        hourly_df=hourly,
        incidents_df=incidents,
        store_lat=cfg.lat,
        store_lon=cfg.lon,
        peak_hours=cfg.peak_hours,
        operating_hours=cfg.operating_hours,
    )
    scenarios = simulate_linebuster_scenarios(baseline=baseline, config=cfg)
    assert len(scenarios) == 3
    print(
        f"[OK] Smoke pipeline ({label}): lags={len(lags)}, scenarios={len(scenarios)}, "
        f"clean_days={baseline['clean_days']}/{baseline['total_days']}"
    )


def main() -> int:
    cfg = load_config(repo_root=REPO_ROOT)
    _check_gitignore_and_no_db_leak(REPO_ROOT)
    _check_config_templates(REPO_ROOT)
    _check_notebooks_contract(REPO_ROOT)
    _check_sample_generation(cfg)

    sample_daily = pd.read_csv(cfg.sample_dir / "daily_sales.csv", parse_dates=["date"])
    sample_hourly = pd.read_csv(cfg.sample_dir / "hourly_sales.csv", parse_dates=["date"])
    _run_smoke_pipeline(sample_daily, sample_hourly, cfg, label="sample")

    _check_readonly_databases(cfg)
    print("[PASS] All repository verification checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
