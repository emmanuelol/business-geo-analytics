"""Tests for src.db_loader."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.db_loader import (
    KIOSK_ALLOWLIST,
    MASTER_ALLOWLIST,
    TABLE_DENYLIST,
    RestrictedTableError,
    categorize_sku,
    connect_readonly,
    inspect_schema,
    load_active_branches,
    load_contextual_split,
    load_daily_sales,
    load_hourly_sales,
    load_product_mix,
    load_sales_channels,
    read_allowlisted_table,
)


def test_readonly_connection_rejects_writes(tmp_master_db: Path) -> None:
    conn = connect_readonly(tmp_master_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM Summary")
    finally:
        conn.close()


def test_security_allowlist_and_denylist_enforced(
    tmp_master_db: Path, tmp_kiosk_db: Path
) -> None:
    master_schema = inspect_schema(tmp_master_db, db_kind="master_sales")
    assert set(master_schema.keys()) == set(MASTER_ALLOWLIST)
    for denied in TABLE_DENYLIST:
        assert denied not in master_schema
        with pytest.raises(RestrictedTableError):
            read_allowlisted_table(tmp_master_db, denied, db_kind="master_sales")

    kiosk_schema = inspect_schema(tmp_kiosk_db, db_kind="kiosk")
    assert set(kiosk_schema.keys()) == set(KIOSK_ALLOWLIST)
    assert "client_vault" not in kiosk_schema
    assert "billing_logs" not in kiosk_schema
    assert "branch_sheet_id" not in kiosk_schema["active_branches"]["columns"]

    for blocked in ("client_vault", "billing_logs", "Unknown_Table"):
        with pytest.raises(RestrictedTableError):
            read_allowlisted_table(tmp_kiosk_db, blocked, db_kind="kiosk")

    branches = load_active_branches(db_path=tmp_kiosk_db)
    assert list(branches["branch_id"]) == ["TC-001"]
    assert "branch_sheet_id" not in branches.columns
    assert "bot_status" not in branches.columns


def test_load_daily_sales_coalesces_tc001_and_default(tmp_master_db: Path) -> None:
    df = load_daily_sales(branch_id="TC-001", db_path=tmp_master_db)

    assert len(df) == 3
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == [
        "2026-09-04",
        "2026-09-05",
        "2026-09-06",
    ]
    # 2026-09-04 coalesced blank TC-001 iva/food/drink/card_sales from DEFAULT
    row0 = df.iloc[0]
    assert row0["iva"] == pytest.approx(551.72)
    assert row0["card_sales"] == pytest.approx(2500.0)
    assert row0["food_sales"] == pytest.approx(1000.0)
    assert row0["drink_sales"] == pytest.approx(2800.0)
    assert row0["tickets"] == pytest.approx(25.0)
    assert row0["is_post_tablet_change"] is False or bool(row0["is_post_tablet_change"]) is False

    # 2026-09-05 and 2026-09-06 use sales_food + (sales_hot_drinks + sales_cold_drinks)
    row1 = df.iloc[1]
    assert row1["food_sales"] == pytest.approx(1100.0)
    assert row1["drink_sales"] == pytest.approx(3300.0)
    assert bool(row1["is_post_tablet_change"]) is True


def test_load_hourly_sales_clamps_6_to_21_and_coalesces(tmp_master_db: Path) -> None:
    df = load_hourly_sales(branch_id="TC-001", db_path=tmp_master_db)

    # Hours 4, 5, 22 clamped out; 2026-09-04 has hours 8 (TC-001), 9 (DEFAULT), 19 (TC-001)
    assert set(df["hour_int"].unique()).issubset(set(range(6, 22)))
    assert 4 not in df["hour_int"].values
    assert 5 not in df["hour_int"].values
    assert 22 not in df["hour_int"].values
    assert len(df) == 4
    assert "volume" in df.columns
    assert "is_peak" in df.columns

    grid = load_hourly_sales(
        branch_id="TC-001", db_path=tmp_master_db, complete_grid=True
    )
    # 2 distinct dates * 16 operating hours (6..21) = 32 rows
    assert len(grid) == 32
    assert grid["hour_int"].min() == 6
    assert grid["hour_int"].max() == 21


def test_load_product_mix_categories_and_mojibake(tmp_master_db: Path) -> None:
    df = load_product_mix(branch_id="TC-001", db_path=tmp_master_db)
    by_name = dict(zip(df["product_name"], df["category"]))

    assert by_name["Americano"] == "Beverages"
    assert by_name["Gelato Frappé Coffee"] == "Beverages"
    assert by_name["Cold Drip V60 + Milk"] == "Beverages"
    assert by_name["Croissant Serrano & Gouda"] == "Food_Pastries"
    assert by_name["Misto + Tomate Cereja"] == "Food_Pastries"
    assert by_name["Iced Latte +  Ham & Cheese Sandwich"] == "Combos"
    assert by_name["White bag 250g"] == "Retail"
    assert by_name["Tumbler (240ml)"] == "Retail"

    assert categorize_sku("Coffee bag 250g + Tumbler 240ml") == "Combos"
    assert categorize_sku("Kit Café White 1kg") == "Retail"
    assert categorize_sku("Brownie com sorvete") == "Food_Pastries"
    assert categorize_sku("Sora Iced Latte (Signature)") == "Beverages"


def test_auxiliary_loaders_contextual_and_channels(tmp_master_db: Path) -> None:
    split_df = load_contextual_split(branch_id="TC-001", db_path=tmp_master_db)
    assert set(split_df["category"]) == {"Drinks", "Foods"}

    ch_df = load_sales_channels(branch_id="TC-001", db_path=tmp_master_db)
    assert set(ch_df["channel"]) == {"Tablet", "Smartphone"}
    assert pd.api.types.is_datetime64_any_dtype(ch_df["date"])


def test_wal_database_in_readonly_mount_succeeds(tmp_path: Path) -> None:
    import os
    import shutil

    ro_dir = tmp_path / "ro_mount"
    ro_dir.mkdir()
    db_file = ro_dir / "wal_sales.db"

    w_conn = sqlite3.connect(db_file)
    w_conn.execute("PRAGMA journal_mode=WAL")
    w_conn.execute("CREATE TABLE Summary (branch_id TEXT, date TEXT, revenue TEXT)")
    w_conn.execute("INSERT INTO Summary VALUES ('TC-001', '2026-10-01', '4000')")
    w_conn.commit()
    w_conn.close()

    # Simulate Docker :ro bind mount without -shm/-wal sidecar files
    for sidecar in (ro_dir / "wal_sales.db-shm", ro_dir / "wal_sales.db-wal"):
        if sidecar.exists():
            sidecar.unlink()
    os.chmod(db_file, 0o444)
    os.chmod(ro_dir, 0o555)
    try:
        schema = inspect_schema(db_file, db_kind="master_sales")
        assert schema["Summary"]["row_count"] == 1
    finally:
        os.chmod(ro_dir, 0o755)
        os.chmod(db_file, 0o644)
        shutil.rmtree(ro_dir, ignore_errors=True)


def test_read_allowlisted_table_validates_columns(
    tmp_master_db: Path, tmp_kiosk_db: Path
) -> None:
    with pytest.raises(ValueError, match="empty"):
        read_allowlisted_table(
            tmp_kiosk_db, "active_branches", db_kind="kiosk", columns=[]
        )

    with pytest.raises(RestrictedTableError):
        read_allowlisted_table(
            tmp_kiosk_db,
            "active_branches",
            db_kind="kiosk",
            columns=["branch_sheet_id"],
        )

    safe_df = read_allowlisted_table(
        tmp_kiosk_db,
        "active_branches",
        db_kind="kiosk",
        columns=["branch_id", "franchise_name"],
    )
    assert list(safe_df.columns) == ["branch_id", "franchise_name"]

    with pytest.raises(ValueError, match="empty"):
        read_allowlisted_table(
            tmp_master_db, "Summary", db_kind="master_sales", columns=[]
        )

    with pytest.raises(ValueError, match="Unknown column"):
        read_allowlisted_table(
            tmp_master_db,
            "Summary",
            db_kind="master_sales",
            columns=['revenue" FROM Summary; --'],
        )

    cols_df = read_allowlisted_table(
        tmp_master_db,
        "Summary",
        db_kind="master_sales",
        columns=["branch_id", "date", "revenue"],
    )
    assert list(cols_df.columns) == ["branch_id", "date", "revenue"]


def test_connect_readonly_handles_special_uri_chars_in_path(tmp_path: Path) -> None:
    db_file = tmp_path / "sales #1?v=2.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE Summary (branch_id TEXT, date TEXT, revenue TEXT)")
    conn.execute("INSERT INTO Summary VALUES ('TC-001', '2026-10-01', '4000')")
    conn.commit()
    conn.close()

    schema = inspect_schema(db_file, db_kind="master_sales")
    assert schema["Summary"]["row_count"] == 1
