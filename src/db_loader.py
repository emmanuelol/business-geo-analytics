"""Read-only SQLite schema inspector and POS DataFrame loaders with strict table allowlisting."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import StoreConfig, load_config

MASTER_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Summary",
        "Items",
        "Hourly_Sales",
        "Master_Financials",
        "Contextual_Split",
        "Sales_Channels",
        "Day_of_Week_Sales",
    }
)

KIOSK_ALLOWLIST: frozenset[str] = frozenset({"active_branches"})

ACTIVE_BRANCHES_SAFE_COLUMNS: tuple[str, ...] = (
    "id",
    "branch_id",
    "franchise_name",
    "location",
    "is_active",
)

TABLE_DENYLIST: frozenset[str] = frozenset(
    {
        "client_vault",
        "billing_logs",
        "Billing_Logs",
        "Team_Directory",
        "Active_Team",
        "Horarios",
        "Google_Reviews",
        "Bank_Transactions",
        "Expenses",
        "Loans_Schedule",
        "Needs_Review",
        "Tips_Log",
        "App_Clients",
    }
)
_DENYLIST_LOWER: frozenset[str] = frozenset(t.lower() for t in TABLE_DENYLIST)

_ALLOWLISTS: dict[str, frozenset[str]] = {
    "master_sales": MASTER_ALLOWLIST,
    "kiosk": KIOSK_ALLOWLIST,
}

_FOOD_PLUS_NON_COMBOS = ("misto + tomate cereja",)
_BEVERAGE_PLUS_NON_COMBOS = ("cold drip v60 + milk",)
_RETAIL_KEYWORDS = (
    "250g",
    "1kg",
    "tumbler",
    "dripper",
    "mug",
    "totebag",
    "kit café",
    "kit cafes",
    "kit cafe",
)
_FOOD_KEYWORDS = (
    "cake",
    "brownie",
    "cookie",
    "croissant",
    "sandwich",
    "roll",
    "pain au chocolat",
    "pan de muerto",
    "gingerbread",
    "queijo",
    "misto",
    "kouign",
)


class RestrictedTableError(PermissionError):
    """Raised when code attempts to access a blocked or non-allowlisted SQLite table."""


def connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection strictly in read-only URI mode (mode=ro)."""
    resolved = Path(db_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    conn = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    try:
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        return conn
    except sqlite3.OperationalError as exc:
        conn.close()
        if "readonly database" not in str(exc).lower():
            raise
        # WAL databases on read-only mounts (:ro) lack writable -shm; immutable=1 allows read-only access
        return sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro&immutable=1", uri=True)


def _validate_table_access(table_name: str, db_kind: str) -> str:
    if db_kind not in _ALLOWLISTS:
        raise ValueError(f"Unknown db_kind {db_kind!r}; expected one of {tuple(_ALLOWLISTS)}")
    if table_name.lower() in _DENYLIST_LOWER:
        raise RestrictedTableError(f"Access denied to restricted table: {table_name!r}")
    allowlist = _ALLOWLISTS[db_kind]
    if table_name not in allowlist:
        raise RestrictedTableError(
            f"Table {table_name!r} is not in allowlist for {db_kind!r}"
        )
    return table_name


def inspect_schema(
    db_path: Path | str, db_kind: str = "master_sales"
) -> dict[str, dict[str, Any]]:
    """Inspect allowlisted table schemas and row counts without querying denylisted tables."""
    allowlist = _ALLOWLISTS.get(db_kind)
    if allowlist is None:
        raise ValueError(f"Unknown db_kind {db_kind!r}")

    conn = connect_readonly(db_path)
    try:
        present = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        }
        schema: dict[str, dict[str, Any]] = {}
        for table in sorted(present & allowlist):
            if table.lower() in _DENYLIST_LOWER:
                continue
            cols = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            if db_kind == "kiosk" and table == "active_branches":
                cols = [c for c in cols if c in ACTIVE_BRANCHES_SAFE_COLUMNS]
            row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            schema[table] = {"columns": cols, "row_count": row_count}
        return schema
    finally:
        conn.close()


def read_allowlisted_table(
    db_path: Path | str,
    table_name: str,
    db_kind: str = "master_sales",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read an allowlisted SQLite table in read-only mode."""
    validated = _validate_table_access(table_name, db_kind)
    conn = connect_readonly(db_path)
    try:
        if db_kind == "kiosk" and validated == "active_branches":
            allowed_cols = ACTIVE_BRANCHES_SAFE_COLUMNS
            selected = (
                [c for c in columns if c in allowed_cols]
                if columns is not None
                else list(allowed_cols)
            )
            col_sql = ", ".join(f'"{c}"' for c in selected)
            return pd.read_sql_query(f'SELECT {col_sql} FROM "{validated}"', conn)

        if columns is not None:
            col_sql = ", ".join(f'"{c}"' for c in columns)
            return pd.read_sql_query(f'SELECT {col_sql} FROM "{validated}"', conn)
        return pd.read_sql_query(f'SELECT * FROM "{validated}"', conn)
    finally:
        conn.close()


def _clean_blank_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            stripped = out[col].astype("string").str.strip()
            out[col] = stripped.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    return out


def _coalesce_branch_rows(
    df: pd.DataFrame, branch_id: str, key_cols: Sequence[str]
) -> pd.DataFrame:
    """Prefer branch_id rows over DEFAULT rows and coalesce missing fields per key."""
    if df.empty:
        return df.copy()
    cleaned = _clean_blank_strings(df)
    scoped = cleaned[cleaned["branch_id"].isin(["DEFAULT", branch_id])].copy()
    if scoped.empty:
        return scoped
    scoped["_priority"] = np.where(scoped["branch_id"] == branch_id, 1, 0)
    scoped = scoped.sort_values(list(key_cols) + ["_priority"], kind="mergesort")
    coalesced = scoped.groupby(list(key_cols), as_index=False).last()
    coalesced = coalesced.drop(columns=["_priority"])
    coalesced["branch_id"] = branch_id
    return coalesced


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _fix_mojibake(text: str) -> str:
    if "Ã" in text or "Â" in text:
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def categorize_sku(item_name: str) -> str:
    """Categorize a menu SKU into Beverages, Food_Pastries, Combos, or Retail."""
    norm = re.sub(r"\s+", " ", _fix_mojibake(str(item_name)).strip().lower())
    if norm in _FOOD_PLUS_NON_COMBOS:
        return "Food_Pastries"
    if norm in _BEVERAGE_PLUS_NON_COMBOS:
        return "Beverages"
    if "+" in norm:
        return "Combos"
    if any(kw in norm for kw in _RETAIL_KEYWORDS):
        return "Retail"
    if any(kw in norm for kw in _FOOD_KEYWORDS):
        return "Food_Pastries"
    return "Beverages"


def load_daily_sales(
    branch_id: str | None = None,
    db_path: Path | str | None = None,
    config: StoreConfig | None = None,
) -> pd.DataFrame:
    """Load normalized daily sales by coalescing Summary and Master_Financials."""
    cfg = config or load_config()
    target_branch = branch_id or cfg.branch_id
    target_db = db_path or cfg.master_sales_db_path

    summary_raw = read_allowlisted_table(target_db, "Summary", db_kind="master_sales")
    summary = _coalesce_branch_rows(summary_raw, target_branch, ["date"])

    mf_raw = read_allowlisted_table(target_db, "Master_Financials", db_kind="master_sales")
    mf = _coalesce_branch_rows(mf_raw, target_branch, ["date"])

    summary_norm = pd.DataFrame({"date": summary["date"]}) if not summary.empty else pd.DataFrame(columns=["date"])
    if not summary.empty:
        summary_norm["revenue"] = _to_numeric(summary["revenue"])
        summary_norm["net_sales"] = _to_numeric(summary["total"])
        summary_norm["iva"] = _to_numeric(summary["iva"])
        summary_norm["tickets"] = _to_numeric(summary["ordenes"])
        summary_norm["avg_ticket"] = _to_numeric(summary["ticket promedio"])
        summary_norm["upt"] = _to_numeric(summary["upt"])

    mf_cols = [
        "gross",
        "net_receivable",
        "food",
        "drink",
        "shop",
        "gross_sales_local",
        "net_revenue_local",
        "card_sales",
        "delivery_sales",
        "app_sales",
        "gross_cash_sales",
        "total_tips",
        "net_cash_deposit",
        "total_tickets",
        "average_ticket",
        "upt",
        "sales_cold_drinks",
        "sales_hot_drinks",
        "sales_food",
    ]
    mf_norm = pd.DataFrame({"date": mf["date"]}) if not mf.empty else pd.DataFrame(columns=["date"])
    if not mf.empty:
        for col in mf_cols:
            mf_norm[f"mf_{col}"] = _to_numeric(mf[col])

    merged = pd.merge(summary_norm, mf_norm, on="date", how="outer")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    merged["branch_id"] = target_branch

    merged["revenue"] = merged["revenue"].combine_first(
        merged.get("mf_net_receivable", pd.Series(dtype=float))
    ).combine_first(merged.get("mf_gross_sales_local", pd.Series(dtype=float)))
    merged["net_sales"] = merged["net_sales"].combine_first(
        merged.get("mf_net_revenue_local", pd.Series(dtype=float))
    )
    merged["tickets"] = merged["tickets"].combine_first(
        merged.get("mf_total_tickets", pd.Series(dtype=float))
    )
    merged["avg_ticket"] = merged["avg_ticket"].combine_first(
        merged.get("mf_average_ticket", pd.Series(dtype=float))
    )
    merged["upt"] = merged["upt"].combine_first(
        merged.get("mf_upt", pd.Series(dtype=float))
    )

    gross_ref = merged.get("mf_gross_sales_local", pd.Series(np.nan, index=merged.index))
    raw_sf = merged.get("mf_sales_food", pd.Series(np.nan, index=merged.index))
    valid_sf = raw_sf.where(gross_ref.isna() | (raw_sf <= gross_ref * 1.05))

    cold = merged.get("mf_sales_cold_drinks", pd.Series(np.nan, index=merged.index))
    hot = merged.get("mf_sales_hot_drinks", pd.Series(np.nan, index=merged.index))
    both_drinks = cold.fillna(0.0) + hot.fillna(0.0)
    has_drinks = cold.notna() | hot.notna()
    raw_sd = both_drinks.where(has_drinks)
    valid_sd = raw_sd.where(gross_ref.isna() | (raw_sd <= gross_ref * 1.05))

    merged["food_sales"] = merged.get("mf_food", pd.Series(np.nan, index=merged.index)).combine_first(valid_sf)
    merged["drink_sales"] = merged.get("mf_drink", pd.Series(np.nan, index=merged.index)).combine_first(valid_sd)
    merged["shop_sales"] = merged.get("mf_shop", pd.Series(np.nan, index=merged.index))

    for field in (
        "gross_sales_local",
        "net_revenue_local",
        "card_sales",
        "delivery_sales",
        "app_sales",
        "gross_cash_sales",
        "total_tips",
        "net_cash_deposit",
    ):
        src_col = f"mf_{field}"
        merged[field] = merged[src_col] if src_col in merged.columns else np.nan

    merged["is_post_tablet_change"] = merged["date"] >= pd.Timestamp(
        cfg.tablet_channel_start_date
    )

    ordered_cols = [
        "date",
        "branch_id",
        "revenue",
        "net_sales",
        "iva",
        "tickets",
        "avg_ticket",
        "upt",
        "gross_sales_local",
        "net_revenue_local",
        "card_sales",
        "delivery_sales",
        "app_sales",
        "gross_cash_sales",
        "total_tips",
        "net_cash_deposit",
        "food_sales",
        "drink_sales",
        "shop_sales",
        "is_post_tablet_change",
    ]
    return merged[ordered_cols]


def load_hourly_sales(
    branch_id: str | None = None,
    db_path: Path | str | None = None,
    config: StoreConfig | None = None,
    clamp_hours: tuple[int, int] | None = None,
    complete_grid: bool = False,
) -> pd.DataFrame:
    """Load hourly revenue clamped to operating hours (default 6..21)."""
    cfg = config or load_config()
    target_branch = branch_id or cfg.branch_id
    target_db = db_path or cfg.master_sales_db_path
    min_h, max_h = clamp_hours or cfg.operating_hours

    raw = read_allowlisted_table(target_db, "Hourly_Sales", db_kind="master_sales")
    coalesced = _coalesce_branch_rows(raw, target_branch, ["date", "hour_int"])
    if coalesced.empty:
        return pd.DataFrame(
            columns=["date", "branch_id", "hour_int", "hour_label", "volume", "is_peak"]
        )

    coalesced["date"] = pd.to_datetime(coalesced["date"], errors="coerce")
    extracted = coalesced["hour_int"].astype("string").str.extract(r"(\d+)")[0]
    coalesced["hour_int"] = pd.to_numeric(extracted, errors="coerce")
    coalesced["volume"] = _to_numeric(coalesced["volume"]).fillna(0.0)

    df = coalesced.dropna(subset=["date", "hour_int"]).copy()
    df["hour_int"] = df["hour_int"].astype(int)
    df = df[(df["hour_int"] >= min_h) & (df["hour_int"] <= max_h)].copy()

    if complete_grid and not df.empty:
        dates = sorted(df["date"].unique())
        hours = list(range(min_h, max_h + 1))
        idx = pd.MultiIndex.from_product([dates, hours], names=["date", "hour_int"])
        df = (
            df.groupby(["date", "hour_int"], as_index=True)["volume"]
            .sum()
            .reindex(idx, fill_value=0.0)
            .reset_index()
        )
        df["branch_id"] = target_branch
    else:
        df = df.sort_values(["date", "hour_int"]).reset_index(drop=True)

    df["hour_label"] = df["hour_int"].map(lambda h: f"{int(h):02d}:00")
    df["is_peak"] = df["hour_int"].isin(cfg.peak_hours)
    return df[["date", "branch_id", "hour_int", "hour_label", "volume", "is_peak"]]


def load_product_mix(
    branch_id: str | None = None,
    db_path: Path | str | None = None,
    config: StoreConfig | None = None,
) -> pd.DataFrame:
    """Load item-level sales and classify SKUs into 4 retail categories."""
    cfg = config or load_config()
    target_branch = branch_id or cfg.branch_id
    target_db = db_path or cfg.master_sales_db_path

    raw = read_allowlisted_table(target_db, "Items", db_kind="master_sales")
    scoped = _clean_blank_strings(raw)
    scoped = scoped[scoped["branch_id"].isin([target_branch, "DEFAULT"])].copy()
    if scoped.empty:
        return pd.DataFrame(
            columns=["date", "branch_id", "product_name", "quantity_sold", "category"]
        )

    scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce")
    scoped["product_name"] = (
        scoped["item"].astype("string").fillna("").map(_fix_mojibake).str.strip()
    )
    scoped["quantity_sold"] = _to_numeric(scoped["qty"]).fillna(0.0)
    scoped = scoped.dropna(subset=["date"])
    scoped = scoped[scoped["product_name"] != ""].copy()
    scoped["category"] = scoped["product_name"].map(categorize_sku)
    scoped["branch_id"] = target_branch

    grouped = (
        scoped.groupby(
            ["date", "branch_id", "product_name", "category"], as_index=False
        )["quantity_sold"]
        .sum()
        .sort_values(["date", "product_name"])
        .reset_index(drop=True)
    )
    return grouped


def load_contextual_split(
    branch_id: str | None = None,
    db_path: Path | str | None = None,
    config: StoreConfig | None = None,
) -> pd.DataFrame:
    """Load Contextual_Split category revenues (post-2026-09-05 channel audit table)."""
    cfg = config or load_config()
    target_branch = branch_id or cfg.branch_id
    target_db = db_path or cfg.master_sales_db_path

    raw = read_allowlisted_table(target_db, "Contextual_Split", db_kind="master_sales")
    coalesced = _coalesce_branch_rows(raw, target_branch, ["date", "category"])
    if coalesced.empty:
        return pd.DataFrame(columns=["date", "branch_id", "category", "revenue"])
    coalesced["date"] = pd.to_datetime(coalesced["date"], errors="coerce")
    coalesced["revenue"] = _to_numeric(coalesced["revenue"]).fillna(0.0)
    return (
        coalesced.dropna(subset=["date"])
        .sort_values(["date", "category"])
        .reset_index(drop=True)[["date", "branch_id", "category", "revenue"]]
    )


def load_sales_channels(
    branch_id: str | None = None,
    db_path: Path | str | None = None,
    config: StoreConfig | None = None,
) -> pd.DataFrame:
    """Load Sales_Channels breakdown (Tablet vs Smartphone segmentation)."""
    cfg = config or load_config()
    target_branch = branch_id or cfg.branch_id
    target_db = db_path or cfg.master_sales_db_path

    raw = read_allowlisted_table(target_db, "Sales_Channels", db_kind="master_sales")
    coalesced = _coalesce_branch_rows(raw, target_branch, ["date", "channel"])
    if coalesced.empty:
        return pd.DataFrame(columns=["date", "branch_id", "channel", "revenue", "orders"])
    coalesced["date"] = pd.to_datetime(coalesced["date"], errors="coerce")
    coalesced["revenue"] = _to_numeric(coalesced["revenue"]).fillna(0.0)
    coalesced["orders"] = _to_numeric(coalesced["orders"]).fillna(0.0)
    return (
        coalesced.dropna(subset=["date"])
        .sort_values(["date", "channel"])
        .reset_index(drop=True)[["date", "branch_id", "channel", "revenue", "orders"]]
    )


def load_active_branches(
    db_path: Path | str | None = None,
    config: StoreConfig | None = None,
) -> pd.DataFrame:
    """Load non-sensitive active_branches metadata from kiosk.db."""
    cfg = config or load_config()
    target_db = db_path or cfg.kiosk_db_path
    return read_allowlisted_table(target_db, "active_branches", db_kind="kiosk")
