"""Shared pytest fixtures using synthetic in-temp SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_master_db(tmp_path: Path) -> Path:
    """Create a synthetic master_sales.db mirroring real schemas and edge cases."""
    db_path = tmp_path / "master_sales.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE Summary (
            branch_id TEXT,
            date TEXT,
            revenue TEXT,
            total TEXT,
            iva TEXT,
            ordenes TEXT,
            "ticket promedio" TEXT,
            upt TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO Summary VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("DEFAULT", "2026-09-04", "4000.0", "3448.28", "551.72", "25", "160.0", "2.1"),
            ("TC-001", "2026-09-04", "4000.0", "3448.28", "", "25", "160.0", "2.1"),
            ("DEFAULT", "2026-09-05", "4500.0", "3879.31", "620.69", "30", "150.0", "2.4"),
            ("TC-001", "2026-09-06", "5100.0", "4396.55", "703.45", "34", "150.0", "2.5"),
        ],
    )

    cur.execute(
        """
        CREATE TABLE Master_Financials (
            branch_id TEXT,
            date TEXT,
            gross TEXT,
            net_receivable TEXT,
            food TEXT,
            drink TEXT,
            shop TEXT,
            store_name TEXT,
            gross_sales_local TEXT,
            net_revenue_local TEXT,
            card_sales TEXT,
            delivery_sales TEXT,
            app_sales TEXT,
            gross_cash_sales TEXT,
            total_tips TEXT,
            net_cash_deposit TEXT,
            total_tickets TEXT,
            average_ticket TEXT,
            upt TEXT,
            sales_cold_drinks TEXT,
            sales_hot_drinks TEXT,
            sales_food TEXT,
            cash_tips TEXT,
            digital_tips TEXT,
            true_net_ex_iva TEXT,
            app_card_sales TEXT,
            app_wallet_sales TEXT
        )
        """
    )
    # Include: (1) DEFAULT fallback for card_sales, (2) cumulative May-like anomaly in sales_food,
    # (3) normal post-Aug sales_food + sales_hot_drinks + sales_cold_drinks
    cur.executemany(
        "INSERT INTO Master_Financials VALUES (" + ",".join(["?"] * 27) + ")",
        [
            (
                "DEFAULT", "2026-09-04", "4000", "3900", "1000", "2800", "200",
                "The Coffee Paseo MS", "4000", "3900", "2500", "0", "300", "1100",
                "180", "920", "25", "160", "2.1", "", "", "", "", "", "", "", "",
            ),
            (
                "TC-001", "2026-09-04", "4000", "3900", "", "", "",
                "The Coffee Paseo MS", "4000", "3900", "", "0", "300", "1100",
                "180", "920", "25", "160", "2.1", "0", "29000", "11000", "", "", "", "", "",
            ),
            (
                "TC-001", "2026-09-05", "4500", "4400", "", "", "",
                "The Coffee Paseo MS", "4500", "4400", "3100", "0", "400", "900",
                "200", "700", "30", "150", "2.4", "600", "2700", "1100", "", "", "", "", "",
            ),
            (
                "TC-001", "2026-09-06", "5100", "5000", "", "", "",
                "The Coffee Paseo MS", "5100", "5000", "3600", "0", "500", "900",
                "220", "680", "34", "150", "2.5", "700", "3000", "1300", "", "", "", "", "",
            ),
        ],
    )

    cur.execute(
        """
        CREATE TABLE Hourly_Sales (
            branch_id TEXT,
            date TEXT,
            hour TEXT,
            hour_int TEXT,
            volume TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO Hourly_Sales VALUES (?, ?, ?, ?, ?)",
        [
            ("TC-001", "2026-09-04", "4", "4", "999"),   # Out of 6..21 clamp
            ("TC-001", "2026-09-04", "5", "5", "888"),   # Out of 6..21 clamp
            ("TC-001", "2026-09-04", "8", "8", "520.5"),
            ("DEFAULT", "2026-09-04", "9", "9", "410.0"),
            ("TC-001", "2026-09-04", "19", "19", "610.0"),
            ("TC-001", "2026-09-04", "22", "22", "777"), # Out of 6..21 clamp
            ("TC-001", "2026-09-05", "8", "8", "580.0"),
        ],
    )

    cur.execute(
        """
        CREATE TABLE Items (
            branch_id TEXT,
            date TEXT,
            item TEXT,
            qty TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO Items VALUES (?, ?, ?, ?)",
        [
            ("TC-001", "2026-08-19", "Americano", "4"),
            ("TC-001", "2026-08-19", "Gelato FrappÃ© Coffee", "2"),
            ("TC-001", "2026-08-19", "Croissant Serrano & Gouda", "3"),
            ("TC-001", "2026-08-19", "Misto + Tomate Cereja", "1"),
            ("TC-001", "2026-08-19", "Cold Drip V60 + Milk", "1"),
            ("TC-001", "2026-08-19", "Iced Latte +  Ham & Cheese Sandwich", "2"),
            ("TC-001", "2026-08-19", "White bag 250g", "1"),
            ("TC-001", "2026-08-20", "Tumbler (240ml)", "1"),
        ],
    )

    cur.execute(
        """
        CREATE TABLE Contextual_Split (
            branch_id TEXT,
            date TEXT,
            category TEXT,
            revenue TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO Contextual_Split VALUES (?, ?, ?, ?)",
        [
            ("TC-001", "2026-09-05", "Drinks", "3300"),
            ("TC-001", "2026-09-05", "Foods", "1100"),
        ],
    )

    cur.execute(
        """
        CREATE TABLE Sales_Channels (
            branch_id TEXT,
            date TEXT,
            channel TEXT,
            revenue TEXT,
            orders TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO Sales_Channels VALUES (?, ?, ?, ?, ?)",
        [
            ("TC-001", "2026-09-05", "Tablet", "4100", "0"),
            ("TC-001", "2026-09-06", "Tablet", "4500", "0"),
            ("TC-001", "2026-09-06", "Smartphone", "600", "0"),
        ],
    )

    cur.execute(
        """
        CREATE TABLE Day_of_Week_Sales (
            branch_id TEXT,
            date TEXT,
            day_of_week TEXT,
            revenue TEXT,
            orders TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO Day_of_Week_Sales VALUES (?, ?, ?, ?, ?)",
        [
            ("TC-001", "2026-09-05", "Saturday", "4,500.00", "30"),
            ("TC-001", "2026-09-06", "2,441.00", "5100", "34"),
        ],
    )

    # Denylisted PII/financial tables with sentinel values that must never be exposed
    for denied in [
        "Team_Directory",
        "Active_Team",
        "Horarios",
        "Google_Reviews",
        "Bank_Transactions",
        "Expenses",
        "Loans_Schedule",
        "Needs_Review",
        "Tips_Log",
        "Billing_Logs",
        "App_Clients",
    ]:
        cur.execute(f'CREATE TABLE "{denied}" (id INTEGER, secret TEXT)')
        cur.execute(f'INSERT INTO "{denied}" VALUES (1, ?)', (f"DENIED_{denied}",))

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def tmp_kiosk_db(tmp_path: Path) -> Path:
    """Create a synthetic kiosk.db with active_branches and blocked client_vault."""
    db_path = tmp_path / "kiosk.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE active_branches (
            id INTEGER PRIMARY KEY,
            branch_id VARCHAR,
            franchise_name VARCHAR,
            bot_status VARCHAR,
            branch_sheet_id VARCHAR,
            location VARCHAR,
            is_active BOOLEAN
        )
        """
    )
    cur.execute(
        "INSERT INTO active_branches VALUES (1, 'TC-001', 'Paseo MS', 'active', 'SECRET_SHEET_ID', NULL, 1)"
    )

    cur.execute(
        """
        CREATE TABLE client_vault (
            id INTEGER PRIMARY KEY,
            odoo_url TEXT,
            username TEXT,
            password TEXT,
            token TEXT
        )
        """
    )
    cur.execute(
        "INSERT INTO client_vault VALUES (1, 'https://secret.example', 'admin', 'pw123', 'tok_xyz')"
    )

    cur.execute(
        """
        CREATE TABLE billing_logs (
            id INTEGER PRIMARY KEY,
            client_vault_id INTEGER,
            amount FLOAT
        )
        """
    )
    cur.execute("INSERT INTO billing_logs VALUES (1, 1, 999.0)")

    conn.commit()
    conn.close()
    return db_path
