"""Executive report generator filling docs/master_franchise_report.md into data/processed/."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.causal_eval import (
    analyze_attach_and_segmentation,
    build_osint_normalized_baseline,
    build_sensitivity_matrix,
    simulate_linebuster_scenarios,
)
from src.config import StoreConfig, load_config
from src.osint_collector import load_incidents_file


def _fmt_num(val: float, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf"
    return f"{val:,.{decimals}f}"


def _fmt_pct(val: float, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val * 100.0:.{decimals}f}%"


def _df_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def _render_scenarios_table(scenarios_df: pd.DataFrame, currency: str) -> str:
    headers = [
        "Scenario",
        "Peak Lift",
        "Attach Lift",
        f"Inc. Rev / Day ({currency})",
        f"Monthly Net Profit ({currency})",
        "Break-Even Tickets / Day",
        "Payback (Months)",
        "% of P95 Peak Ceiling",
    ]
    rows: list[list[str]] = []
    for _, r in scenarios_df.iterrows():
        rows.append(
            [
                str(r["scenario"]),
                _fmt_pct(float(r["peak_lift_pct"]), 0),
                _fmt_pct(float(r["attach_lift_pct"]), 0),
                _fmt_num(float(r["total_incremental_rev_day"])),
                _fmt_num(float(r["monthly_net_incremental_profit"])),
                _fmt_num(float(r["breakeven_tickets_per_day"])),
                _fmt_num(float(r["payback_months"])),
                _fmt_pct(float(r["pct_of_p95_peak_ceiling"]), 1),
            ]
        )
    return _df_to_markdown(headers, rows)


def _render_sensitivity_table(sens_df: pd.DataFrame) -> str:
    margins = sorted(sens_df["gross_margin_pct"].unique())
    lifts = sorted(sens_df["peak_lift_pct"].unique())
    lookup = {
        (float(r["gross_margin_pct"]), float(r["peak_lift_pct"])): r
        for _, r in sens_df.iterrows()
    }
    headers = ["Gross Margin \\ Peak Lift"] + [_fmt_pct(lift, 0) for lift in lifts]
    rows: list[list[str]] = []
    for margin in margins:
        row_cells = [_fmt_pct(float(margin), 0)]
        for lift in lifts:
            rec = lookup[(float(margin), float(lift))]
            net_p = _fmt_num(float(rec["monthly_net_incremental_profit"]), 0)
            pb = _fmt_num(float(rec["payback_months"]), 1)
            row_cells.append(f"{net_p} ({pb}m)")
        rows.append(row_cells)
    return _df_to_markdown(headers, rows)


def _load_inputs_for_mode(
    mode: str, cfg: StoreConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    if mode == "raw":
        from src.db_loader import (
            load_contextual_split,
            load_daily_sales,
            load_hourly_sales,
            load_product_mix,
            load_sales_channels,
        )

        return (
            load_daily_sales(config=cfg),
            load_hourly_sales(config=cfg, complete_grid=True),
            load_product_mix(config=cfg),
            load_contextual_split(config=cfg),
            load_sales_channels(config=cfg),
        )
    if mode == "sample":
        return (
            pd.read_csv(cfg.sample_dir / "daily_sales.csv", parse_dates=["date"]),
            pd.read_csv(cfg.sample_dir / "hourly_sales.csv", parse_dates=["date"]),
            pd.read_csv(cfg.sample_dir / "product_mix.csv", parse_dates=["date"]),
            None,
            None,
        )
    raise ValueError(f"Unsupported mode {mode!r}; expected 'sample' or 'raw'")


def generate_franchise_report(
    mode: str | None = None,
    config: StoreConfig | None = None,
    output_path: Path | str | None = None,
    template_path: Path | str | None = None,
    labor_hours_per_day: float | None = None,
) -> dict[str, Any]:
    """Populate executive report template and write output safely to data/processed/."""
    cfg = config or load_config()
    norm_mode = (mode if mode is not None else cfg.data_mode).strip().lower()

    resolved_output = (
        Path(output_path)
        if output_path is not None
        else cfg.processed_dir / f"master_franchise_report_{norm_mode}.md"
    )
    if norm_mode == "raw":
        proc_dir = cfg.processed_dir.resolve()
        out_resolved = resolved_output.resolve()
        if not out_resolved.is_relative_to(proc_dir):
            raise ValueError(
                f"Real-metric report in 'raw' mode must stay inside data/processed/ ({proc_dir}), got {out_resolved}"
            )

    tpl_file = (
        Path(template_path)
        if template_path is not None
        else cfg.repo_root / "docs" / "master_franchise_report.md"
    )
    template_text = tpl_file.read_text(encoding="utf-8")

    daily_df, hourly_df, mix_df, ctx_df, ch_df = _load_inputs_for_mode(norm_mode, cfg)
    inc_path = cfg.sample_dir / "incidents.csv"
    incidents_df = (
        load_incidents_file(inc_path, relevant_only=True)
        if inc_path.is_file()
        else pd.DataFrame()
    )

    baseline = build_osint_normalized_baseline(
        daily_df=daily_df,
        hourly_df=hourly_df,
        incidents_df=incidents_df,
        store_lat=cfg.lat,
        store_lon=cfg.lon,
        max_distance_m=1500.0,
        peak_hours=cfg.peak_hours,
        operating_hours=cfg.operating_hours,
    )
    seg = analyze_attach_and_segmentation(
        daily_df=daily_df,
        product_mix_df=mix_df,
        contextual_split_df=ctx_df,
        sales_channels_df=ch_df,
        tablet_start_date=cfg.tablet_channel_start_date,
        tablet_interpretation=cfg.tablet_channel_interpretation,
        items_end_date=cfg.items_coverage_end_date,
    )
    scenarios_df = simulate_linebuster_scenarios(
        baseline=baseline,
        config=cfg,
        labor_hours_per_day=labor_hours_per_day,
    )
    sens_df = build_sensitivity_matrix(
        baseline=baseline,
        config=cfg,
        labor_hours_per_day=labor_hours_per_day,
    )

    base_row = scenarios_df.loc[scenarios_df["scenario"] == "Base"].iloc[0]
    coords_status = (
        f"Verified ({cfg.lat:.6f}, {cfg.lon:.6f})"
        if cfg.coords_verified
        else f"Approximate ({cfg.lat:.4f}, {cfg.lon:.4f}, COORDS_VERIFIED=False)"
    )
    date_min = pd.to_datetime(daily_df["date"]).min().strftime("%Y-%m-%d")
    date_max = pd.to_datetime(daily_df["date"]).max().strftime("%Y-%m-%d")
    peak_label = ", ".join(f"{h:02d}:00" for h in cfg.peak_hours)

    replacements = {
        "data_mode": norm_mode,
        "branch_id": str(daily_df["branch_id"].iloc[0]) if not daily_df.empty else cfg.branch_id,
        "store_name": cfg.store_name if norm_mode == "raw" else f"{cfg.store_name} (Synthetic Sample)",
        "city": cfg.city,
        "currency": cfg.currency,
        "coords_status": coords_status,
        "date_range": f"{date_min} to {date_max}",
        "total_days": str(baseline["total_days"]),
        "incident_days_excluded": str(baseline["incident_days_excluded"]),
        "clean_days": str(baseline["clean_days"]),
        "peak_hours_label": peak_label,
        "peak_revenue_share_pct": _fmt_pct(float(baseline["peak_revenue_share"])),
        "clean_mean_daily_revenue": _fmt_num(float(baseline["clean_mean_daily_revenue"])),
        "clean_mean_daily_peak_revenue": _fmt_num(float(baseline["clean_mean_daily_peak_revenue"])),
        "peak_p90_ceiling": _fmt_num(float(baseline["peak_p90_daily_ceiling"])),
        "peak_p95_ceiling": _fmt_num(float(baseline["peak_p95_daily_ceiling"])),
        "clean_mean_avg_ticket": _fmt_num(float(baseline["clean_mean_avg_ticket"])),
        "clean_mean_tickets_per_day": _fmt_num(float(baseline["clean_mean_tickets_per_day"]), 1),
        "tablet_start_date": cfg.tablet_channel_start_date,
        "tablet_interpretation": cfg.tablet_channel_interpretation,
        "food_revenue_share_pct": _fmt_pct(float(seg["food_revenue_share"])),
        "combo_unit_share_pct": _fmt_pct(float(seg["combo_unit_share_pre_cutoff"])),
        "hardware_capex": _fmt_num(cfg.hardware_capex),
        "monthly_saas_fee": _fmt_num(cfg.monthly_saas_fee),
        "peak_labor_cost_per_hr": _fmt_num(cfg.peak_labor_cost_per_hr),
        "gross_margin_pct": _fmt_pct(cfg.gross_margin_pct, 0),
        "scenarios_table_md": _render_scenarios_table(scenarios_df, cfg.currency),
        "sensitivity_table_md": _render_sensitivity_table(sens_df),
        "base_monthly_net_profit": _fmt_num(float(base_row["monthly_net_incremental_profit"])),
        "base_payback_months": _fmt_num(float(base_row["payback_months"])),
        "base_breakeven_tickets_per_day": _fmt_num(float(base_row["breakeven_tickets_per_day"])),
    }

    rendered = template_text
    for key, val in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", val)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(rendered, encoding="utf-8")

    return {
        "output_path": resolved_output,
        "mode": norm_mode,
        "baseline": baseline,
        "scenarios": scenarios_df,
        "sensitivity": sens_df,
    }


if __name__ == "__main__":
    res = generate_franchise_report(mode="sample")
    print(f"Wrote sample report: {res['output_path']}")
