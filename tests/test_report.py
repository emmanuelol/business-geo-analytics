"""Tests for src.report and docs templates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import load_config
from src.report import generate_franchise_report


def test_docs_exist_and_template_is_unfilled() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    methodology = repo_root / "docs" / "methodology.md"
    template = repo_root / "docs" / "master_franchise_report.md"

    assert methodology.is_file()
    assert template.is_file()

    tpl_text = template.read_text(encoding="utf-8")
    placeholders = re.findall(r"\{\{[a-z0-9_]+\}\}", tpl_text)
    assert len(placeholders) >= 12
    assert "{{scenarios_table_md}}" in tpl_text
    assert "{{sensitivity_table_md}}" in tpl_text


def test_generate_sample_report_fills_all_placeholders(tmp_path: Path) -> None:
    cfg = load_config()
    out_file = tmp_path / "sample_report.md"

    result = generate_franchise_report(
        mode="sample",
        config=cfg,
        output_path=out_file,
    )
    assert result["output_path"] == out_file
    assert out_file.is_file()

    rendered = out_file.read_text(encoding="utf-8")
    assert "{{" not in rendered and "}}" not in rendered
    assert "Conservative" in rendered
    assert "Base" in rendered
    assert "Optimistic" in rendered
    assert "Difference-in-Differences" in rendered
    assert "pos_channel_change" in rendered

    # Confirm source template on disk remained untouched
    tpl_text = (cfg.repo_root / "docs" / "master_franchise_report.md").read_text(
        encoding="utf-8"
    )
    assert "{{scenarios_table_md}}" in tpl_text


def test_raw_report_cannot_be_written_outside_processed_dir(tmp_path: Path) -> None:
    cfg = load_config()
    outside = tmp_path / "leaked_report.md"

    with pytest.raises(ValueError, match="data/processed"):
        generate_franchise_report(
            mode="raw",
            config=cfg,
            output_path=outside,
        )
