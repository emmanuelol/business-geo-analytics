"""Structural, mode-isolation, and headless execution tests for all 5 notebooks."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from src.config import StoreConfig
from src.notebook_utils import load_incidents_for_mode

NOTEBOOK_REQUIREMENTS: tuple[tuple[str, int, int], ...] = (
    ("01_sales_baseline.ipynb", 3, 3),
    ("02_silao_osint_pipeline.ipynb", 1, 2),
    ("03_incident_correlation.ipynb", 1, 4),
    ("04_location_diagnostics.ipynb", 1, 3),
    ("05_linebuster_roi_study.ipynb", 1, 5),
)
NOTEBOOKS: tuple[str, ...] = tuple(item[0] for item in NOTEBOOK_REQUIREMENTS)


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebooks_narrative_backend_and_mode_isolation(nb_name: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / nb_name
    assert nb_path.is_file(), f"Missing notebook: {nb_path}"

    nb = nbformat.read(nb_path, as_version=4)
    code_sources: list[str] = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        assert idx > 0 and nb.cells[idx - 1].cell_type == "markdown", (
            f"Code cell {cell.get('id')} in {nb_name} must be preceded by a Markdown cell"
        )
        for line in cell.source.splitlines():
            assert not re.match(r"^\s*matplotlib\.use\(", line), (
                f"Unguarded matplotlib.use() in {nb_name} cell {cell.get('id')}: {line}"
            )
            assert ".to_string(" not in line, (
                f"Plain-text .to_string() table print in {nb_name} cell {cell.get('id')}"
            )
        code_sources.append(cell.source)

    full_src = "\n".join(code_sources)
    for forbidden_env in ("BGA_DATA_MODE", "BGA_FETCH_LIVE", "BGA_OSINT_START", "BGA_OSINT_END"):
        assert forbidden_env not in full_src, (
            f"{nb_name} should read {forbidden_env} via load_config(), not os.environ directly"
        )
    assert "DATA_MODE = cfg.data_mode" in full_src

    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for stale_doc in ("local_assumptions.json", 'BGA_DATA_MODE="sample"', 'BGA_FETCH_LIVE="0"'):
        assert stale_doc not in md_text, (
            f"{nb_name} markdown still references stale config text {stale_doc!r}"
        )

    if nb_name == "02_silao_osint_pipeline.ipynb":
        assert "fetch_rss_feed" not in full_src
        assert "build_google_news_rss_urls_for_window" in full_src
        assert "ignore_fetch_errors=True" in full_src
        assert 'DATA_MODE != "sample"' in full_src or "DATA_MODE != 'sample'" in full_src
        assert "OSINT_START" in full_src and "OSINT_END" in full_src
        tree = ast.parse(full_src)
        sample_ifs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "DATA_MODE" in ast.unparse(node.test)
            and "sample_feed.xml" in ast.unparse(node.body)
        ]
        assert sample_ifs, "sample_feed.xml must only be loaded inside DATA_MODE == 'sample'"
        assert not any("sample_feed.xml" in ast.unparse(n.orelse) for n in sample_ifs)
    elif nb_name in (
        "03_incident_correlation.ipynb",
        "04_location_diagnostics.ipynb",
        "05_linebuster_roi_study.ipynb",
    ):
        assert 'sample_dir / "incidents.csv"' not in full_src
        assert "sample_feed.xml" not in full_src
        assert "load_incidents_for_mode" in full_src
        if nb_name == "03_incident_correlation.ipynb":
            assert "filter_incidents_for_analysis" in full_src
            assert "regime_dates=" in full_src
            assert "control_cols=" in full_src
            assert "regime_and_trend_audit" in full_src
        if nb_name == "04_location_diagnostics.ipynb":
            assert "filter_incidents_for_analysis" in full_src
            assert "exclude_coarse_centroids=True" in full_src
        if nb_name == "05_linebuster_roi_study.ipynb":
            assert "weekdays_only=True" in full_src
            assert "(7, 8, 9)" in full_src
            assert "estimate_hourly_bar_throughput" in full_src
            assert 'c != "gross_margin_pct"' in full_src


def test_raw_mode_never_falls_back_to_sample_incidents(tmp_path: Path) -> None:
    (tmp_path / "data" / "sample").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "sample" / "incidents.csv").write_text(
        "incident_id,date,title,category\n1,2026-06-01,Test,traffic_accident\n",
        encoding="utf-8",
    )
    cfg = StoreConfig(repo_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="incidents_raw.csv"):
        load_incidents_for_mode(cfg, mode="raw")


@pytest.mark.parametrize(("nb_name", "min_png", "min_html"), NOTEBOOK_REQUIREMENTS)
def test_notebook_executes_headless_on_sample(
    nb_name: str,
    min_png: int,
    min_html: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / nb_name
    assert nb_path.is_file(), f"Missing notebook: {nb_path}"

    monkeypatch.setenv("BGA_DATA_MODE", "sample")
    monkeypatch.setenv("BGA_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "jup_rt"))
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(tmp_path / "jup_data"))

    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    executed, _ = ep.preprocess(
        nb, {"metadata": {"path": str(nb_path.parent)}}
    )

    code_cells = [c for c in executed.cells if c.cell_type == "code"]
    assert len(code_cells) >= 3

    png_count = 0
    html_count = 0
    for cell in code_cells:
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", f"Cell error in {nb_name}: {out}"
            if out.get("output_type") == "stream":
                assert "FigureCanvasAgg is non-interactive" not in out.get("text", ""), (
                    f"Non-interactive Agg warning emitted in {nb_name}"
                )
            data = out.get("data", {})
            if "image/png" in data:
                png_count += 1
            if "text/html" in data:
                html_count += 1

    assert png_count >= min_png, f"{nb_name}: expected >= {min_png} image/png, got {png_count}"
    assert html_count >= min_html, f"{nb_name}: expected >= {min_html} text/html, got {html_count}"
