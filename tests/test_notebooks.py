"""Headless execution tests for all 5 notebooks in data/sample mode."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS: tuple[str, ...] = (
    "01_sales_baseline.ipynb",
    "02_silao_osint_pipeline.ipynb",
    "03_incident_correlation.ipynb",
    "04_location_diagnostics.ipynb",
    "05_linebuster_roi_study.ipynb",
)


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_executes_headless_on_sample(
    nb_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / nb_name
    assert nb_path.is_file(), f"Missing notebook: {nb_path}"

    monkeypatch.setenv("BGA_DATA_MODE", "sample")
    monkeypatch.setenv("BGA_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "jup_rt"))
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(tmp_path / "jup_data"))

    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=90, kernel_name="python3")
    executed, _ = ep.preprocess(
        nb, {"metadata": {"path": str(nb_path.parent)}}
    )

    code_cells = [c for c in executed.cells if c.cell_type == "code"]
    assert len(code_cells) >= 3
    for cell in code_cells:
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", f"Cell error in {nb_name}: {out}"
    assert any(cell.get("outputs") for cell in code_cells), f"No outputs in {nb_name}"
