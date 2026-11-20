"""Tests for Docker configuration, build-context hygiene, and read-only data mounts."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_IN_MINIMAL_CONTAINER = not (REPO_ROOT / ".git").exists() and not (
    REPO_ROOT / "Dockerfile"
).exists()


def _dockerignore_entries() -> set[str]:
    path = REPO_ROOT / ".dockerignore"
    assert path.is_file(), "Missing .dockerignore"
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.add(stripped)
    return entries


@pytest.mark.skipif(
    _IN_MINIMAL_CONTAINER,
    reason=".dockerignore is intentionally excluded from /app inside the minimal runtime image",
)
def test_dockerignore_blocks_sensitive_and_ephemeral_paths() -> None:
    entries = _dockerignore_entries()
    required_patterns = {
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "*.db-wal",
        "*.db-shm",
        "config/local.toml",
        "notebooks/*.html",
        ".env",
        "__pycache__/",
        ".pytest_cache/",
        ".ipynb_checkpoints/",
    }
    missing = required_patterns - entries
    assert not missing, f"Missing .dockerignore patterns: {sorted(missing)}"

    assert any(p in entries for p in ("data/raw/", "data/raw/*", "data/raw"))
    assert any(p in entries for p in ("data/processed/", "data/processed/*", "data/processed"))
    assert any(p in entries for p in (".git/", ".git"))
    assert any(p in entries for p in (".venv/", ".venv"))


@pytest.mark.skipif(
    _IN_MINIMAL_CONTAINER,
    reason="Dockerfile is intentionally excluded from /app inside the minimal runtime image",
)
def test_dockerfile_security_layering_and_defaults() -> None:
    df_path = REPO_ROOT / "Dockerfile"
    assert df_path.is_file(), "Missing Dockerfile"
    text = df_path.read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in text
    assert "BGA_DATA_MODE=" not in text
    assert "EXPOSE 8888" in text
    assert "HEALTHCHECK" in text
    assert "--ip=0.0.0.0" in text
    assert "--no-browser" in text
    assert "jupyter" in text

    user_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("USER ")]
    assert user_lines, "Dockerfile must specify a non-root USER"
    assert user_lines[-1] != "USER root"

    req_idx = text.index("COPY --chown=appuser:appuser requirements.txt")
    install_idx = text.index("requirements.txt", req_idx + 1)
    src_idx = text.index("COPY --chown=appuser:appuser src/")
    assert req_idx < install_idx < src_idx

    assert not re.search(r"(?m)^COPY\s+(?:--[^\s]+\s+)*\.\s+", text)
    assert "data/raw/" not in text.split("COPY", 1)[1] or "mkdir" in text
    assert "config/local.toml" not in text
    assert "COPY --chown=appuser:appuser config/default.toml ./config/default.toml" in text
    assert "pyproject.toml" in text
    for required_dir in ("src/", "notebooks/", "docs/", "scripts/", "tests/", "data/sample/"):
        assert required_dir in text


def test_pyproject_and_bounded_requirements() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), "Missing pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert data["project"]["name"] == "business-geo-analytics"
    assert ">=3.12" in data["project"]["requires-python"]
    pytest_opts = data["tool"]["pytest"]["ini_options"]
    assert "." in pytest_opts["pythonpath"]
    assert "tests" in pytest_opts["testpaths"]

    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.is_file(), "Missing requirements.txt"
    req_lines = [
        ln.strip()
        for ln in req_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    pkgs = {re.split(r"[<>=!~]", ln, maxsplit=1)[0].lower(): ln for ln in req_lines}
    for required_pkg in ("pandas", "numpy", "statsmodels", "nbconvert", "jupyterlab", "pytest"):
        assert required_pkg in pkgs, f"Missing {required_pkg} in requirements.txt"

    for ln in req_lines:
        assert re.match(
            r"^[A-Za-z0-9_.-]+>=[0-9.]+,<[0-9.]+$", ln
        ), f"Requirement must specify lower and upper bounds: {ln}"


@pytest.mark.skipif(
    _IN_MINIMAL_CONTAINER,
    reason="docker-compose.yml is intentionally excluded from /app inside the minimal runtime image",
)
def test_docker_compose_services_and_readonly_raw_mount() -> None:
    dc_path = REPO_ROOT / "docker-compose.yml"
    assert dc_path.is_file(), "Missing docker-compose.yml"
    text = dc_path.read_text(encoding="utf-8")

    for svc in ("jupyter:", "test:", "report:"):
        assert svc in text, f"Missing service {svc} in docker-compose.yml"

    assert "8888:8888" in text
    assert "BGA_DATA_MODE=${BGA_DATA_MODE:-sample}" not in text
    assert 'profiles: ["tools"]' in text or "- tools" in text
    assert "./data/processed:/app/data/processed" in text
    assert "./data/raw:/app/data/raw:ro" in text
    assert "./config:/app/config:ro" in text

    ro_mounts = re.findall(r"[^\s'\"]*(?:data/raw|:/app/config)[^\s'\"]*", text)
    assert ro_mounts
    for mount in ro_mounts:
        if ":" in mount:
            assert mount.endswith(":ro"), f"Mount is not read-only: {mount}"

    assert "pytest" in text
    assert "scripts/verify_repo.py" in text
    assert "load_config()" in text
    assert "src.report" in text or "src/report.py" in text


@pytest.mark.skipif(
    _IN_MINIMAL_CONTAINER,
    reason="README.md is intentionally excluded from /app inside the minimal runtime image",
)
def test_readme_documents_docker_quickstart() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Docker Quickstart" in readme
    assert "docker compose up" in readme
    assert "BGA_DATA_MODE=raw docker compose up" in readme


@pytest.mark.skipif(
    not _IN_MINIMAL_CONTAINER,
    reason="Container isolation runtime checks only apply inside the built Docker image",
)
def test_container_runtime_non_root_and_clean_context() -> None:
    from src.config import load_config

    assert os.geteuid() != 0, "Container must execute as non-root user"
    assert not (REPO_ROOT / ".git").exists()
    assert not (REPO_ROOT / ".env").exists()
    assert (REPO_ROOT / "config" / "default.toml").is_file()
    assert load_config(repo_root=REPO_ROOT).data_mode in {"sample", "raw"}
