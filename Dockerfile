FROM python:3.12-slim

ARG UID=1000
ARG GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/bga-matplotlib \
    IPYTHONDIR=/tmp/bga-ipython \
    JUPYTER_RUNTIME_DIR=/tmp/bga-jupyter-runtime \
    JUPYTER_DATA_DIR=/tmp/bga-jupyter-data \
    HOME=/home/appuser

RUN groupadd --gid "${GID}" appuser \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir --disable-pip-version-check uv \
    && uv pip install --system --no-cache -r requirements.txt

RUN mkdir -p /app/data/raw /app/data/processed /tmp/bga-matplotlib /tmp/bga-ipython /tmp/bga-jupyter-runtime /tmp/bga-jupyter-data \
    && chown -R appuser:appuser /app /tmp/bga-* /home/appuser

COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser config/default.toml ./config/default.toml
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser notebooks/ ./notebooks/
COPY --chown=appuser:appuser docs/ ./docs/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser data/sample/ ./data/sample/

USER appuser

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8888/login')" || exit 1

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
