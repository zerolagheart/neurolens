# NeuroLens backend image (CPU). The same image also runs the Streamlit frontend in compose.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch wheels keep the image small; install them before the project.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" "torchvision>=0.17,<1"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,app,onnx]"

COPY api ./api
COPY app ./app
COPY configs ./configs
COPY docs ./docs
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 neurolens && mkdir -p /app/artifacts && chown -R neurolens /app
USER neurolens

ENV NEUROLENS_RUNS_DIR=/app/artifacts/runs \
    NEUROLENS_REPORTS_DIR=/app/artifacts/reports \
    NEUROLENS_DEVICE=cpu

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
