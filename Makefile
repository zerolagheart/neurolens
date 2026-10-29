# Convenience targets (GNU make). Every target is a plain command you can also run directly.
PYTHON ?= python
DATA ?= data/raw/brain-tumor-mri-dataset
RUN ?= resnet18_baseline
CONFIG ?= configs/resnet18.yaml

.PHONY: install install-dev lint format test test-fast audit split train calibrate evaluate \
        benchmark export compare explain api app docker-build docker-up clean

install:
	$(PYTHON) -m pip install -e ".[api,app,onnx]"

install-dev:
	$(PYTHON) -m pip install -e ".[all]"

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

audit:
	neurolens audit --data-root $(DATA)

split:
	neurolens make-split --config $(CONFIG)

train:
	neurolens train --config $(CONFIG)

calibrate:
	neurolens calibrate --run $(RUN)

evaluate:
	neurolens evaluate --run $(RUN)

benchmark:
	neurolens benchmark --run $(RUN)

export:
	neurolens export-onnx --run $(RUN) --quantize

compare:
	neurolens compare

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
