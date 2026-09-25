# Reproduce the whole project from a clean checkout.
PY := .venv/bin/python

.PHONY: help setup data train evaluate report test app all clean

help:
	@echo "setup     Create .venv and install pinned dependencies"
	@echo "data      Download the fraud dataset and generate the synthetic ERP extract"
	@echo "train     Train the autoencoder and the Isolation Forest baseline"
	@echo "evaluate  Produce the comparison table, figures and defect-recall metrics"
	@echo "report    Build the example batch reports in results/examples/"
	@echo "test      Run the test suite"
	@echo "app       Launch the Streamlit demo"
	@echo "all       data -> train -> evaluate -> report -> test"

setup:
	uv venv --python 3.11
	uv pip install -r requirements-dev.txt
	uv pip install -e .
	@# macOS can mark .pth files with the BSD hidden flag, and CPython then
	@# skips them silently, which disables the editable install. Harmless on
	@# Linux, where chflags does not exist.
	-@chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true

data:
	$(PY) scripts/01_download_data.py
	$(PY) scripts/02_generate_erp_data.py

train:
	$(PY) scripts/03_train_models.py

evaluate:
	$(PY) scripts/04_evaluate.py

report:
	$(PY) scripts/05_build_report.py

test:
	$(PY) -m pytest tests/ -q

# Each project under Proyectos_Do_it owns a fixed Streamlit port so their
# demos can run side by side. Override with: make app PORT=xxxx
PORT ?= 8502

app:  ## Launch the Streamlit demo (override with PORT=xxxx)
	.venv/bin/streamlit run app/streamlit_app.py --server.port $(PORT)

all: data train evaluate report test

clean:
	rm -rf data/raw/* data/processed/* data/synthetic/* models/* results/examples/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
