# Single command surface for humans, CI, and coding agents.
# Uses the local venv if present, otherwise whatever python3 is on PATH.
PY := $(shell test -x venv/bin/python && echo venv/bin/python || echo python3)

.PHONY: test lint format bench-smoke m2-smoke docs-check

test:
	$(PY) -m pytest tests/ modules/ -q

lint:
	$(PY) -m ruff check src tests benchmarks modules

format:
	$(PY) -m ruff format src tests benchmarks modules
	$(PY) -m ruff check --fix src tests benchmarks modules

# Fast CPU sanity benchmark: builds the tiny preset, runs a forward pass and
# a short generation, writes benchmarks/results/smoke.json. Must stay <1 min.
bench-smoke:
	$(PY) benchmarks/smoke.py --preset tiny --device cpu

m2-smoke:
	$(PY) -m pytest modules/m2_inference_opt/ -q
	$(PY) modules/m2_inference_opt/benchmark.py --device cpu --preset tiny

docs-check:
	$(PY) scripts/check_module_docs.py
