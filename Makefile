PY := .venv/bin/python
PIP := uv pip install --python $(PY)

.PHONY: install test eval ask serve clean

install:
	uv venv .venv --python 3.12
	$(PIP) -e ".[dev]"

test: install
	$(PY) -m pytest

eval: install
	.venv/bin/ragbench eval

ask: install
	.venv/bin/ragbench ask "$(Q)"

serve: install
	$(PIP) -e ".[serve]"
	.venv/bin/uvicorn ragbench.server:app --reload

clean:
	rm -rf .venv .pytest_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
