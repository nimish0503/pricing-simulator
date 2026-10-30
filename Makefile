CLIENT ?= client_a

.PHONY: setup onboard cluster model backtest simulate app report test lint

setup:
	uv sync
	uv run pre-commit install

# Phase 2 to 5 targets fail until their module exists.
onboard:
	uv run python -m shelfsim.onboarding --client $(CLIENT)

cluster:
	uv run python -m shelfsim.segmentation --client $(CLIENT)

model:
	uv run python -m shelfsim.demand --client $(CLIENT)

backtest:
	uv run python -m shelfsim.evaluation --client $(CLIENT)

simulate:
	uv run python -m shelfsim.choice --client $(CLIENT)

app:
	uv run streamlit run app/streamlit_app.py

report:
	uv run python -m shelfsim.report --client $(CLIENT)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
