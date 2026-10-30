CLIENT ?= client_a

.PHONY: setup onboard cluster model backtest simulate app report test lint

setup:
	uv sync
	uv run pre-commit install

onboard:
	uv run python -m shelfsim onboard --client $(CLIENT)

cluster:
	uv run python -m shelfsim cluster --client $(CLIENT)

model:
	uv run python -m shelfsim model --client $(CLIENT)

backtest:
	uv run python -m shelfsim backtest --client $(CLIENT)

simulate:
	uv run python -m shelfsim simulate --client $(CLIENT)

app:
	uv run streamlit run app/streamlit_app.py

report:
	uv run python -m shelfsim report --client $(CLIENT)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
