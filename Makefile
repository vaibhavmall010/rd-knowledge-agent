MODEL ?= llama3.1

.PHONY: install test lint eval-tools eval app

install:
	python -m pip install -e ".[app,dev]"

test:
	python -m pytest -q

lint:
	ruff check src tests app && ruff format --check src tests app

eval-tools:
	python -m rdagent.evaluate --mode tools

# Needs Ollama running locally: ollama pull $(MODEL)
eval:
	python -m rdagent.evaluate --mode compare --model $(MODEL)

app:
	python -m streamlit run app/streamlit_app.py
