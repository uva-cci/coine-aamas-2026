# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

Research project for COINE @ AAMAS 2026 — Petri net analysis using pm4py. The `lib` package (`src/lib/`) provides thin wrappers around pm4py for loading, analyzing, and visualizing Petri nets. Experiments live in `experiments/scenarios/<name>/` each with PNML files and a `run.py` script.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_bar

# Lint
uv run ruff check src/ experiments/
uv run ruff format --check src/ experiments/

# Run an experiment scenario
uv run experiments/scenarios/<name>/run.py
```

## Architecture

- **`src/lib/`** — Core library (installed as `lib` package via uv)
  - `io.py` — PNML loading (`load_pnml`)
  - `analysis.py` — Structural analysis (incidence matrix, reachability graph)
  - `viz.py` — PNG visualization of nets
- **`experiments/scenarios/<name>/`** — Each scenario has its own PNML files and `run.py`; outputs go to an `outputs/` subfolder (gitignored)

## Conventions

- Python 3.12+, managed with uv
- Ruff for linting/formatting (line-length 99, rules: E, F, I, UP)
- Petri net types come from `pm4py.objects.petri_net.obj` (PetriNet, Marking)
- Experiment scripts use `from lib import ...` — the package is importable because it's declared in `pyproject.toml`
