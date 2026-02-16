# Experiments

Each scenario lives in `scenarios/<name>/` with its own PNML files and `run.py` script.

## Running a scenario

```bash
uv run experiments/scenarios/<name>/run.py
```

## Adding a new scenario

1. Create `experiments/scenarios/<name>/`
2. Place `.pnml` files in it
3. Create `run.py` importing from `lib`
4. Run: `uv run experiments/scenarios/<name>/run.py`

Outputs should go to an `outputs/` subfolder (gitignored).
