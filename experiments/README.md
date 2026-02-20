# Experiments

Each scenario lives in `scenarios/<name>/` with its own PNML files and `run.py` script.

## Running a scenario

```bash
uv run experiments/scenarios/<name>/run.py [OPTIONS]
```

## Scenario options

### sequential

```
--format {markdown,latex}   Output table format (default: markdown)
--output, -o PATH           Write tables to file instead of stdout
--plot                      Generate analysis plots
--plot-format {pdf,png}     Plot output format (default: pdf)
```

### fork-join

```
--format {markdown,latex}   Output table format (default: markdown)
--output, -o PATH           Write tables to file instead of stdout
--plot                      Generate analysis plots
--plot-format {pdf,png}     Plot output format (default: pdf)
```

### football

```
--format {markdown,latex}   Output table format (default: markdown)
--output, -o PATH           Write table to file instead of stdout
--skip-viz                  Skip PNG/PDF rendering (faster iteration)
--plot                      Generate analysis plots
--plot-format {pdf,png}     Plot output format (default: pdf)
```

### football-2team

```
--format {markdown,latex}   Output table format (default: markdown)
--output, -o PATH           Write table to file instead of stdout
--skip-viz                  Skip PNG rendering (faster iteration)
--plot                      Generate analysis plots
--plot-format {pdf,png}     Plot output format (default: pdf)
```

## Adding a new scenario

1. Create `experiments/scenarios/<name>/`
2. Place `.pnml` files in it
3. Create `run.py` importing from `lib`
4. Run: `uv run experiments/scenarios/<name>/run.py`

Outputs should go to an `outputs/` subfolder (gitignored).
