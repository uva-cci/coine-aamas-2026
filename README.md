# Power-Index Analysis of Petri-Net Multi-Agent Workflows

Reproduction package for COINE @ AAMAS 2026 paper submission "Exploring Measures of Agentic Power for Organisational Workflow Structures".

## Project Structure

```
├── pyproject.toml / uv.lock
├── src/lib/                          # core library (installed as `lib` via uv)
│   ├── __init__.py                   # re-exports all public API
│   ├── io.py                         # PNML loading
│   ├── analysis.py                   # power indices & structural analysis
│   ├── viz.py                        # PNG visualization & comparative plots
│   └── py.typed
└── experiments/scenarios/
    ├── example/                      # minimal net
    ├── sequential/                   # sequential net
    ├── fork-join/                    # fork-join net
    ├── football/                     # zone-based stochastic SPN
    │   ├── run.py
    │   └── generate_pnml.py          # builds football.pnml programmatically
    └── football-2team/               # two-team adversarial SPN
        ├── run.py
        └── generate_pnml.py          # builds football-2team.pnml programmatically
```

Each scenario directory contains a `run.py` and one `.pnml` file. Outputs go to an `outputs/` subfolder (gitignored).

## Getting Started

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                        # install dependencies
uv run ruff check src/ experiments/            # lint
uv run ruff format --check src/ experiments/   # format check
```

## Running Experiments

```bash
uv run experiments/scenarios/<name>/run.py [OPTIONS]
```

### Scenario options

#### sequential / fork-join

| Flag                        | Description                             |
| --------------------------- | --------------------------------------- |
| `--format {markdown,latex}` | Output table format (default: markdown) |
| `--output, -o PATH`         | Write tables to file instead of stdout  |
| `--plot`                    | Generate analysis plots                 |
| `--plot-format {pdf,png}`   | Plot output format (default: pdf)       |

#### football / football-2team

| Flag                        | Description                               |
| --------------------------- | ----------------------------------------- |
| `--format {markdown,latex}` | Output table format (default: markdown)   |
| `--output, -o PATH`         | Write table to file instead of stdout     |
| `--skip-viz`                | Skip PNG/PDF rendering (faster iteration) |
| `--plot`                    | Generate analysis plots                   |
| `--plot-format {pdf,png}`   | Plot output format (default: pdf)         |

### Adding a new scenario

1. Create `experiments/scenarios/<name>/`
2. Place `.pnml` files in it
3. Create `run.py` importing from `lib`
4. Run: `uv run experiments/scenarios/<name>/run.py`

Outputs should go to an `outputs/` subfolder (gitignored).

## Analysis Tooling

The `lib` package exposes the following public API.

### PNML I/O

| Function                     | Description                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| `load_pnml(path)`            | Load a Petri net from a PNML file. Returns `(net, initial_marking, final_marking)`.    |
| `load_pnml_stochastic(path)` | Load a Petri net with stochastic information. Returns `(net, im, fm, stochastic_map)`. |

### Structural Utilities

| Function                                        | Description                                                      |
| ----------------------------------------------- | ---------------------------------------------------------------- |
| `incidence_matrix(net)`                         | Compute the incidence matrix (rows=places, columns=transitions). |
| `reachability_graph(net, im)`                   | Build the full reachability graph from an initial marking.       |
| `is_reachable_restricted(net, im, fm, allowed)` | BFS reachability check using only allowed transitions.           |

### Power Indices

| Function                                     | Description                                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `shapley_shubik(net, im, fm, agent_mapping)` | Shapley-Shubik index via coalition enumeration over the reachability characteristic function.                             |
| `banzhaf(net, im, fm, agent_mapping)`        | Banzhaf index — normalized swing count over all coalitions.                                                               |
| `usability(net, im, fm, agent_mapping)`      | Shared-credit index over firing-sequence prefixes; divides credit by the number of capable agents per transition.         |
| `gatekeeper(net, im, fm, agent_mapping)`     | Immediate-dominator-based index; weights transitions by how many other transitions they dominate in the transition graph. |

Pre-computed characteristic function variants (`shapley_shubik_from_values`, `banzhaf_from_values`) accept a `v: dict[frozenset[str], float]` mapping coalitions to values, for use with custom characteristic functions (e.g. stochastic absorbing-chain probabilities).

### Distributional Metrics

| Function                     | Description                                                            |
| ---------------------------- | ---------------------------------------------------------------------- |
| `gini_coefficient(values)`   | Gini coefficient of a distribution (0 = equal, 1 = maximally unequal). |
| `granularity(agent_mapping)` | Gini index of the supply-degree distribution across transitions.       |

### Visualization

| Function                          | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `save_net_png(net, im, fm, path)` | Render a Petri net as PNG (optional stochastic decorations).      |
| `plot_power_bars(...)`            | Grouped bar chart of per-agent power across configurations.       |
| `plot_power_heatmap(...)`         | Heatmap grid (configs x agents, one subplot per index).           |
| `plot_index_correlation(...)`     | Pairwise scatter plots between indices, colored by configuration. |
| `plot_lorenz_curves(...)`         | Lorenz curves per configuration (one subplot per index).          |
| `plot_rank_agreement(...)`        | Spearman rank-correlation heatmaps between indices.               |
| `plot_power_deltas(...)`          | Power change from a baseline configuration per agent/index.       |
| `plot_granularity_scatter(...)`   | Granularity vs per-index summary statistics scatter plot.         |
