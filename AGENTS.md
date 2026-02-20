# AGENTS.md

Project-scoped memory for AI coding agents working on this repository.

## Project Overview

Research project for COINE @ AAMAS 2026 — Petri net analysis using pm4py. The `lib` package (`src/lib/`) provides thin wrappers around pm4py for loading, analyzing, and visualizing Petri nets. Experiments live in `experiments/scenarios/<name>/`.

## Project Structure

```
├── AGENTS.md
├── pyproject.toml / uv.lock
├── src/lib/                          # core library (installed as `lib` via uv)
│   ├── __init__.py                   # re-exports all public API
│   ├── io.py                         # PNML loading
│   ├── analysis.py                   # power indices & structural analysis
│   ├── viz.py                        # PNG visualization
│   └── py.typed
├── experiments/scenarios/
│   ├── example/                      # minimal net — simple_net.pnml
│   ├── sequential/                   # sequential net — sequential.pnml
│   ├── fork-join/                    # fork-join net — fork_join.pnml
│   ├── football/                     # zone-based stochastic SPN — football.pnml
│   │   ├── run.py
│   │   └── generate_pnml.py          # builds football.pnml programmatically
│   └── football-2team/               # two-team adversarial SPN — football-2team.pnml
│       ├── run.py
│       └── generate_pnml.py          # builds football-2team.pnml programmatically
└── tests/                            # configured in pyproject.toml (empty)
```

Each scenario directory contains a `run.py` and one `.pnml` file. Outputs go to an `outputs/` subfolder (gitignored).

## Commands

```bash
uv sync                                       # install dependencies
uv run pytest                                  # run tests
uv run pytest tests/test_foo.py::test_bar      # run a single test
uv run ruff check src/ experiments/            # lint
uv run ruff format --check src/ experiments/   # format check
uv run experiments/scenarios/<name>/run.py      # run a scenario
```

## Key Types

```python
# analysis.py — maps each transition name to the set of agents that can fire it
AgentMapping = dict[str, set[str]]
```

From pm4py:
- `PetriNet`, `Marking` — from `pm4py.objects.petri_net.obj`
- `PetriNet.Transition` — transition objects within a net
- `TransitionSystem` — from `pm4py.objects.petri_net.utils.reachability_graph`
- Stochastic map: `dict[PetriNet.Transition, RandomVariable]` — maps transitions to pm4py `RandomVariable` (weight, priority, distribution)

## Library API

### io.py — PNML Loading

```python
def load_pnml(path: str | Path) -> tuple[PetriNet, Marking, Marking]
```
Load a Petri net from a PNML file. Returns `(net, initial_marking, final_marking)`.

```python
def load_pnml_stochastic(path: str | Path) -> tuple[PetriNet, Marking, Marking, dict[Any, Any]]
```
Load a Petri net with stochastic information. Returns `(net, im, fm, stochastic_map)`.

### viz.py — Visualization

#### Petri net rendering

```python
def build_stochastic_decorations(stochastic_map: dict[Any, Any]) -> dict[Any, dict[str, str]]
```
Convert a stochastic map into a pm4py decorations dict (label with weight, blue color).

```python
def save_net_png(
    net: PetriNet, initial_marking: Marking, final_marking: Marking,
    output_path: str | Path, decorations: dict[Any, Any] | None = None,
) -> None
```
Save a Petri net visualization as PNG.

#### Comparative plot helpers

All plot functions below share a common signature pattern for multi-config, multi-agent, multi-index data:

- `config_labels: list[str]` — label per configuration
- `agent_labels: list[str]` — label per agent
- `index_powers: dict[str, list[list[float]]]` — `{index_name: [[power per agent] per config]}`
- `output_path: Path` — destination PNG
- `title: str = ""` — optional figure title

```python
def plot_power_bars(
    config_labels, agent_labels, index_powers, output_path, title="",
) -> None
```
Grouped bar chart of per-agent power across configurations (one subplot per index).

```python
def plot_power_heatmap(
    config_labels, agent_labels, index_powers, output_path, title="",
) -> None
```
Heatmap grid (rows = configs, columns = agents, color = power value; one subplot per index).

```python
def plot_index_correlation(
    config_labels, agent_labels, index_powers, output_path, title="",
) -> None
```
Pairwise scatter plots of index values; each point is one agent in one config, colored by configuration.

```python
def plot_lorenz_curves(
    config_labels, index_powers, output_path, title="",
) -> None
```
Lorenz curves per config (one subplot per index). Curves closer to the diagonal indicate more equal distribution.

```python
def plot_rank_agreement(
    config_labels, index_powers, output_path, title="",
) -> None
```
Spearman rank-correlation heatmaps between indices (one subplot per config).

```python
def plot_power_deltas(
    config_labels, agent_labels, index_powers, output_path,
    baseline_idx: int = 0, title="",
) -> None
```
Bar chart of power change from a baseline config for each agent and each index.

```python
def plot_granularity_scatter(
    labels, granularities: list[float], index_values: dict[str, list[float]],
    output_path, title="", ylabel="",
) -> None
```
Scatter plot of granularity (x-axis) vs per-index summary statistics (y-axis) with point annotations.

### analysis.py — Structural Analysis, Power Indices & Distributional Metrics

#### Structural utilities

```python
def incidence_matrix(net: PetriNet) -> np.ndarray
```
Compute the incidence matrix (rows=places, columns=transitions).

```python
def reachability_graph(net: PetriNet, initial_marking: Marking) -> TransitionSystem
```
Build the full reachability graph from an initial marking.

```python
def is_reachable_restricted(
    net: PetriNet, im: Marking, fm: Marking,
    allowed_transitions: set[PetriNet.Transition],
) -> bool
```
BFS reachability check using only allowed transitions. Uses covering semantics (`m >= fm`).

#### Coalition-based power indices

```python
def shapley_shubik(
    net: PetriNet, im: Marking, fm: Marking, agent_mapping: AgentMapping,
) -> dict[str, float]
```
Shapley-Shubik power index. `phi_i = sum_{S⊆N\{i}} [|S|!(n-|S|-1)!/n!] * (v(S∪{i}) - v(S))`.

```python
def banzhaf(
    net: PetriNet, im: Marking, fm: Marking, agent_mapping: AgentMapping,
    *, normalized: bool = True,
) -> dict[str, float]
```
Banzhaf power index. Raw swing count normalized by default. `eta_i = sum_{S⊆N\{i}} (v(S∪{i}) - v(S))`.

#### Pre-computed characteristic function variants

```python
def shapley_shubik_from_values(
    agents: list[str],
    v: dict[frozenset[str], float],
) -> dict[str, float]
```
Shapley-Shubik index from a pre-computed continuous-valued characteristic function `v: 2^N → [0,1]`. Normalized to sum to 1.

```python
def banzhaf_from_values(
    agents: list[str],
    v: dict[frozenset[str], float],
    *, normalized: bool = True,
) -> dict[str, float]
```
Banzhaf index from a pre-computed continuous-valued characteristic function.

#### Path-based indices

```python
def usability(
    net: PetriNet, im: Marking, fm: Marking, agent_mapping: AgentMapping,
    *, normalized: bool = True, start_place: str | None = None,
) -> dict[str, float]
```
Usability index via prefix-based shared credit. For each simple path, all non-empty prefixes are treated as firing sequences. For each sequence of length *L*, every transition contributes `1/(L * k)` to each of the *k* agents that can fire it. Averaged across all sequences. Optional `start_place` overrides initial marking.

```python
def gatekeeper(
    net: PetriNet, im: Marking, fm: Marking, agent_mapping: AgentMapping,
    *, normalized: bool = True,
) -> dict[str, float]
```
Gatekeeper power index based on immediate dominators (idom) in the transition graph. `idom_count[t]` = number of transitions whose immediate dominator is `t`. For each simple path, each transition gets credit `idom_count[t] / (|T| * k)` shared among `k` capable agents. Averaged over all paths.

```python
def gatekeeper_reach(
    net: PetriNet, im: Marking, fm: Marking, agent_mapping: AgentMapping,
    *, normalized: bool = True,
) -> dict[str, float]
```
Reachability-weighted gatekeeper variant. Weight = `|R(m')|` (number of markings reachable from the marking after firing the transition). Replaces the positional `(L-p)/L` proxy with actual forward-reachability set size. Credit shared equally among capable agents, summed across all simple paths.

#### Distributional metrics

```python
def gini_coefficient(values: list[float]) -> float
```
Gini coefficient of a distribution (0 = perfectly equal, 1 = maximally unequal).

```python
def granularity(agent_mapping: AgentMapping) -> float
```
Gini index of the supply-degree distribution across transitions (how unevenly agents are distributed).

#### Internal helpers (not exported)

- `_resolve_transitions(net, agent_mapping)` — map transition names to `Transition` objects
- `_precompute_characteristic_function(net, im, fm, agent_mapping)` — enumerate all coalitions, compute characteristic function `v`
- `_all_simple_paths(net, im, fm, *, start_place=None)` — DFS for all simple paths (no repeated markings)
- `_reachability_set_size(net, marking)` — BFS count of all markings reachable from a given marking
- `_build_transition_graph(net, im)` — BFS to build directed transition graph with virtual root `_ROOT_`
- `_compute_idom(adj, root, transitions)` — iterative dominator computation, returns idom mapping

## Conventions

- Python 3.12+, managed with `uv`
- `ruff` for linting/formatting (line-length 99, rules: E, F, I, UP)
- Petri net types from `pm4py.objects.petri_net.obj`
- Experiment scripts use `from lib import ...` (package declared in `pyproject.toml`)
- No tests yet; `tests/` directory is configured in pyproject.toml
