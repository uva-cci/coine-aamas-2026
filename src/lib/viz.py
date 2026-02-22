"""Visualization helpers for Petri nets."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet


def build_stochastic_decorations(
    stochastic_map: dict[Any, Any],
) -> dict[Any, dict[str, str]]:
    """Build a decorations dict labelling each transition with its weight.

    Accepts a stochastic_map as returned by ``load_pnml_stochastic``.
    Returns a dict suitable for pm4py's ``decorations`` parameter.
    """
    decorations: dict[Any, dict[str, str]] = {}
    for transition, rv in stochastic_map.items():
        weight = rv.get_weight()
        decorations[transition] = {
            "label": f"{transition.label}\nw={weight}",
            "color": "#AAAAFF",
        }
    return decorations


def save_net_png(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    output_path: str | Path,
    decorations: dict[Any, Any] | None = None,
) -> None:
    """Save a Petri net visualization as PNG."""
    pm4py.save_vis_petri_net(
        net, initial_marking, final_marking, str(output_path), decorations=decorations
    )


_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_granularity_scatter(
    labels: list[str],
    granularities: list[float],
    index_values: dict[str, list[float]],
    output_path: Path,
    title: str = "",
    ylabel: str = "",
) -> None:
    """Scatter plot: granularity (x) vs a per-index summary statistic (y).

    Parameters
    ----------
    labels:        config names (for point annotations)
    granularities: x values (one per config)
    index_values:  {index_name: [y_value_per_config]}
    output_path:   where to save (PDF/PNG)
    title:         optional plot title
    ylabel:        y-axis label
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (idx_name, vals) in enumerate(index_values.items()):
        color = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.scatter(granularities, vals, marker=marker, color=color, label=idx_name,
                   s=60, zorder=3)

    # Annotate points with config labels; offset vertically for duplicate x
    first_vals = next(iter(index_values.values()))
    bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray",
                alpha=0.8, linewidth=0.5)
    dup_count: dict[float, int] = {}
    for j, lbl in enumerate(labels):
        g = granularities[j]
        k = dup_count.get(g, 0)
        dup_count[g] = k + 1
        y_off = 8 + k * 14
        ax.annotate(lbl, (granularities[j], first_vals[j]),
                    textcoords="offset points", xytext=(6, y_off),
                    fontsize=7, fontweight="bold", color="#333333", bbox=bbox)

    ax.set_xlabel("Granularity ($\\mathcal{G}$)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fmt = Path(output_path).suffix.lstrip(".")
    fig.savefig(output_path, format=fmt, dpi=300)
    plt.close(fig)


def plot_power_bars(
    config_labels: list[str],
    agent_labels: list[str],
    index_powers: dict[str, list[list[float]]],
    output_path: Path,
    title: str = "",
) -> None:
    """Grouped bar chart of per-agent power across configurations.

    Parameters
    ----------
    config_labels: names for x-axis groups (one per configuration)
    agent_labels:  names for bars within each group (one per agent/role)
    index_powers:  {index_name: [[power_per_agent] for each config]}
    output_path:   where to save (PDF/PNG)
    title:         optional suptitle
    """
    n_indices = len(index_powers)
    n_configs = len(config_labels)
    n_agents = len(agent_labels)

    fig, axes = plt.subplots(1, n_indices, figsize=(4 * n_indices, 4), sharey=True)
    if n_indices == 1:
        axes = [axes]

    x = np.arange(n_configs)
    width = 0.8 / n_agents

    for ax, (idx_name, powers) in zip(axes, index_powers.items()):
        for a_i, a_label in enumerate(agent_labels):
            vals = [powers[c][a_i] for c in range(n_configs)]
            offset = (a_i - (n_agents - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=a_label,
                   color=_COLORS[a_i % len(_COLORS)])
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, fontsize=8)
        ax.set_title(idx_name, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Power")
    axes[0].legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fmt = Path(output_path).suffix.lstrip(".")
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save(fig: plt.Figure, output_path: Path, title: str) -> None:
    """Common save logic for all plots."""
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fmt = Path(output_path).suffix.lstrip(".")
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fractional_ranks(values: list[float]) -> np.ndarray:
    """1-based fractional ranks (average rank for ties)."""
    arr = np.array(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(arr)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    for val in np.unique(arr):
        mask = arr == val
        ranks[mask] = ranks[mask].mean()
    return ranks


# ── 1. Index correlation scatter ──────────────────────────────────────────


def plot_index_correlation(
    config_labels: list[str],
    agent_labels: list[str],
    index_powers: dict[str, list[list[float]]],
    output_path: Path,
    title: str = "",
) -> None:
    """Pairwise scatter of index values; each point is one agent in one config.

    Points are colored by configuration.
    A dashed diagonal shows perfect agreement between two indices.
    """
    names = list(index_powers.keys())
    n = len(names)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // ncols, idx % ncols]
        for c, cfg in enumerate(config_labels):
            x = index_powers[names[i]][c]
            y = index_powers[names[j]][c]
            ax.scatter(x, y, s=40, alpha=0.8, color=_COLORS[c % len(_COLORS)],
                       label=cfg, zorder=3)
        all_v = [v for nm in [names[i], names[j]]
                 for cfg_vals in index_powers[nm] for v in cfg_vals]
        lo, hi = min(all_v), max(all_v)
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel(names[i], fontsize=9)
        ax.set_ylabel(names[j], fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7)
    _save(fig, output_path, title)


# ── 2. Power heatmap ─────────────────────────────────────────────────────


def plot_power_heatmap(
    config_labels: list[str],
    agent_labels: list[str],
    index_powers: dict[str, list[list[float]]],
    output_path: Path,
    title: str = "",
) -> None:
    """Heatmap grid: rows = configs, cols = agents, color = power value.

    One subplot per power index, arranged in a 2-column grid.
    """
    n_indices = len(index_powers)
    ncols = min(2, n_indices)
    nrows = (n_indices + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 + 1.2 * len(agent_labels),
                             2.5 * nrows), squeeze=False)

    for idx, (idx_name, powers) in enumerate(index_powers.items()):
        ax = axes[idx // ncols, idx % ncols]
        data = np.array(powers)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(agent_labels)))
        ax.set_xticklabels(agent_labels, fontsize=8)
        ax.set_yticks(range(len(config_labels)))
        ax.set_yticklabels(config_labels, fontsize=8)
        ax.set_title(idx_name, fontsize=10)
        for r in range(len(config_labels)):
            for c in range(len(agent_labels)):
                rgba = im.cmap(im.norm(data[r, c]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(c, r, f"{data[r, c]:.3f}", ha="center", va="center",
                        fontsize=7, color="black" if lum > 0.5 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)

    for idx in range(n_indices, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    _save(fig, output_path, title)


# ── 3. Lorenz curves ─────────────────────────────────────────────────────


def plot_lorenz_curves(
    config_labels: list[str],
    index_powers: dict[str, list[list[float]]],
    output_path: Path,
    title: str = "",
) -> None:
    """Lorenz curve per config (one subplot per index).

    The closer a curve is to the diagonal, the more equal the distribution.
    """
    n_indices = len(index_powers)
    ncols = min(2, n_indices)
    nrows = (n_indices + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, (idx_name, powers) in enumerate(index_powers.items()):
        ax = axes[idx // ncols, idx % ncols]
        for c, cfg in enumerate(config_labels):
            vals = sorted(powers[c])
            n = len(vals)
            total = sum(vals)
            if total == 0:
                continue
            cum = np.cumsum(vals) / total
            xs = np.arange(1, n + 1) / n
            ax.plot(np.concatenate([[0], xs]), np.concatenate([[0], cum]),
                    marker="o", markersize=4, label=cfg,
                    color=_COLORS[c % len(_COLORS)])
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Cumulative share of agents", fontsize=9)
        ax.set_ylabel("Cumulative share of power", fontsize=9)
        ax.set_title(idx_name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_indices, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    _save(fig, output_path, title)


# ── 4. Rank agreement (Spearman correlation heatmaps) ────────────────────


def plot_rank_agreement(
    config_labels: list[str],
    index_powers: dict[str, list[list[float]]],
    output_path: Path,
    title: str = "",
) -> None:
    """Spearman rank-correlation matrix between indices, one heatmap per config."""
    names = list(index_powers.keys())
    n_idx = len(names)
    n_configs = len(config_labels)

    fig, axes = plt.subplots(1, n_configs, figsize=(3.5 * n_configs, 3.5),
                             squeeze=False)

    for c, cfg in enumerate(config_labels):
        ax = axes[0, c]
        corr = np.ones((n_idx, n_idx))
        for i in range(n_idx):
            for j in range(i + 1, n_idx):
                ri = _fractional_ranks(index_powers[names[i]][c])
                rj = _fractional_ranks(index_powers[names[j]][c])
                cc = np.corrcoef(ri, rj)[0, 1]
                val = cc if np.isfinite(cc) else 1.0
                corr[i, j] = corr[j, i] = val

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_idx))
        ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n_idx))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(cfg, fontsize=10)
        for i in range(n_idx):
            for j in range(n_idx):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)

    _save(fig, output_path, title)


# ── 5. Power deltas from baseline ────────────────────────────────────────


def plot_power_deltas(
    config_labels: list[str],
    agent_labels: list[str],
    index_powers: dict[str, list[list[float]]],
    output_path: Path,
    baseline_idx: int = 0,
    title: str = "",
) -> None:
    """Bar chart of power change from a baseline config, per agent per index."""
    n_indices = len(index_powers)
    n_agents = len(agent_labels)
    non_base = [(i, cfg) for i, cfg in enumerate(config_labels) if i != baseline_idx]
    n_non_base = len(non_base)

    fig, axes = plt.subplots(1, n_indices, figsize=(4 * n_indices, 4), sharey=True)
    if n_indices == 1:
        axes = [axes]

    x = np.arange(n_agents)
    width = 0.8 / max(n_non_base, 1)

    for ax, (idx_name, powers) in zip(axes, index_powers.items()):
        baseline = powers[baseline_idx]
        for k, (c_idx, cfg) in enumerate(non_base):
            deltas = [powers[c_idx][a] - baseline[a] for a in range(n_agents)]
            offset = (k - (n_non_base - 1) / 2) * width
            ax.bar(x + offset, deltas, width, label=f"→ {cfg}",
                   color=_COLORS[(k + 1) % len(_COLORS)])
        ax.set_xticks(x)
        ax.set_xticklabels(agent_labels, fontsize=8)
        ax.set_title(idx_name, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(f"$\\Delta$ Power (from {config_labels[baseline_idx]})")
    axes[0].legend(fontsize=7)
    _save(fig, output_path, title)
