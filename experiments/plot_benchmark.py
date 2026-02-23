"""Visualize hyperfine benchmark results for power index scaling.

Reads JSON files produced by benchmark.sh and generates:
  1. benchmark_scaling.{ext}  — line plot with error bands (log scale)
  2. benchmark_boxplots.{ext} — 2x2 box plots (one per index)
  3. benchmark_table.tex      — LaTeX table with mean, stddev, min, max

Usage:
    uv run experiments/plot_benchmark.py [--format pdf|png]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"

INDEX_NAMES = ["usability", "gatekeeper", "shapley-shubik", "banzhaf"]
INDEX_DISPLAY = {
    "usability": "Usability",
    "gatekeeper": "Gatekeeper",
    "shapley-shubik": "Shapley-Shubik",
    "banzhaf": "Banzhaf",
}
INDEX_COLORS = {
    "usability": "#2ca02c",
    "gatekeeper": "#1f77b4",
    "shapley-shubik": "#d62728",
    "banzhaf": "#ff7f0e",
}

# Formation → |A| mapping
FORMATION_AGENTS = {
    "1-1-1": 6,
    "2-1-1": 8,
    "3-1-1": 10,
    "4-1-1": 12,
    "4-2-1": 14,
    "4-3-1": 16,
    "4-4-1": 18,
    "4-4-2": 20,
}


def load_results(index_name: str) -> dict | None:
    path = RESULTS_DIR / f"{index_name}.json"
    if not path.exists():
        print(f"Warning: {path} not found, skipping {index_name}")
        return None
    with open(path) as f:
        return json.load(f)


def extract_formation(command: str) -> str:
    """Extract formation string from hyperfine command."""
    parts = command.split("--formation")
    if len(parts) >= 2:
        return parts[1].strip().split()[0]
    return "?"


def _fmt_cell(mean: float, stddev: float) -> str:
    """Format mean ± stddev as a single compact cell (unitless, seconds)."""
    if mean < 1.0:
        return f"${mean:.2f}" + r" \pm " + f"{stddev:.2f}$"
    if mean < 10.0:
        return f"${mean:.1f}" + r" \pm " + f"{stddev:.1f}$"
    return f"${mean:.0f}" + r" \pm " + f"{stddev:.0f}$"


def generate_latex_table(all_data: dict[str, list[dict]]) -> str:
    """Generate a LaTeX table: |A| as rows, indices as columns.

    Each cell shows mean ± stddev in seconds (unit stated once in header).
    """
    # Collect all formations across indices (ordered by |A|)
    formations_seen: dict[str, int] = {}
    for idx_name in INDEX_NAMES:
        if idx_name not in all_data:
            continue
        for r in all_data[idx_name]:
            form = r["parameters"]["formation"]
            if form not in formations_seen:
                formations_seen[form] = FORMATION_AGENTS.get(form, 0)

    formations = sorted(formations_seen.keys(), key=lambda f: formations_seen[f])

    # Build lookup: (index, formation) → result dict
    lookup: dict[tuple[str, str], dict] = {}
    for idx_name in INDEX_NAMES:
        if idx_name not in all_data:
            continue
        for r in all_data[idx_name]:
            form = r["parameters"]["formation"]
            lookup[(idx_name, form)] = r

    active_indices = [i for i in INDEX_NAMES if i in all_data]
    n_idx = len(active_indices)
    col_spec = " | ".join(["r"] * n_idx)

    lines = [
        f"\\begin{{tabular}}{{l@{{\\quad}} | {col_spec}}}",
        "\\toprule",
    ]

    # Header row
    header_parts = ["$|\\mathcal{A}|$"]
    for idx_name in active_indices:
        header_parts.append(f"\\textbf{{{INDEX_DISPLAY[idx_name]}}}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # One row per formation
    for form in formations:
        n_agents = formations_seen[form]
        row_parts = [str(n_agents)]
        for idx_name in active_indices:
            r = lookup.get((idx_name, form))
            if r is None:
                row_parts.append("--")
            else:
                row_parts.append(_fmt_cell(r["mean"], r["stddev"]))
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--format", choices=["pdf", "png"], default="pdf", help="Output format (default: pdf)"
    )
    args = parser.parse_args()
    ext = args.format

    # Load all results
    all_data: dict[str, list[dict]] = {}
    for idx_name in INDEX_NAMES:
        data = load_results(idx_name)
        if data is not None:
            all_data[idx_name] = data["results"]

    if not all_data:
        print("No benchmark results found. Run benchmark.sh first.")
        return

    # --- Plot 1: Line plot with error bands ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx_name in INDEX_NAMES:
        if idx_name not in all_data:
            continue
        results = all_data[idx_name]
        agents_list = []
        means = []
        stddevs = []
        for r in results:
            formation = extract_formation(r["command"])
            n_agents = FORMATION_AGENTS.get(formation, 0)
            if n_agents == 0:
                continue
            agents_list.append(n_agents)
            means.append(r["mean"])
            stddevs.append(r["stddev"])

        agents_arr = np.array(agents_list)
        means_arr = np.array(means)
        stddevs_arr = np.array(stddevs)
        color = INDEX_COLORS[idx_name]

        ax.plot(
            agents_arr,
            means_arr,
            "o-",
            color=color,
            label=INDEX_DISPLAY[idx_name],
            markersize=4,
        )
        ax.fill_between(
            agents_arr,
            means_arr - stddevs_arr,
            means_arr + stddevs_arr,
            alpha=0.2,
            color=color,
        )

    ax.set_yscale("log")
    ax.set_xlabel("|A| (number of agents)")
    ax.set_ylabel("Time (s)")
    # No title — captions go in LaTeX
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(6, 22, 2)))

    output_path = RESULTS_DIR / f"benchmark_scaling.{ext}"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")

    # --- Plot 2: Box plots (2x2 grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax_idx, idx_name in enumerate(INDEX_NAMES):
        ax = axes_flat[ax_idx]
        if idx_name not in all_data:
            ax.set_title(f"{INDEX_DISPLAY[idx_name]} (no data)")
            continue

        results = all_data[idx_name]
        box_data = []
        tick_labels = []
        for r in results:
            formation = extract_formation(r["command"])
            n_agents = FORMATION_AGENTS.get(formation, 0)
            if n_agents == 0:
                continue
            box_data.append(r["times"])
            tick_labels.append(str(n_agents))

        color = INDEX_COLORS[idx_name]
        bp = ax.boxplot(
            box_data,
            tick_labels=tick_labels,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        for median_line in bp["medians"]:
            median_line.set_color(color)
            median_line.set_linewidth(2)

        ax.set_xlabel("|A|")
        ax.set_ylabel("Time (s)")
        ax.set_title(INDEX_DISPLAY[idx_name])
        ax.grid(True, alpha=0.3, axis="y")

    output_path = RESULTS_DIR / f"benchmark_boxplots.{ext}"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

    # --- LaTeX table ---
    table = generate_latex_table(all_data)
    table_path = RESULTS_DIR / "benchmark_table.tex"
    table_path.write_text(table, encoding="utf-8")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
