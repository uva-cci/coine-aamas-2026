"""Fork-join scenario: power index analysis."""

import argparse
import sys
from functools import partial
from pathlib import Path

from statistics import median

from lib import (
    banzhaf,
    gatekeeper,
    gini_coefficient,
    granularity,
    load_pnml,
    plot_granularity_scatter,
    plot_index_correlation,
    plot_lorenz_curves,
    plot_power_bars,
    plot_power_deltas,
    plot_power_heatmap,
    plot_rank_agreement,
    shapley_shubik,
    usability,
)

SCENARIO_DIR = Path(__file__).parent

AGENTS = ["1", "2", "3"]
TRANSITION_ORDER = ["fork", "branch_a", "branch_b", "join"]
TRANSITION_LABELS: dict[str, str] = {
    "fork": "t_{fork}",
    "branch_a": "t_a",
    "branch_b": "t_b",
    "join": "t_{join}",
}

CONFIGS: dict[str, dict[str, set[str]]] = {
    "Even": {
        "fork": {"1", "2", "3"},
        "branch_a": {"1", "2", "3"},
        "branch_b": {"1", "2", "3"},
        "join": {"1", "2", "3"},
    },
    "Skewed": {
        "fork": {"1", "2", "3"},
        "branch_a": {"1", "2"},
        "branch_b": {"3"},
        "join": {"1", "2", "3"},
    },
}

INDEX_SPECS = [
    ("Usability", r"$U(a_i)$", partial(usability, start_place="p0")),
    ("Gatekeeper", r"$G(a_i)$", gatekeeper),
    ("Shapley-Shubik", r"$\phi_{a_i}$", shapley_shubik),
    ("Banzhaf", r"$\beta_{a_i}$", banzhaf),
]


def _agent_set_latex(agents: set[str]) -> str:
    """Format agent IDs as LaTeX set, e.g. {1,2} -> $\\{a_1, a_2\\}$."""
    inner = ", ".join(f"a_{a}" for a in sorted(agents))
    return rf"$\{{{inner}\}}$"


def _agent_set_md(agents: set[str]) -> str:
    """Format agent IDs for markdown, e.g. {1,2} -> {a1, a2}."""
    inner = ", ".join(f"a{a}" for a in sorted(agents))
    return "{" + inner + "}"


# ---------------------------------------------------------------------------
# Distribution table: which agents can fire which transitions
# ---------------------------------------------------------------------------


def format_distribution_latex() -> str:
    """Distribution of transitions over agents as LaTeX table."""
    n = len(TRANSITION_ORDER)
    t_headers = " & ".join(f"${TRANSITION_LABELS[t]}$" for t in TRANSITION_ORDER)

    lines = [
        r"\begin{table}[ht!]",
        r"    \centering",
        r"    \small",
        r"    \caption{Distribution of transitions over agents"
        r" in the fork-join scenarios.}",
        r"    \begin{tabular}{|l|" + "c|" * n + "}",
        r"    \hline",
        r"    \diagbox[width=1.8cm, height=0.7cm]"
        r"{\textbf{A}}{\textbf{T}} & " + t_headers + r" \\ \hline",
    ]

    for cfg_name, mapping in CONFIGS.items():
        cells = " & ".join(_agent_set_latex(mapping[t]) for t in TRANSITION_ORDER)
        lines.append(f"    {cfg_name} & {cells}" + r" \\ \hline")

    lines += [
        r"    \end{tabular}",
        r"    \label{tab:fork-join}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def format_distribution_markdown() -> str:
    """Distribution of transitions over agents as markdown table."""
    lines = [
        "| A \\ T | " + " | ".join(TRANSITION_ORDER) + " |",
        "|---|" + "|".join("---" for _ in TRANSITION_ORDER) + "|",
    ]
    for cfg_name, mapping in CONFIGS.items():
        cells = " | ".join(_agent_set_md(mapping[t]) for t in TRANSITION_ORDER)
        lines.append(f"| {cfg_name} | {cells} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Scores table: power index values per config
# ---------------------------------------------------------------------------


def format_scores_latex(results: list[tuple[str, dict]]) -> str:
    """Power index scores as LaTeX table (indices on rows, configs on columns)."""
    n_a = len(AGENTS)
    col_groups = " || ".join("c" * n_a for _ in results)

    multi = " & ".join(
        rf"\multicolumn{{{n_a}}}{{c}}{{{cfg_name}}}" for cfg_name, _ in results
    )
    agent_cols = " & ".join(f"$a_{a}$" for a in AGENTS)
    sub_header = " & ".join(agent_cols for _ in results)

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        rf"\begin{{tabular}}{{l {col_groups}}}",
        r"\hline",
        f"& {multi}" + r" \\",
        f"& {sub_header}" + r" \\ \hline",
    ]

    for idx_name, idx_lbl, _ in INDEX_SPECS:
        parts = []
        for _, scores in results:
            vals = scores[idx_name]
            parts.append(" & ".join(f"{vals[a]:.4f}" for a in AGENTS))
        lines.append(f"{idx_lbl} & " + " & ".join(parts) + r" \\")

    # Granularity row: one value per config, spanning all agent columns
    gran_parts = []
    for cfg_name, scores in results:
        g = scores["Granularity"]
        gran_parts.append(
            rf"\multicolumn{{{n_a}}}{{c}}{{{g:.4f}}}"
        )
    lines.append(r"\hline")
    lines.append(r"$\mathcal{G}$ & " + " & ".join(gran_parts) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Fork-Join Scenario: Power Index Comparison}",
        r"\label{tab:fork-join_scores}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def format_scores_markdown(results: list[tuple[str, dict]]) -> str:
    """Power index scores as markdown table (indices on rows, configs on columns)."""
    header_parts = ["Index"]
    for cfg_name, _ in results:
        for a in AGENTS:
            header_parts.append(f"{cfg_name} a{a}")

    lines = [
        "| " + " | ".join(header_parts) + " |",
        "|---|" + "|".join("---:" for _ in header_parts[1:]) + "|",
    ]

    for idx_name, _, _ in INDEX_SPECS:
        cells = []
        for _, scores in results:
            vals = scores[idx_name]
            cells.extend(f"{vals[a]:.4f}" for a in AGENTS)
        lines.append(f"| {idx_name} | " + " | ".join(cells) + " |")

    # Granularity row
    gran_cells = []
    for _, scores in results:
        g = scores["Granularity"]
        gran_cells.append(f"{g:.4f}")
        gran_cells.extend("" for _ in AGENTS[1:])
    lines.append(f"| Granularity | " + " | ".join(gran_cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fork-join scenario analysis")
    parser.add_argument(
        "--format",
        choices=["markdown", "latex"],
        default="markdown",
        help="Output table format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write tables to file instead of stdout",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate granularity vs inequality plot",
    )
    parser.add_argument(
        "--plot-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Plot output format (default: pdf)",
    )
    args = parser.parse_args()

    net, im, fm = load_pnml(SCENARIO_DIR / "fork_join.pnml")

    results: list[tuple[str, dict]] = []
    for name, agent_mapping in CONFIGS.items():
        print(f"Computing indices for {name}...", file=sys.stderr)
        scores = {idx_name: fn(net, im, fm, agent_mapping) for idx_name, _, fn in INDEX_SPECS}
        scores["Granularity"] = granularity(agent_mapping)
        results.append((name, scores))

    # Always save both formats to outputs/
    out_dir = SCENARIO_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)

    md_table = format_distribution_markdown() + "\n" + format_scores_markdown(results)
    tex_table = format_distribution_latex() + "\n" + format_scores_latex(results)

    (out_dir / "table.md").write_text(md_table, encoding="utf-8")
    (out_dir / "table.tex").write_text(tex_table, encoding="utf-8")
    print(f"Wrote table.md and table.tex to {out_dir}", file=sys.stderr)

    # Still support explicit --output / stdout for the chosen format
    chosen = tex_table if args.format == "latex" else md_table
    if args.output:
        args.output.write_text(chosen, encoding="utf-8")
        print(f"Wrote table to {args.output}", file=sys.stderr)
    else:
        print(chosen)

    if args.plot:
        labels = [name for name, _ in results]
        grans = [scores["Granularity"] for _, scores in results]
        ext = args.plot_format

        # Scatter plots: median and Gini vs granularity
        for stat_name, stat_fn, ylabel in [
            ("median_power", lambda v: median(v), "Median Power"),
            ("gini_power", lambda v: gini_coefficient(list(v)), "Power Inequality (Gini)"),
        ]:
            series: dict[str, list[float]] = {}
            for idx_name, _, _ in INDEX_SPECS:
                series[idx_name] = [
                    stat_fn(scores[idx_name].values()) for _, scores in results
                ]
            path = out_dir / f"granularity_vs_{stat_name}.{ext}"
            plot_granularity_scatter(labels, grans, series, path,
                                     title=f"Fork-Join: Granularity vs {ylabel}",
                                     ylabel=ylabel)
            print(f"Saved plot to {path}", file=sys.stderr)

        # Shared data for remaining plots
        agent_labels = [f"$a_{a}$" for a in AGENTS]
        index_powers: dict[str, list[list[float]]] = {}
        for idx_name, _, _ in INDEX_SPECS:
            index_powers[idx_name] = [
                [scores[idx_name][a] for a in AGENTS] for _, scores in results
            ]

        for name, fn in [
            ("power_bars", lambda p: plot_power_bars(
                labels, agent_labels, index_powers, p,
                title="Fork-Join: Power per Agent")),
            ("index_correlation", lambda p: plot_index_correlation(
                labels, agent_labels, index_powers, p,
                title="Fork-Join: Index Correlation")),
            ("power_heatmap", lambda p: plot_power_heatmap(
                labels, agent_labels, index_powers, p,
                title="Fork-Join: Power Heatmap")),
            ("lorenz_curves", lambda p: plot_lorenz_curves(
                labels, index_powers, p,
                title="Fork-Join: Lorenz Curves")),
            ("rank_agreement", lambda p: plot_rank_agreement(
                labels, index_powers, p,
                title="Fork-Join: Rank Agreement")),
            ("power_deltas", lambda p: plot_power_deltas(
                labels, agent_labels, index_powers, p, baseline_idx=0,
                title="Fork-Join: Power Deltas from Even")),
        ]:
            path = out_dir / f"{name}.{ext}"
            fn(path)
            print(f"Saved plot to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
