"""Sequential scenario: power index analysis."""

import argparse
import sys
from functools import partial
from pathlib import Path

from lib import banzhaf, gatekeeper, load_pnml, shapley_shubik, usability

SCENARIO_DIR = Path(__file__).parent

AGENTS = ["1", "2", "3"]
TRANSITION_ORDER = ["t0", "t1", "t2"]
TRANSITION_LABELS: dict[str, str] = {
    "t0": "t_0",
    "t1": "t_1",
    "t2": "t_2",
}

CONFIGS: dict[str, dict[str, set[str]]] = {
    "Even": {
        "t0": {"1", "2", "3"},
        "t1": {"1", "2", "3"},
        "t2": {"1", "2", "3"},
    },
    "Bottleneck": {
        "t0": {"1", "2", "3"},
        "t1": {"1", "2", "3"},
        "t2": {"1"},
    },
    "Funnel": {
        "t0": {"1", "2", "3"},
        "t1": {"1", "2"},
        "t2": {"1"},
    },
}

INDEX_SPECS = [
    ("Shapley-Shubik", r"$\phi_{a_i}$", shapley_shubik),
    ("Banzhaf", r"$\beta_{a_i}$", banzhaf),
    ("Usability", r"$U(a_i)$", partial(usability, start_place="p0")),
    ("Gatekeeper", r"$G(a_i)$", gatekeeper),
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
        r" in the sequential scenarios.}",
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
        r"    \label{tab:sequential}",
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
    """Power index scores as LaTeX table."""
    n_a = len(AGENTS)
    col_groups = " || ".join("c" * n_a for _ in INDEX_SPECS)

    multi = " & ".join(rf"\multicolumn{{{n_a}}}{{c}}{{{lbl}}}" for _, lbl, _ in INDEX_SPECS)
    agent_cols = " & ".join(f"$a_{a}$" for a in AGENTS)
    sub_header = " & ".join(agent_cols for _ in INDEX_SPECS)

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        rf"\begin{{tabular}}{{l {col_groups}}}",
        r"\hline",
        f"& {multi}" + r" \\",
        f"$A$ & {sub_header}" + r" \\ \hline",
    ]

    for cfg_name, scores in results:
        parts = []
        for idx_name, _, _ in INDEX_SPECS:
            vals = scores[idx_name]
            parts.append(" & ".join(f"{vals[a]:.4f}" for a in AGENTS))
        lines.append(f"{cfg_name} & " + " & ".join(parts) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Sequential Scenario: Power Index Comparison}",
        r"\label{tab:sequential_scores}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def format_scores_markdown(results: list[tuple[str, dict]]) -> str:
    """Power index scores as markdown table."""
    header_parts = ["A"]
    for idx_name, _, _ in INDEX_SPECS:
        for a in AGENTS:
            header_parts.append(f"{idx_name} a{a}")

    lines = [
        "| " + " | ".join(header_parts) + " |",
        "|---|" + "|".join("---:" for _ in header_parts[1:]) + "|",
    ]

    for cfg_name, scores in results:
        cells = []
        for idx_name, _, _ in INDEX_SPECS:
            vals = scores[idx_name]
            cells.extend(f"{vals[a]:.4f}" for a in AGENTS)
        lines.append(f"| {cfg_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential scenario analysis")
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
    args = parser.parse_args()

    net, im, fm = load_pnml(SCENARIO_DIR / "sequential.pnml")

    results: list[tuple[str, dict]] = []
    for name, agent_mapping in CONFIGS.items():
        print(f"Computing indices for {name}...", file=sys.stderr)
        scores = {idx_name: fn(net, im, fm, agent_mapping) for idx_name, _, fn in INDEX_SPECS}
        results.append((name, scores))

    if args.format == "latex":
        output = format_distribution_latex() + "\n" + format_scores_latex(results)
    else:
        output = format_distribution_markdown() + "\n" + format_scores_markdown(results)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote tables to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
