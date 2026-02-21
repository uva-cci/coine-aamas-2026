"""Football scenario: load, visualize, and compute power indices."""

import argparse
import base64
import io
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from statistics import median

import graphviz
import numpy as np
from PIL import Image

from lib import (
    banzhaf_from_values,
    gatekeeper,
    gini_coefficient,
    granularity,
    load_pnml_stochastic,
    plot_granularity_scatter,
    plot_index_correlation,
    plot_lorenz_curves,
    plot_power_bars,
    plot_power_deltas,
    plot_power_heatmap,
    plot_rank_agreement,
    shapley_shubik_from_values,
    usability,
)

SCENARIO_DIR = Path(__file__).parent

# --- Formations: name → (n_def, n_mid, n_att) ---
FORMATIONS = {
    "1-2-5-3": (2, 5, 3),
    "1-4-4-2": (4, 4, 2),
    "1-3-5-2": (3, 5, 2),
    "1-4-3-3": (4, 3, 3),
    "1-1-1-1": (1, 1, 1),
    "1-4-4-4": (4, 4, 4),
}

# Zone ordering: left-to-right on the pitch
ZONE_ORDER = ["Defense", "Midfield", "Attack", "Goal"]

# Stochastic model parameters (must match generate_pnml.py)
M = 10  # cells per zone
P_GOAL = 3  # goal width in cells
GAMMA = 0.99  # discount factor for stochastic CF

# Display labels for each power index: (markdown, latex)
INDEX_LABELS: dict[str, tuple[str, str]] = {
    "Shapley-Shubik": ("Shapley-Shubik", r"$\phi_{a_i}$"),
    "Banzhaf": ("Banzhaf", r"$\beta_{a_i}$"),
    "Usability": ("Usability", r"$U(a_i)$"),
    "Gatekeeper": ("Gatekeeper", r"$G(a_i)$"),
}


def stochastic_value(
    d: int, m: int, a: int, M: int, P_GOAL: int, gamma: float
) -> float:
    """Expected discounted hitting time E[gamma^T] for a 3-state absorbing chain.

    States: Defense (0), Midfield (1), Attack (2) → Goal (absorbing).
    Each transition probability is the geometric mean of the passer and
    receiver contributions, so all three roles affect the chain:
      Defense→Midfield:  sqrt(d·m) / M
      Midfield→Attack:   sqrt(m·a) / M
      Attack→Goal:       sqrt(a·P_GOAL) / M
    Fail transitions loop back to Defense.

    Returns 0.0 if any role is missing from the coalition.
    """
    if d == 0 or m == 0 or a == 0:
        return 0.0

    p1 = (d * m) ** 0.5 / M  # Defense → Midfield
    p2 = (m * a) ** 0.5 / M  # Midfield → Attack
    p3 = (a * P_GOAL) ** 0.5 / M  # Attack → Goal

    # Solve V = gamma * P * V + gamma * p_absorb for each state.
    # V[0] = gamma * (p1 * V[1] + (1-p1) * V[0])
    # V[1] = gamma * (p2 * V[2] + (1-p2) * V[0])
    # V[2] = gamma * (p3 * 1    + (1-p3) * V[0])
    A = np.array([
        [1 - gamma * (1 - p1), -gamma * p1,         0.0],
        [-gamma * (1 - p2),     1.0,                -gamma * p2],
        [-gamma * (1 - p3),     0.0,                 1.0],
    ])
    b = np.array([0.0, 0.0, gamma * p3])
    V = np.linalg.solve(A, b)
    return float(V[0])


def build_stochastic_cf(
    agent_mapping: dict[str, set[str]],
    M: int,
    P_GOAL: int,
    gamma: float,
) -> tuple[list[str], dict[frozenset[str], float]]:
    """Build a stochastic characteristic function for all coalitions.

    For each coalition, count (d, m, a) from the agent mapping and compute
    stochastic_value.  Returns (agents, v) ready for *_from_values functions.
    """
    agents = sorted({a for s in agent_mapping.values() for a in s})
    n = len(agents)

    # Precompute role sets
    defenders = agent_mapping["pass_def_mid"]
    midfielders = agent_mapping["pass_mid_att"]
    attackers = agent_mapping["shoot"]

    v: dict[frozenset[str], float] = {}
    for size in range(n + 1):
        for combo in combinations(agents, size):
            coalition = frozenset(combo)
            d_count = len(defenders & coalition)
            m_count = len(midfielders & coalition)
            a_count = len(attackers & coalition)
            v[coalition] = stochastic_value(d_count, m_count, a_count, M, P_GOAL, gamma)

    return agents, v


def _index_label(name: str, fmt: str) -> str:
    """Return the display label for an index name given the output format."""
    col = 1 if fmt == "latex" else 0
    if name in INDEX_LABELS:
        return INDEX_LABELS[name][col]
    return name


def _build_dot(net, im, fm, smap) -> graphviz.Digraph:
    """Build a graphviz Digraph with zones strictly left-to-right.

    Backward (fail) edges use constraint=false so they don't pull
    Defense rightward.
    """
    weights = {t.name: rv.get_weight() for t, rv in smap.items()}

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", bgcolor="transparent", nodesep="0.6", ranksep="1.0")
    dot.attr("edge", penwidth="2.0")

    # Places
    initial_places = {p.name for p in im}
    final_places = {p.name for p in fm}
    for p in net.places:
        shape = "doublecircle" if p.name in final_places else "circle"
        label = p.name
        if p.name in initial_places:
            label += "\n\u25cf"
        dot.node(
            p.name,
            label=label,
            shape=shape,
            width="0.8",
            fixedsize="false",
            style="filled",
            fillcolor="white",
        )

    # Transitions
    for t in net.transitions:
        w = weights.get(t.name, "")
        label = f"{t.name}\nw={w}"
        dot.node(
            t.name,
            label=label,
            shape="box",
            style="filled",
            fillcolor="#AAAAFF",
        )

    # Edges
    zone_rank = {name: i for i, name in enumerate(ZONE_ORDER)}
    for arc in net.arcs:
        src = arc.source.name
        tgt = arc.target.name
        attrs = {}
        if isinstance(arc.source, type(list(net.transitions)[0])):
            t_obj = arc.source
            input_place = None
            for a2 in net.arcs:
                if a2.target == t_obj and a2.source in net.places:
                    input_place = a2.source.name
                    break
            if (
                input_place
                and tgt in zone_rank
                and input_place in zone_rank
                and zone_rank[tgt] <= zone_rank[input_place]
            ):
                attrs["constraint"] = "false"
        dot.edge(src, tgt, **attrs)

    return dot


def render_football_net(net, im, fm, smap, output_path: Path) -> None:
    """Render the Petri net to the requested format (png, svg, pdf)."""
    dot = _build_dot(net, im, fm, smap)
    fmt = output_path.suffix.lstrip(".")
    dot.format = fmt
    if fmt == "png":
        dot.attr(dpi="300")
    dot.render(str(output_path.with_suffix("")), cleanup=True)


def render_football_net_svg(net, im, fm, smap) -> str:
    """Render the Petri net and return the SVG string."""
    dot = _build_dot(net, im, fm, smap)
    return dot.pipe(format="svg").decode("utf-8")


def composite_with_background(
    net_svg: str,
    background_jpg: Path,
    output_path: Path,
    bg_opacity: float = 0.3,
) -> None:
    """Create an SVG overlay: faded raster background + vector Petri net.

    The background image is auto-cropped to the playing field (green area).
    The Petri net is scaled to fit and centred vertically.
    All Petri net elements (text, lines, shapes) remain vector.
    """
    # Auto-crop to the green playing field
    bg = Image.open(background_jpg)
    arr = np.array(bg)
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    green = (g > 120) & ((g - r) > 40) & ((g - b) > 40)
    field_rows = np.where(green.mean(axis=1) > 0.3)[0]
    field_cols = np.where(green.mean(axis=0) > 0.3)[0]
    margin = 5
    crop_box = (
        max(0, int(field_cols[0]) - margin),
        max(0, int(field_rows[0]) - margin),
        min(bg.width, int(field_cols[-1]) + 1 + margin),
        min(bg.height, int(field_rows[-1]) + 1 + margin),
    )
    bg = bg.crop(crop_box)
    bg_w, bg_h = bg.size

    # Encode the cropped image as a base64 data URI
    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Extract the Petri net SVG dimensions (in pt) from the <svg> tag
    m = re.search(
        r'<svg[^>]*\bwidth="([\d.]+)pt"[^>]*\bheight="([\d.]+)pt"',
        net_svg,
    )
    net_w_pt, net_h_pt = float(m.group(1)), float(m.group(2))

    # Scale factor: fit net width to background width
    scale = bg_w / net_w_pt
    net_scaled_h = net_h_pt * scale
    y_offset = (bg_h - net_scaled_h) / 2

    # Extract the inner content of the SVG (everything inside <svg ...> ... </svg>)
    inner_start = net_svg.index(">", net_svg.index("<svg")) + 1
    inner_end = net_svg.rindex("</svg>")
    svg_inner = net_svg[inner_start:inner_end]

    svg_out = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{bg_w}" height="{bg_h}"
     viewBox="0 0 {bg_w} {bg_h}">
  <!-- Faded background image -->
  <image href="data:image/jpeg;base64,{img_b64}"
         width="{bg_w}" height="{bg_h}"
         opacity="{bg_opacity}"
         preserveAspectRatio="xMidYMid slice"/>
  <!-- Vector Petri net, scaled to fit and centred vertically -->
  <g transform="translate(0,{y_offset}) scale({scale})">
    {svg_inner}
  </g>
</svg>"""

    output_path.write_text(svg_out, encoding="utf-8")


def build_agent_mapping(
    n_def: int,
    n_mid: int,
    n_att: int,
) -> dict[str, set[str]]:
    """Build agent mapping from formation.

    Each player controls the transitions of their zone:
    - Defenders fire pass_def_mid and fail_def
    - Midfielders fire pass_mid_att and fail_mid
    - Attackers fire shoot and fail_att
    """
    defenders = {f"D{i}" for i in range(1, n_def + 1)}
    midfielders = {f"M{i}" for i in range(1, n_mid + 1)}
    attackers = {f"A{i}" for i in range(1, n_att + 1)}

    return {
        "pass_def_mid": defenders,
        "fail_def": defenders,
        "pass_mid_att": midfielders,
        "fail_mid": midfielders,
        "shoot": attackers,
        "fail_att": attackers,
    }


def format_table_markdown(results: list[dict], index_names: list[str]) -> str:
    """Format a markdown table (indices on rows, formations on columns)."""
    role_prefixes = [prefix for prefix, _ in results[0]["roles"]]

    header_parts = ["Index"]
    for entry in results:
        display_name = entry["formation"].removeprefix("1-")
        for p in role_prefixes:
            header_parts.append(f"{display_name} {p}_i")

    lines = [
        "| " + " | ".join(header_parts) + " |",
        "|---|" + "|".join("---:" for _ in header_parts[1:]) + "|",
    ]

    for idx_name in index_names:
        label = _index_label(idx_name, "markdown")
        cells = []
        for entry in results:
            vals = entry["indices"][idx_name]
            cells.extend(f"{vals[f'{p}1']:.4f}" for p in role_prefixes)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Granularity row
    gran_cells = []
    for entry in results:
        gran_cells.append(f"{entry['granularity']:.4f}")
        gran_cells.extend("" for _ in role_prefixes[1:])
    lines.append("| Granularity | " + " | ".join(gran_cells) + " |")

    return "\n".join(lines) + "\n"


def format_table_latex(results: list[dict], index_names: list[str]) -> str:
    """Format a LaTeX table (indices on rows, formations on columns)."""
    role_prefixes = [prefix for prefix, _ in results[0]["roles"]]
    n_roles = len(role_prefixes)
    col_groups = " || ".join("c" * n_roles for _ in results)
    multi = " & ".join(
        rf"\multicolumn{{{n_roles}}}{{c}}{{{entry['formation'].removeprefix('1-')}}}"
        for entry in results
    )
    role_headers = " & ".join(f"${p}_i$" for p in role_prefixes)
    sub_header = " & ".join(role_headers for _ in results)

    lines = [
        rf"\begin{{tabular}}{{l {col_groups}}}",
        r"\toprule",
        f"& {multi}" + r" \\",
        f"& {sub_header}" + r" \\ \midrule",
    ]

    for idx_name in index_names:
        label = _index_label(idx_name, "latex")
        parts = []
        for entry in results:
            vals = entry["indices"][idx_name]
            parts.append(" & ".join(f"{vals[f'{p}1']:.4f}" for p in role_prefixes))
        lines.append(f"{label} & " + " & ".join(parts) + r" \\")

    # Granularity row
    gran_parts = []
    for entry in results:
        gran_parts.append(rf"\multicolumn{{{n_roles}}}{{c}}{{{entry['granularity']:.4f}}}")
    lines.append(r"\midrule")
    lines.append(r"$\mathcal{G}$ & " + " & ".join(gran_parts) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Football scenario analysis")
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
        help="Write table to file instead of stdout",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip PNG/PDF rendering (faster iteration)",
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

    net, im, fm, smap = load_pnml_stochastic(SCENARIO_DIR / "football.pnml")

    print("Stochastic weights:", file=sys.stderr)
    for t, rv in sorted(smap.items(), key=lambda x: x[0].name):
        print(f"  {t.name}: weight={rv.get_weight()}", file=sys.stderr)

    output_dir = SCENARIO_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    if not args.skip_viz:
        # High-DPI raster
        render_football_net(net, im, fm, smap, output_dir / "football.png")
        print(f"Saved visualization to {output_dir / 'football.png'}", file=sys.stderr)

        # Composite: vector Petri net over faded background
        bg_path = SCENARIO_DIR / "football.jpg"
        if bg_path.exists():
            net_svg = render_football_net_svg(net, im, fm, smap)
            svg_path = output_dir / "football_overlay.svg"
            composite_with_background(net_svg, bg_path, svg_path, bg_opacity=0.3)
            # Convert to PDF (vector Petri net preserved)
            pdf_path = output_dir / "football_overlay.pdf"
            subprocess.run(
                ["rsvg-convert", "-f", "pdf", str(svg_path), "-o", str(pdf_path)],
                check=True,
            )
            svg_path.unlink()
            print(f"Saved overlay to {pdf_path}", file=sys.stderr)

    # Compute power indices for each formation
    index_names = [
        "Usability",
        "Gatekeeper",
        "Shapley-Shubik",
        "Banzhaf",
    ]
    results: list[dict] = []

    for name, (n_def, n_mid, n_att) in FORMATIONS.items():
        print(f"Computing indices for {name}...", file=sys.stderr)
        agent_mapping = build_agent_mapping(n_def, n_mid, n_att)

        indices = {
            "Usability": usability(net, im, fm, agent_mapping, start_place="Defense"),
            "Gatekeeper": gatekeeper(net, im, fm, agent_mapping),
        }
        s_agents, s_v = build_stochastic_cf(agent_mapping, M, P_GOAL, GAMMA)
        indices["Shapley-Shubik"] = shapley_shubik_from_values(s_agents, s_v)
        indices["Banzhaf"] = banzhaf_from_values(s_agents, s_v)

        results.append(
            {
                "formation": name,
                "roles": [("D", n_def), ("M", n_mid), ("A", n_att)],
                "indices": indices,
                "granularity": granularity(agent_mapping),
            }
        )

    # Format and output table
    if args.format == "latex":
        table = format_table_latex(results, index_names)
    else:
        table = format_table_markdown(results, index_names)

    if args.output:
        args.output.write_text(table, encoding="utf-8")
        print(f"Wrote table to {args.output}", file=sys.stderr)
    else:
        print(table)

    if args.plot:
        labels = [e["formation"].removeprefix("1-") for e in results]
        grans = [e["granularity"] for e in results]
        ext = args.plot_format

        # Scatter plots: median and Gini vs granularity
        for stat_name, stat_fn, ylabel in [
            ("median_power", lambda v: median(v), "Median Power"),
            ("gini_power", lambda v: gini_coefficient(list(v)), "Power Inequality (Gini)"),
        ]:
            series: dict[str, list[float]] = {}
            for idx_name in index_names:
                series[idx_name] = [stat_fn(e["indices"][idx_name].values()) for e in results]
            path = output_dir / f"granularity_vs_{stat_name}.{ext}"
            plot_granularity_scatter(
                labels,
                grans,
                series,
                path,
                title=f"Football: Granularity vs {ylabel}",
                ylabel=ylabel,
            )
            print(f"Saved plot to {path}", file=sys.stderr)

        # Shared data for remaining plots (one representative per role)
        role_prefixes = ["D", "M", "A"]
        agent_labels = [f"${p}_i$" for p in role_prefixes]
        index_powers: dict[str, list[list[float]]] = {}
        for idx_name in index_names:
            index_powers[idx_name] = [
                [e["indices"][idx_name][f"{p}1"] for p in role_prefixes] for e in results
            ]

        for name, fn in [
            (
                "power_bars",
                lambda p: plot_power_bars(
                    labels, agent_labels, index_powers, p, title="Football: Power per Role"
                ),
            ),
            (
                "index_correlation",
                lambda p: plot_index_correlation(
                    labels, agent_labels, index_powers, p, title="Football: Index Correlation"
                ),
            ),
            (
                "power_heatmap",
                lambda p: plot_power_heatmap(
                    labels, agent_labels, index_powers, p, title="Football: Power Heatmap"
                ),
            ),
            (
                "lorenz_curves",
                lambda p: plot_lorenz_curves(
                    labels, index_powers, p, title="Football: Lorenz Curves"
                ),
            ),
            (
                "rank_agreement",
                lambda p: plot_rank_agreement(
                    labels, index_powers, p, title="Football: Rank Agreement"
                ),
            ),
            (
                "power_deltas",
                lambda p: plot_power_deltas(
                    labels,
                    agent_labels,
                    index_powers,
                    p,
                    baseline_idx=0,
                    title="Football: Power Deltas from 2-5-3",
                ),
            ),
        ]:
            path = output_dir / f"{name}.{ext}"
            fn(path)
            print(f"Saved plot to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
