"""Two-team adversarial football scenario: load, visualize, and compute power indices."""

import argparse
import base64
import io
import re
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

import graphviz
import numpy as np
from generate_pnml import FINAL_PLACE, INITIAL_PLACE, PLACES, build_edges, build_pnml
from PIL import Image

from pm4py.objects.petri_net.obj import Marking

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

M = 10
P_GOAL = 3
GAMMA = 0.99

# Curated matchups: (formation_A, formation_B) as (n1, n2, n3) tuples
MATCHUPS: list[tuple[str, tuple[int, int, int], str, tuple[int, int, int]]] = [
    ("2-5-3", (2, 5, 3), "4-3-3", (4, 3, 3)),
    ("4-3-3", (4, 3, 3), "4-3-3", (4, 3, 3)),
    ("4-4-2", (4, 4, 2), "2-5-3", (2, 5, 3)),
    ("3-5-2", (3, 5, 2), "4-4-2", (4, 4, 2)),
    ("4-3-3", (4, 3, 3), "3-5-2", (3, 5, 2)),
]

INDEX_LABELS: dict[str, tuple[str, str]] = {
    "Shapley-Shubik": ("Shapley-Shubik", r"$\phi_{a_i}$"),
    "Banzhaf": ("Banzhaf", r"$\beta_{a_i}$"),
    "Usability": ("Usability", r"$U(a_i)$"),
    "Gatekeeper": ("Gatekeeper", r"$G(a_i)$"),
}

# --- Physical zone columns (left to right on the pitch) ---
# Zone 1: Defense_A, Attack_B  (near Goal_A)
# Zone 2: Midfield_A, Midfield_B  (center)
# Zone 3: Attack_A, Defense_B  (near Goal_B)
ZONE_COLUMNS = [
    ["Goal_A"],
    ["Defense_A", "Attack_B"],
    ["Midfield_A", "Midfield_B"],
    ["Attack_A", "Defense_B"],
    ["Goal_B"],
]

TEAM_A_PLACES = {"Defense_A", "Midfield_A", "Attack_A", "Goal_A"}
TEAM_B_PLACES = {"Defense_B", "Midfield_B", "Attack_B", "Goal_B"}

TEAM_A_TRANSITIONS = {
    "pass_def_mid_A",
    "fail_def_A",
    "pass_mid_att_A",
    "fail_mid_A",
    "shoot_A",
    "fail_att_A",
}
TEAM_B_TRANSITIONS = {
    "pass_def_mid_B",
    "fail_def_B",
    "pass_mid_att_B",
    "fail_mid_B",
    "shoot_B",
    "fail_att_B",
}

FAIL_TRANSITIONS = {
    "fail_def_A",
    "fail_mid_A",
    "fail_att_A",
    "fail_def_B",
    "fail_mid_B",
    "fail_att_B",
}


def build_agent_mapping_2team(
    n1_a: int,
    n2_a: int,
    n3_a: int,
    n1_b: int,
    n2_b: int,
    n3_b: int,
) -> dict[str, set[str]]:
    """Build agent mapping for a two-team match.

    Team A agents: DA1..DA_n, MA1..MA_n, AA1..AA_n
    Team B agents: DB1..DB_n, MB1..MB_n, AB1..AB_n
    """
    da = {f"DA{i}" for i in range(1, n1_a + 1)}
    ma = {f"MA{i}" for i in range(1, n2_a + 1)}
    aa = {f"AA{i}" for i in range(1, n3_a + 1)}
    db = {f"DB{i}" for i in range(1, n1_b + 1)}
    mb = {f"MB{i}" for i in range(1, n2_b + 1)}
    ab = {f"AB{i}" for i in range(1, n3_b + 1)}

    return {
        # Team A transitions
        "pass_def_mid_A": da,
        "fail_def_A": da,
        "pass_mid_att_A": ma,
        "fail_mid_A": ma,
        "shoot_A": aa,
        "fail_att_A": aa,
        # Team B transitions
        "pass_def_mid_B": db,
        "fail_def_B": db,
        "pass_mid_att_B": mb,
        "fail_mid_B": mb,
        "shoot_B": ab,
        "fail_att_B": ab,
    }


def stochastic_value_2team(
    da: int,
    ma: int,
    aa: int,
    db: int,
    mb: int,
    ab: int,
    m: int,
    p_goal: int,
    gamma: float,
    *,
    team: str = "A",
) -> float:
    """Expected discounted probability of *team* scoring in the 2-team model.

    6 transient states: Defense_A(0), Midfield_A(1), Attack_A(2),
                        Defense_B(3), Midfield_B(4), Attack_B(5)
    2 absorbing states: Goal_B, Goal_A.

    When *team* is ``"A"``, Goal_B gets reward 1 and Goal_A gets reward 0.
    When *team* is ``"B"``, Goal_A gets reward 1 and Goal_B gets reward 0.
    """
    if da == 0 and ma == 0 and aa == 0:
        return 0.0
    if db == 0 and mb == 0 and ab == 0:
        return 0.0

    # Absorbing-state markers: -1 = Goal_B, -2 = Goal_A
    reward = {"-1": 1.0, "-2": 0.0} if team == "A" else {"-1": 0.0, "-2": 1.0}

    states = [
        # (success_target, success_weight, fail_target, fail_weight)
        (1, ma, 5, ab),    # Defense_A
        (2, aa, 4, mb),    # Midfield_A
        (-1, p_goal, 3, db),  # Attack_A (→Goal_B on success)
        (4, mb, 2, aa),    # Defense_B
        (5, ab, 1, ma),    # Midfield_B
        (-2, p_goal, 0, da),  # Attack_B (→Goal_A on success)
    ]

    Q = np.zeros((6, 6))
    r = np.zeros(6)

    for i, (s_tgt, s_w, f_tgt, f_w) in enumerate(states):
        total = s_w + f_w
        if total == 0:
            continue
        p_success = s_w / total
        p_fail = f_w / total

        for tgt, prob in [(s_tgt, p_success), (f_tgt, p_fail)]:
            if tgt < 0:
                r[i] += prob * reward[str(tgt)]
            else:
                Q[i, tgt] += prob

    A = np.eye(6) - gamma * Q
    b = gamma * r
    V = np.linalg.solve(A, b)
    return float(V[0])  # Value at Defense_A (starting state)


def build_stochastic_cf_2team(
    agent_mapping: dict[str, set[str]],
    m: int,
    p_goal: int,
    gamma: float,
    *,
    team: str = "A",
) -> tuple[list[str], dict[frozenset[str], float]]:
    """Build stochastic characteristic function for all coalitions in 2-team model.

    When *team* is ``"A"``, the CF measures P(Team A scores).
    When *team* is ``"B"``, the CF measures P(Team B scores).
    """
    agents = sorted({a for s in agent_mapping.values() for a in s})
    n = len(agents)

    da_set = agent_mapping["pass_def_mid_A"]
    ma_set = agent_mapping["pass_mid_att_A"]
    aa_set = agent_mapping["shoot_A"]
    db_set = agent_mapping["pass_def_mid_B"]
    mb_set = agent_mapping["pass_mid_att_B"]
    ab_set = agent_mapping["shoot_B"]

    v: dict[frozenset[str], float] = {}
    for size in range(n + 1):
        for combo in combinations(agents, size):
            coalition = frozenset(combo)
            v[coalition] = stochastic_value_2team(
                len(da_set & coalition),
                len(ma_set & coalition),
                len(aa_set & coalition),
                len(db_set & coalition),
                len(mb_set & coalition),
                len(ab_set & coalition),
                m,
                p_goal,
                gamma,
                team=team,
            )
    return agents, v


def _build_dot_2team(net, im, fm, smap) -> graphviz.Digraph:
    """Build a compact graphviz Digraph with direct place-to-place edges.

    Places are the only visible nodes, arranged in physical zone columns.
    Edge labels show transition probabilities.  Both goals are drawn as
    doublecircle (absorbing states).  Team A forward path flows L→R.
    """
    weights = {t.name: rv.get_weight() for t, rv in smap.items()}

    # Reconstruct source_place → transition → target_place from arcs
    t_objs = {t.name: t for t in net.transitions}
    t_input: dict[str, str] = {}  # transition_name → source place
    t_output: dict[str, str] = {}  # transition_name → target place
    for arc in net.arcs:
        src, tgt = arc.source.name, arc.target.name
        if src in t_objs:
            t_output[src] = tgt
        else:
            t_input[tgt] = src

    # Compute probabilities: group transitions by source place
    source_totals: dict[str, float] = defaultdict(float)
    for t_name, src_place in t_input.items():
        source_totals[src_place] += weights.get(t_name, 0)

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", bgcolor="transparent", nodesep="0.6", ranksep="1.2")
    dot.attr("edge", penwidth="1.5", fontsize="9", fontcolor="#333333")

    initial_places = {p.name for p in im}
    goal_places = {"Goal_A", "Goal_B"}

    def _place_node(g, pname):
        shape = "doublecircle" if pname in goal_places else "circle"
        label = pname.replace("_", "\n")
        if pname in initial_places:
            label += "\n\u25cf"
        fillcolor = "#C5D9F1" if pname in TEAM_A_PLACES else "#F4CCCC"
        g.node(
            pname,
            label=label,
            shape=shape,
            width="0.7",
            fixedsize="false",
            style="filled",
            fillcolor=fillcolor,
            fontsize="10",
        )

    # Zone columns via rank=same subgraphs
    for zone_places in ZONE_COLUMNS:
        with dot.subgraph() as s:
            s.attr(rank="same")
            for pname in zone_places:
                _place_node(s, pname)

    # Invisible edges: anchor Team B below Team A in each zone
    for zone_places in ZONE_COLUMNS:
        if len(zone_places) == 2:
            dot.edge(zone_places[0], zone_places[1], style="invis")

    # Invisible spine: Goal_A → Defense_A and Attack_A → Goal_B keep goals aligned
    dot.edge("Goal_A", "Defense_A", style="invis")

    # Direct edges from source place to target place
    team_a_forward = {"pass_def_mid_A", "pass_mid_att_A", "shoot_A"}

    for t_name in t_input:
        src_place = t_input[t_name]
        tgt_place = t_output[t_name]
        w = weights.get(t_name, 0)
        total = source_totals[src_place]
        prob = w / total if total > 0 else 0
        p_label = f"{prob:.2f}"

        is_fail = t_name in FAIL_TRANSITIONS
        is_team_a = t_name in TEAM_A_TRANSITIONS
        color = "#4472C4" if is_team_a else "#C0504D"

        attrs: dict[str, str] = {
            "label": p_label,
            "color": color,
            "fontcolor": color,
        }

        if is_fail:
            attrs["style"] = "dashed"

        if t_name not in team_a_forward:
            attrs["constraint"] = "false"

        dot.edge(src_place, tgt_place, **attrs)

    return dot


def render_2team_net(net, im, fm, smap, output_path: Path) -> None:
    """Render the two-team Petri net to the requested format (png, svg, pdf)."""
    dot = _build_dot_2team(net, im, fm, smap)
    fmt = output_path.suffix.lstrip(".")
    dot.format = fmt
    if fmt == "png":
        dot.attr(dpi="300")
    dot.render(str(output_path.with_suffix("")), cleanup=True)


def render_2team_net_svg(net, im, fm, smap) -> str:
    """Render the two-team Petri net and return the SVG string."""
    dot = _build_dot_2team(net, im, fm, smap)
    return dot.pipe(format="svg").decode("utf-8")


def composite_with_background(
    net_svg: str,
    background_jpg: Path,
    output_path: Path,
    bg_opacity: float = 0.3,
) -> None:
    """Create an SVG overlay: faded raster background + vector Petri net."""
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

    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    m = re.search(
        r'<svg[^>]*\bwidth="([\d.]+)pt"[^>]*\bheight="([\d.]+)pt"',
        net_svg,
    )
    net_w_pt, net_h_pt = float(m.group(1)), float(m.group(2))

    scale = bg_w / net_w_pt
    net_scaled_h = net_h_pt * scale
    y_offset = (bg_h - net_scaled_h) / 2

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


def _index_label(name: str, fmt: str) -> str:
    col = 1 if fmt == "latex" else 0
    if name in INDEX_LABELS:
        return INDEX_LABELS[name][col]
    return name


def format_table_markdown(results: list[dict], index_names: list[str]) -> str:
    """Format a markdown table for two-team matchups."""
    role_prefixes = ["DA", "MA", "AA", "DB", "MB", "AB"]

    header_parts = ["Matchup", "Index"]
    for p in role_prefixes:
        header_parts.append(f"${p}_i$")

    lines = [
        "| " + " | ".join(header_parts) + " |",
        "|---|---|" + "|".join("---:" for _ in role_prefixes) + "|",
    ]

    for entry in results:
        matchup_label = entry["matchup_label"]
        for i, idx_name in enumerate(index_names):
            label = _index_label(idx_name, "markdown")
            vals = entry["indices"][idx_name]
            cells = []
            for p in role_prefixes:
                cells.append(f"{vals[f'{p}1']:.4f}")
            prefix = matchup_label if i == 0 else ""
            lines.append(f"| {prefix} | {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def format_table_latex(results: list[dict], index_names: list[str]) -> str:
    """Format a LaTeX table for two-team matchups."""
    role_prefixes = ["DA", "MA", "AA", "DB", "MB", "AB"]
    n_roles = len(role_prefixes)
    role_headers = " & ".join(f"${p}_i$" for p in role_prefixes)

    lines = [
        rf"\begin{{tabular}}{{l l {'c' * n_roles}}}",
        r"\toprule",
        f"Matchup & Index & {role_headers}" + r" \\",
        r"\midrule",
    ]

    for entry in results:
        matchup_label = entry["matchup_label"]
        for i, idx_name in enumerate(index_names):
            label = _index_label(idx_name, "latex")
            vals = entry["indices"][idx_name]
            cells = " & ".join(f"{vals[f'{p}1']:.4f}" for p in role_prefixes)
            prefix = matchup_label if i == 0 else ""
            lines.append(f"{prefix} & {label} & {cells}" + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-team football scenario analysis")
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
        help="Skip PNG rendering (faster iteration)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate analysis plots",
    )
    parser.add_argument(
        "--plot-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Plot output format (default: pdf)",
    )
    args = parser.parse_args()

    output_dir = SCENARIO_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    index_names = [
        "Usability",
        "Gatekeeper",
        "Shapley-Shubik",
        "Banzhaf",
    ]
    results: list[dict] = []

    for form_a_name, (n1_a, n2_a, n3_a), form_b_name, (n1_b, n2_b, n3_b) in MATCHUPS:
        matchup_label = f"{form_a_name} vs {form_b_name}"
        print(f"Computing indices for {matchup_label}...", file=sys.stderr)

        # Generate PNML for this matchup
        edges = build_edges(n1_a, n2_a, n3_a, n1_b, n2_b, n3_b, M, P_GOAL)
        xml_str = build_pnml(PLACES, edges, INITIAL_PLACE, FINAL_PLACE)
        pnml_path = SCENARIO_DIR / "football-2team.pnml"
        pnml_path.write_text(xml_str, encoding="utf-8")

        net, im, fm, smap = load_pnml_stochastic(pnml_path)

        if not args.skip_viz:
            viz_name = f"football-2team_{form_a_name}_vs_{form_b_name}"
            # High-DPI raster
            render_2team_net(net, im, fm, smap, output_dir / f"{viz_name}.png")
            print(f"  Saved {viz_name}.png", file=sys.stderr)
            # PDF
            render_2team_net(net, im, fm, smap, output_dir / f"{viz_name}.pdf")
            print(f"  Saved {viz_name}.pdf", file=sys.stderr)
            # Composite overlay with football background
            bg_path = SCENARIO_DIR.parent / "football" / "football.jpg"
            if bg_path.exists():
                net_svg = render_2team_net_svg(net, im, fm, smap)
                svg_path = output_dir / f"{viz_name}_overlay.svg"
                composite_with_background(net_svg, bg_path, svg_path, bg_opacity=0.3)
                pdf_path = output_dir / f"{viz_name}_overlay.pdf"
                subprocess.run(
                    ["rsvg-convert", "-f", "pdf", str(svg_path), "-o", str(pdf_path)],
                    check=True,
                )
                svg_path.unlink()
                print(f"  Saved {viz_name}_overlay.pdf", file=sys.stderr)

        agent_mapping = build_agent_mapping_2team(n1_a, n2_a, n3_a, n1_b, n2_b, n3_b)
        team_a_agents = {f"DA{i}" for i in range(1, n1_a + 1)} | \
                        {f"MA{i}" for i in range(1, n2_a + 1)} | \
                        {f"AA{i}" for i in range(1, n3_a + 1)}

        # Build final markings for each team's goal
        goal_b = next(p for p in net.places if p.name == "Goal_B")
        goal_a = next(p for p in net.places if p.name == "Goal_A")
        fm_a = Marking({goal_b: 1})  # Team A wants Goal_B
        fm_b = Marking({goal_a: 1})  # Team B wants Goal_A

        def _merge(vals_a: dict[str, float], vals_b: dict[str, float]) -> dict[str, float]:
            """Take Team A agents from vals_a, Team B agents from vals_b."""
            merged = {}
            for agent in vals_a:
                merged[agent] = vals_a[agent] if agent in team_a_agents else vals_b[agent]
            return merged

        # --- Path-based indices: compute per-team, merge ---
        indices: dict[str, dict[str, float]] = {
            "Usability": _merge(
                usability(net, im, fm_a, agent_mapping, start_place="Defense_A"),
                usability(net, im, fm_b, agent_mapping, start_place="Defense_B"),
            ),
            "Gatekeeper": _merge(
                gatekeeper(net, im, fm_a, agent_mapping, start_place="Defense_A"),
                gatekeeper(net, im, fm_b, agent_mapping, start_place="Defense_B"),
            ),
        }

        # --- Stochastic coalition indices: compute per-team, merge ---
        s_agents_a, s_v_a = build_stochastic_cf_2team(
            agent_mapping, M, P_GOAL, GAMMA, team="A"
        )
        s_agents_b, s_v_b = build_stochastic_cf_2team(
            agent_mapping, M, P_GOAL, GAMMA, team="B"
        )
        ss_merged = _merge(
            shapley_shubik_from_values(s_agents_a, s_v_a),
            shapley_shubik_from_values(s_agents_b, s_v_b),
        )
        bz_merged = _merge(
            banzhaf_from_values(s_agents_a, s_v_a),
            banzhaf_from_values(s_agents_b, s_v_b),
        )
        ss_total = sum(ss_merged.values())
        bz_total = sum(bz_merged.values())
        indices["Shapley-Shubik"] = {a: v / ss_total for a, v in ss_merged.items()} if ss_total > 0 else ss_merged
        indices["Banzhaf"] = {a: v / bz_total for a, v in bz_merged.items()} if bz_total > 0 else bz_merged

        results.append(
            {
                "matchup_label": matchup_label,
                "form_a": form_a_name,
                "form_b": form_b_name,
                "roles": [
                    ("DA", n1_a),
                    ("MA", n2_a),
                    ("AA", n3_a),
                    ("DB", n1_b),
                    ("MB", n2_b),
                    ("AB", n3_b),
                ],
                "indices": indices,
                "granularity": granularity(agent_mapping),
            }
        )

    # Always save both formats to outputs/
    md_table = format_table_markdown(results, index_names)
    tex_table = format_table_latex(results, index_names)

    (output_dir / "table.md").write_text(md_table, encoding="utf-8")
    (output_dir / "table.tex").write_text(tex_table, encoding="utf-8")
    print(f"Wrote table.md and table.tex to {output_dir}", file=sys.stderr)

    # Still support explicit --output / stdout for the chosen format
    chosen = tex_table if args.format == "latex" else md_table
    if args.output:
        args.output.write_text(chosen, encoding="utf-8")
        print(f"Wrote table to {args.output}", file=sys.stderr)
    else:
        print(chosen)

    if args.plot:
        labels = [e["matchup_label"] for e in results]
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
                title=f"Football 2-Team: Granularity vs {ylabel}",
                ylabel=ylabel,
            )
            print(f"Saved plot to {path}", file=sys.stderr)

        # Per-role representative agent powers
        role_prefixes = ["DA", "MA", "AA", "DB", "MB", "AB"]
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
                    labels,
                    agent_labels,
                    index_powers,
                    p,
                    title="Football 2-Team: Power per Role",
                ),
            ),
            (
                "index_correlation",
                lambda p: plot_index_correlation(
                    labels,
                    agent_labels,
                    index_powers,
                    p,
                    title="Football 2-Team: Index Correlation",
                ),
            ),
            (
                "power_heatmap",
                lambda p: plot_power_heatmap(
                    labels,
                    agent_labels,
                    index_powers,
                    p,
                    title="Football 2-Team: Power Heatmap",
                ),
            ),
            (
                "lorenz_curves",
                lambda p: plot_lorenz_curves(
                    labels,
                    index_powers,
                    p,
                    title="Football 2-Team: Lorenz Curves",
                ),
            ),
            (
                "rank_agreement",
                lambda p: plot_rank_agreement(
                    labels,
                    index_powers,
                    p,
                    title="Football 2-Team: Rank Agreement",
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
                    title="Football 2-Team: Power Deltas",
                ),
            ),
        ]:
            path = output_dir / f"{name}.{ext}"
            fn(path)
            print(f"Saved plot to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
