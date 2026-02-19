"""Football scenario: load, visualize, and compute power indices."""

import argparse
import base64
import io
import re
import subprocess
import sys
from functools import partial
from pathlib import Path

import graphviz
import numpy as np
from PIL import Image

from lib import (
    banzhaf,
    gatekeeper,
    load_pnml_stochastic,
    shapley_shubik,
    usability,
)

SCENARIO_DIR = Path(__file__).parent

# --- Formations: name → (n_def, n_mid, n_att) ---
FORMATIONS = {
    "1-2-5-3": (2, 5, 3),
    "1-4-4-2": (4, 4, 2),
    "1-3-5-2": (3, 5, 2),
    "1-4-3-3": (4, 3, 3),
}

# Zone ordering: left-to-right on the pitch
ZONE_ORDER = ["Defense", "Midfield", "Attack", "Goal"]

# Display labels for each power index: (markdown, latex)
INDEX_LABELS: dict[str, tuple[str, str]] = {
    "Shapley-Shubik": ("Shapley-Shubik", r"$\phi_{a_i}$"),
    "Banzhaf": ("Banzhaf", r"$\beta_{a_i}$"),
    "Usability": ("Usability", r"$U(a_i)$"),
    "Gatekeeper": ("Gatekeeper", r"$G(a_i)$"),
}


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
    index_names = ["Shapley-Shubik", "Banzhaf", "Usability", "Gatekeeper"]
    results: list[dict] = []

    for name, (n_def, n_mid, n_att) in FORMATIONS.items():
        print(f"Computing indices for {name}...", file=sys.stderr)
        agent_mapping = build_agent_mapping(n_def, n_mid, n_att)

        indices = {
            idx: fn(net, im, fm, agent_mapping)
            for idx, fn in [
                ("Shapley-Shubik", shapley_shubik),
                ("Banzhaf", banzhaf),
                ("Usability", partial(usability, start_place="Defense")),
                ("Gatekeeper", gatekeeper),
            ]
        }

        results.append(
            {
                "formation": name,
                "roles": [("D", n_def), ("M", n_mid), ("A", n_att)],
                "indices": indices,
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


if __name__ == "__main__":
    main()
