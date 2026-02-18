"""Generate the football scenario PNML with stochastic weights.

Zone-based model: the field is divided into Defense, Midfield, and Attack
zones. Pass probabilities are derived from player counts (N_i) and the
number of cells per zone (M). Each zone has M cells; a pass succeeds if
the ball lands on a cell occupied by a teammate in the target zone.
"""

from pathlib import Path
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

SCENARIO_DIR = Path(__file__).parent

# --- Team composition and field parameters ---
N_1 = 2   # defenders
N_2 = 5   # midfielders
N_3 = 3   # attackers
M = 10    # cells per zone
P_GOAL = 3  # goal width in cells

assert M > N_2, f"M ({M}) must be greater than N_2 ({N_2})"
assert M > N_3, f"M ({M}) must be greater than N_3 ({N_3})"
assert M > P_GOAL, f"M ({M}) must be greater than P_GOAL ({P_GOAL})"

# --- Places ---
PLACES = ["Defense", "Midfield", "Attack", "Goal"]
INITIAL_PLACE = "Defense"
FINAL_PLACE = "Goal"

# --- Transitions: (source, target, name, weight) ---
EDGES: list[tuple[str, str, str, float]] = [
    ("Defense", "Midfield", "pass_def_mid", N_2),
    ("Defense", "Defense", "fail_def", M - N_2),
    ("Midfield", "Attack", "pass_mid_att", N_3),
    ("Midfield", "Defense", "fail_mid", M - N_3),
    ("Attack", "Goal", "shoot", P_GOAL),
    ("Attack", "Defense", "fail_att", M - P_GOAL),
]


def build_pnml(edges: list[tuple[str, str, str, float]]) -> str:
    pnml = Element("pnml")
    net = SubElement(
        pnml, "net", id="football",
        type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel",
    )
    page = SubElement(net, "page", id="page0")

    # Places
    for place_id in PLACES:
        p = SubElement(page, "place", id=place_id)
        name = SubElement(p, "name")
        SubElement(name, "text").text = place_id
        if place_id == INITIAL_PLACE:
            im = SubElement(p, "initialMarking")
            SubElement(im, "text").text = "1"

    # Transitions with stochastic weights
    for source, target, tid, weight in edges:
        t = SubElement(page, "transition", id=tid)
        name = SubElement(t, "name")
        SubElement(name, "text").text = tid
        ts = SubElement(t, "toolspecific", tool="StochasticPetriNet", version="0.2")
        w = SubElement(ts, "property", key="weight")
        w.text = str(weight)
        d = SubElement(ts, "property", key="distributionType")
        d.text = "IMMEDIATE"
        pr = SubElement(ts, "property", key="priority")
        pr.text = "1"

    # Arcs
    arc_idx = 0
    for source, target, tid, _weight in edges:
        SubElement(page, "arc", id=f"a{arc_idx}", source=source, target=tid)
        arc_idx += 1
        SubElement(page, "arc", id=f"a{arc_idx}", source=tid, target=target)
        arc_idx += 1

    # Final marking
    fms = SubElement(net, "finalmarkings")
    marking = SubElement(fms, "marking")
    fm_place = SubElement(marking, "place", idref=FINAL_PLACE)
    SubElement(fm_place, "text").text = "1"

    raw = tostring(pnml, encoding="unicode")
    dom = parseString(raw)
    return dom.toprettyxml(indent="  ", encoding=None)


def main() -> None:
    xml_str = build_pnml(EDGES)
    output = SCENARIO_DIR / "football.pnml"
    output.write_text(xml_str, encoding="utf-8")
    print(f"Generated {output}")
    print(f"  {len(EDGES)} transitions")


if __name__ == "__main__":
    main()
