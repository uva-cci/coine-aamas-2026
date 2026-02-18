"""Generate the foosball scenario PNML with stochastic weights.

The foosball Petri net models a soccer formation (1-2-5-3) where:
- GK can pass to D1/D2, M1-M5, A1-A3, or attempt a shot on Goal
- D1/D2 can pass to M1-M5, A1-A3, or attempt a shot on Goal
- M1-M5 can pass to A1-A3 or attempt a shot on Goal
- A1-A3 can shoot on Goal

Stochastic weights reflect realistic pass/shot probabilities.
"""

from pathlib import Path
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

SCENARIO_DIR = Path(__file__).parent

# Formation layers
PLACES = ["GK", "D1", "D2", "M1", "M2", "M3", "M4", "M5", "A1", "A2", "A3", "Goal"]

DEFENDERS = ["D1", "D2"]
MIDFIELDERS = ["M1", "M2", "M3", "M4", "M5"]
ATTACKERS = ["A1", "A2", "A3"]

# Edges: (source, target, action_prefix, weight)
EDGES: list[tuple[str, str, str, float]] = []

# GK -> Defenders (safe short pass)
for d in DEFENDERS:
    EDGES.append(("GK", d, "pass", 3.0))
# GK -> Midfielders (medium range)
for m in MIDFIELDERS:
    EDGES.append(("GK", m, "pass", 1.0))
# GK -> Attackers (long ball, risky)
for a in ATTACKERS:
    EDGES.append(("GK", a, "pass", 0.3))
# GK -> Goal (near-impossible)
EDGES.append(("GK", "Goal", "shot", 0.01))

# Defenders -> Midfielders (standard build-up)
for d in DEFENDERS:
    for m in MIDFIELDERS:
        EDGES.append((d, m, "pass", 2.0))
    # Defenders -> Attackers (long ball forward)
    for a in ATTACKERS:
        EDGES.append((d, a, "pass", 0.5))
    # Defenders -> Goal (very unlikely long-range shot)
    EDGES.append((d, "Goal", "shot", 0.05))

# Midfielders -> Attackers (primary role: feed attackers)
for m in MIDFIELDERS:
    for a in ATTACKERS:
        EDGES.append((m, a, "pass", 3.0))
    # Midfielders -> Goal (occasional long-range attempt)
    EDGES.append((m, "Goal", "shot", 0.3))

# Attackers -> Goal (attackers are the goal-scorers)
for a in ATTACKERS:
    EDGES.append((a, "Goal", "shot", 5.0))


def _transition_id(source: str, target: str, prefix: str) -> str:
    return f"{prefix}_{source}_{target}"


def build_pnml(edges: list[tuple[str, str, str, float]]) -> str:
    pnml = Element("pnml")
    net = SubElement(
        pnml, "net", id="foosball",
        type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel",
    )
    page = SubElement(net, "page", id="page0")

    # Places
    for place_id in PLACES:
        p = SubElement(page, "place", id=place_id)
        name = SubElement(p, "name")
        SubElement(name, "text").text = place_id
        if place_id == "GK":
            im = SubElement(p, "initialMarking")
            SubElement(im, "text").text = "1"

    # Transitions with stochastic weights
    for source, target, prefix, weight in edges:
        tid = _transition_id(source, target, prefix)
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
    for source, target, prefix, _weight in edges:
        tid = _transition_id(source, target, prefix)
        SubElement(page, "arc", id=f"a{arc_idx}", source=source, target=tid)
        arc_idx += 1
        SubElement(page, "arc", id=f"a{arc_idx}", source=tid, target=target)
        arc_idx += 1

    # Final marking
    fms = SubElement(net, "finalmarkings")
    marking = SubElement(fms, "marking")
    fm_place = SubElement(marking, "place", idref="Goal")
    SubElement(fm_place, "text").text = "1"

    raw = tostring(pnml, encoding="unicode")
    dom = parseString(raw)
    return dom.toprettyxml(indent="  ", encoding=None)


def main() -> None:
    xml_str = build_pnml(EDGES)
    output = SCENARIO_DIR / "foosball.pnml"
    output.write_text(xml_str, encoding="utf-8")
    print(f"Generated {output}")
    print(f"  {len(EDGES)} transitions (passes + shots)")


if __name__ == "__main__":
    main()
