"""Generate the two-team adversarial football scenario PNML with stochastic weights.

Two teams compete on a pitch with 3 shared physical zones. Failed passes give
possession to the opposing team in the same zone. Success/fail weights depend
on both receiving teammates and opposing players.

Physical zones:
  Zone 1 (near Goal_A): Defense_A / Attack_B
  Zone 2 (center):      Midfield_A / Midfield_B
  Zone 3 (near Goal_B): Attack_A / Defense_B
"""

from pathlib import Path
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

SCENARIO_DIR = Path(__file__).parent

PLACES = [
    "Defense_A",
    "Midfield_A",
    "Attack_A",
    "Goal_B",
    "Defense_B",
    "Midfield_B",
    "Attack_B",
    "Goal_A",
]
INITIAL_PLACE = "Defense_A"
FINAL_PLACE = "Goal_B"


def build_edges(
    n1_a: int,
    n2_a: int,
    n3_a: int,
    n1_b: int,
    n2_b: int,
    n3_b: int,
    m: int,
    p_goal: int,
) -> list[tuple[str, str, str, float]]:
    """Build the 12 transition edges for a two-team match.

    Returns list of (source, target, transition_name, weight).
    """
    return [
        # Team A: progresses Defense_A → Midfield_A → Attack_A → Goal_B
        ("Defense_A", "Midfield_A", "pass_def_mid_A", n2_a),
        ("Defense_A", "Attack_B", "fail_def_A", n3_b),
        ("Midfield_A", "Attack_A", "pass_mid_att_A", n3_a),
        ("Midfield_A", "Midfield_B", "fail_mid_A", n2_b),
        ("Attack_A", "Goal_B", "shoot_A", p_goal),
        ("Attack_A", "Defense_B", "fail_att_A", n1_b),
        # Team B: progresses Defense_B → Midfield_B → Attack_B → Goal_A
        ("Defense_B", "Midfield_B", "pass_def_mid_B", n2_b),
        ("Defense_B", "Attack_A", "fail_def_B", n3_a),
        ("Midfield_B", "Attack_B", "pass_mid_att_B", n3_b),
        ("Midfield_B", "Midfield_A", "fail_mid_B", n2_a),
        ("Attack_B", "Goal_A", "shoot_B", p_goal),
        ("Attack_B", "Defense_A", "fail_att_B", n1_a),
    ]


def build_pnml(
    places: list[str],
    edges: list[tuple[str, str, str, float]],
    initial_place: str,
    final_place: str,
) -> str:
    """Build PNML XML string from places, edges, initial and final markings."""
    pnml = Element("pnml")
    net = SubElement(
        pnml,
        "net",
        id="football-2team",
        type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel",
    )
    page = SubElement(net, "page", id="page0")

    # Places
    for place_id in places:
        p = SubElement(page, "place", id=place_id)
        name = SubElement(p, "name")
        SubElement(name, "text").text = place_id
        if place_id == initial_place:
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
    fm_place = SubElement(marking, "place", idref=final_place)
    SubElement(fm_place, "text").text = "1"

    raw = tostring(pnml, encoding="unicode")
    dom = parseString(raw)
    return dom.toprettyxml(indent="  ", encoding=None)


def main() -> None:
    # Default: Team A 2-5-3 vs Team B 4-3-3
    n1_a, n2_a, n3_a = 2, 5, 3
    n1_b, n2_b, n3_b = 4, 3, 3
    m, p_goal = 10, 3

    edges = build_edges(n1_a, n2_a, n3_a, n1_b, n2_b, n3_b, m, p_goal)
    xml_str = build_pnml(PLACES, edges, INITIAL_PLACE, FINAL_PLACE)

    output = SCENARIO_DIR / "football-2team.pnml"
    output.write_text(xml_str, encoding="utf-8")
    print(f"Generated {output}")
    print(f"  {len(PLACES)} places, {len(edges)} transitions")


if __name__ == "__main__":
    main()
