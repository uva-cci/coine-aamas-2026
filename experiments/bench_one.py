"""Benchmark entry point: compute one power index for one symmetric 2-team formation.

Usage (invoked by hyperfine):
    uv run experiments/bench_one.py --index usability --formation 4-3-3 --pnml /path/to/file.pnml

Generate PNMLs first (setup):
    uv run experiments/bench_one.py --generate --formation 4-3-3 --pnml /path/to/file.pnml
"""

import argparse
import sys
from itertools import combinations
from math import factorial
from pathlib import Path
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from pm4py.objects.petri_net.obj import Marking

from lib.analysis import gatekeeper, usability
from lib.io import load_pnml_stochastic

# ---------------------------------------------------------------------------
# Inlined from football-2team/generate_pnml.py (avoids graphviz/PIL imports)
# ---------------------------------------------------------------------------

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
    return [
        ("Defense_A", "Midfield_A", "pass_def_mid_A", n2_a),
        ("Defense_A", "Attack_B", "fail_def_A", n3_b),
        ("Midfield_A", "Attack_A", "pass_mid_att_A", n3_a),
        ("Midfield_A", "Midfield_B", "fail_mid_A", n2_b),
        ("Attack_A", "Goal_B", "shoot_A", p_goal),
        ("Attack_A", "Defense_B", "fail_att_A", n1_b),
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
    pnml = Element("pnml")
    net = SubElement(
        pnml,
        "net",
        id="football-2team",
        type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel",
    )
    page = SubElement(net, "page", id="page0")
    for place_id in places:
        p = SubElement(page, "place", id=place_id)
        name = SubElement(p, "name")
        SubElement(name, "text").text = place_id
        if place_id == initial_place:
            im = SubElement(p, "initialMarking")
            SubElement(im, "text").text = "1"
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
    arc_idx = 0
    for source, target, tid, _weight in edges:
        SubElement(page, "arc", id=f"a{arc_idx}", source=source, target=tid)
        arc_idx += 1
        SubElement(page, "arc", id=f"a{arc_idx}", source=tid, target=target)
        arc_idx += 1
    fms = SubElement(net, "finalmarkings")
    marking = SubElement(fms, "marking")
    fm_place = SubElement(marking, "place", idref=final_place)
    SubElement(fm_place, "text").text = "1"
    raw = tostring(pnml, encoding="unicode")
    dom = parseString(raw)
    return dom.toprettyxml(indent="  ", encoding=None)


# ---------------------------------------------------------------------------
# Inlined from football-2team/run.py
# ---------------------------------------------------------------------------

M = 10
P_GOAL = 3
GAMMA = 0.99


def build_agent_mapping_2team(
    n1_a: int, n2_a: int, n3_a: int, n1_b: int, n2_b: int, n3_b: int
) -> dict[str, set[str]]:
    da = {f"DA{i}" for i in range(1, n1_a + 1)}
    ma = {f"MA{i}" for i in range(1, n2_a + 1)}
    aa = {f"AA{i}" for i in range(1, n3_a + 1)}
    db = {f"DB{i}" for i in range(1, n1_b + 1)}
    mb = {f"MB{i}" for i in range(1, n2_b + 1)}
    ab = {f"AB{i}" for i in range(1, n3_b + 1)}
    return {
        "pass_def_mid_A": da,
        "fail_def_A": da,
        "pass_mid_att_A": ma,
        "fail_mid_A": ma,
        "shoot_A": aa,
        "fail_att_A": aa,
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
    if da == 0 and ma == 0 and aa == 0 and team == "A":
        return 0.0
    if db == 0 and mb == 0 and ab == 0 and team == "B":
        return 0.0
    reward = {"-1": 1.0, "-2": 0.0} if team == "A" else {"-1": 0.0, "-2": 1.0}
    states = [
        (1, ma, 5, ab),
        (2, aa, 4, mb),
        (-1, p_goal, 3, db),
        (4, mb, 2, aa),
        (5, ab, 1, ma),
        (-2, p_goal, 0, da),
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
    return float(V[0]) if team == "A" else float(V[3])


def build_stochastic_cf_2team(
    agent_mapping: dict[str, set[str]],
    m: int,
    p_goal: int,
    gamma: float,
    *,
    team: str = "A",
) -> tuple[list[str], dict[frozenset[str], float]]:
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


# ---------------------------------------------------------------------------
# Stochastic Shapley-Shubik / Banzhaf (inlined from lib.analysis)
# ---------------------------------------------------------------------------


def shapley_shubik_from_values(
    agents: list[str], v: dict[frozenset[str], float]
) -> dict[str, float]:
    n = len(agents)
    n_fact = factorial(n)
    phi: dict[str, float] = {}
    for i in agents:
        total = 0.0
        others = [a for a in agents if a != i]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / n_fact
            for combo in combinations(others, size):
                s = frozenset(combo)
                total += weight * (v[s | {i}] - v[s])
        phi[i] = total
    total_phi = sum(phi.values())
    if total_phi > 0:
        phi = {a: val / total_phi for a, val in phi.items()}
    return phi


def banzhaf_from_values(agents: list[str], v: dict[frozenset[str], float]) -> dict[str, float]:
    n = len(agents)
    raw: dict[str, float] = {}
    for i in agents:
        total = 0.0
        others = [a for a in agents if a != i]
        for size in range(n):
            for combo in combinations(others, size):
                s = frozenset(combo)
                total += v[s | {i}] - v[s]
        raw[i] = total
    total = sum(raw.values())
    if total == 0:
        return {a: 0.0 for a in agents}
    return {a: val / total for a, val in raw.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

INDEX_FNS = {
    "usability": "path",
    "gatekeeper": "path",
    "shapley-shubik": "coalition",
    "banzhaf": "coalition",
}


def parse_formation(s: str) -> tuple[int, int, int]:
    parts = s.split("-")
    if len(parts) != 3:
        raise ValueError(f"Formation must be N1-N2-N3, got '{s}'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one power index")
    parser.add_argument(
        "--index",
        choices=list(INDEX_FNS.keys()),
        help="Power index to compute",
    )
    parser.add_argument(
        "--formation",
        required=True,
        help="Symmetric formation N1-N2-N3 (e.g. 4-3-3)",
    )
    parser.add_argument(
        "--pnml",
        type=Path,
        required=True,
        help="Path to pre-generated PNML file",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate PNML file and exit (setup step)",
    )
    args = parser.parse_args()

    n1, n2, n3 = parse_formation(args.formation)
    n_agents = 2 * (n1 + n2 + n3)

    if args.generate:
        edges = build_edges(n1, n2, n3, n1, n2, n3, M, P_GOAL)
        xml_str = build_pnml(PLACES, edges, INITIAL_PLACE, FINAL_PLACE)
        args.pnml.parent.mkdir(parents=True, exist_ok=True)
        args.pnml.write_text(xml_str, encoding="utf-8")
        print(f"Generated {args.pnml} ({n_agents} agents)", file=sys.stderr)
        return

    if args.index is None:
        parser.error("--index is required when not using --generate")

    net, im, fm, smap = load_pnml_stochastic(args.pnml)
    agent_mapping = build_agent_mapping_2team(n1, n2, n3, n1, n2, n3)
    team_a_agents = (
        {f"DA{i}" for i in range(1, n1 + 1)}
        | {f"MA{i}" for i in range(1, n2 + 1)}
        | {f"AA{i}" for i in range(1, n3 + 1)}
    )

    goal_b = next(p for p in net.places if p.name == "Goal_B")
    goal_a = next(p for p in net.places if p.name == "Goal_A")
    fm_a = Marking({goal_b: 1})
    fm_b = Marking({goal_a: 1})

    def _merge(vals_a: dict[str, float], vals_b: dict[str, float]) -> dict[str, float]:
        merged = {}
        for agent in vals_a:
            merged[agent] = vals_a[agent] if agent in team_a_agents else vals_b[agent]
        return merged

    index_name = args.index

    if INDEX_FNS[index_name] == "path":
        fn = usability if index_name == "usability" else gatekeeper
        _merge(
            fn(net, im, fm_a, agent_mapping, start_place="Defense_A"),
            fn(net, im, fm_b, agent_mapping, start_place="Defense_B"),
        )
    else:
        s_agents_a, s_v_a = build_stochastic_cf_2team(agent_mapping, M, P_GOAL, GAMMA, team="A")
        s_agents_b, s_v_b = build_stochastic_cf_2team(agent_mapping, M, P_GOAL, GAMMA, team="B")
        if index_name == "shapley-shubik":
            _merge(
                shapley_shubik_from_values(s_agents_a, s_v_a),
                shapley_shubik_from_values(s_agents_b, s_v_b),
            )
        else:
            _merge(
                banzhaf_from_values(s_agents_a, s_v_a),
                banzhaf_from_values(s_agents_b, s_v_b),
            )

    print(f"{index_name} | {args.formation} | |A|={n_agents} | done", file=sys.stderr)


if __name__ == "__main__":
    main()
