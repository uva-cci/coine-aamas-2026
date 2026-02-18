"""Structural analysis of Petri nets."""

from itertools import combinations
from math import factorial

import numpy as np
from pm4py.objects.petri_net import semantics as pn_semantics
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import incidence_matrix as im_util
from pm4py.objects.petri_net.utils import reachability_graph as rg_util

# A: T → 2^N — maps each transition name to the set of agents that can fire it
AgentMapping = dict[str, set[str]]


def incidence_matrix(net: PetriNet) -> np.ndarray:
    """Compute the incidence matrix of a Petri net.

    Rows correspond to places, columns to transitions.
    """
    im = im_util.construct(net)
    return np.array(im.a_matrix)


def reachability_graph(
    net: PetriNet, initial_marking: Marking
) -> rg_util.TransitionSystem:
    """Build the reachability graph from an initial marking."""
    return rg_util.construct_reachability_graph(net, initial_marking)


def is_reachable_restricted(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    allowed_transitions: set[PetriNet.Transition],
) -> bool:
    """Check if the final marking is reachable using only allowed transitions.

    BFS over the state space, firing only transitions in *allowed_transitions*.
    Uses covering semantics: a marking *m* satisfies the goal when m >= fm.
    """
    def _covers(m: Marking) -> bool:
        return all(m.get(p, 0) >= fm[p] for p in fm)

    visited: set[Marking] = set()
    queue: list[Marking] = [im]
    visited.add(im)

    while queue:
        marking = queue.pop(0)
        if _covers(marking):
            return True
        enabled = pn_semantics.enabled_transitions(net, marking)
        for t in enabled:
            if t not in allowed_transitions:
                continue
            new_marking = pn_semantics.execute(t, net, marking)
            if new_marking not in visited:
                visited.add(new_marking)
                queue.append(new_marking)
    return False


def _resolve_transitions(
    net: PetriNet, agent_mapping: AgentMapping
) -> dict[PetriNet.Transition, set[str]]:
    """Map Transition objects to agent sets based on transition names."""
    by_name = {t.name: t for t in net.transitions}
    resolved: dict[PetriNet.Transition, set[str]] = {}
    for tname, agents in agent_mapping.items():
        t = by_name.get(tname)
        if t is None:
            raise KeyError(f"Transition '{tname}' not found in net")
        resolved[t] = agents
    return resolved


def _precompute_characteristic_function(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    agent_mapping: AgentMapping,
) -> tuple[list[str], dict[frozenset[str], bool]]:
    """Enumerate all coalitions and compute the characteristic function v.

    Returns (sorted_agents, v) where v maps each coalition (frozenset) to
    whether the coalition can reach the final marking.
    """
    resolved = _resolve_transitions(net, agent_mapping)

    # Collect the full set of agents
    agents = sorted({a for s in agent_mapping.values() for a in s})
    n = len(agents)

    v: dict[frozenset[str], bool] = {}
    for size in range(n + 1):
        for combo in combinations(agents, size):
            coalition = frozenset(combo)
            # A coalition can fire a transition if it contains at least one
            # agent assigned to that transition.
            allowed = {t for t, ag in resolved.items() if ag & coalition}
            v[coalition] = is_reachable_restricted(net, im, fm, allowed)

    return agents, v


def shapley_shubik(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    agent_mapping: AgentMapping,
) -> dict[str, float]:
    """Compute the Shapley-Shubik power index for each agent.

    phi_i = sum_{S subset N\\{i}} [|S|!(n-|S|-1)!/n!] * (v(S+{i}) - v(S))
    """
    agents, v = _precompute_characteristic_function(net, im, fm, agent_mapping)
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
                s_with_i = s | {i}
                if v[s_with_i] and not v[s]:
                    total += weight
        phi[i] = total

    return phi


def banzhaf(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    agent_mapping: AgentMapping,
    *,
    normalized: bool = True,
) -> dict[str, float]:
    """Compute the Banzhaf power index for each agent.

    Raw swing count: eta_i = sum_{S subset N\\{i}} (v(S+{i}) - v(S))
    Normalized: beta_i = eta_i / sum_j eta_j
    """
    agents, v = _precompute_characteristic_function(net, im, fm, agent_mapping)
    n = len(agents)
    eta: dict[str, int] = {}

    for i in agents:
        swings = 0
        others = [a for a in agents if a != i]
        for size in range(n):
            for combo in combinations(others, size):
                s = frozenset(combo)
                s_with_i = s | {i}
                if v[s_with_i] and not v[s]:
                    swings += 1
        eta[i] = swings

    if not normalized:
        return {a: float(c) for a, c in eta.items()}

    total = sum(eta.values())
    if total == 0:
        return {a: 0.0 for a in agents}
    return {a: c / total for a, c in eta.items()}
