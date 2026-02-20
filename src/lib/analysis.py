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


def reachability_graph(net: PetriNet, initial_marking: Marking) -> rg_util.TransitionSystem:
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


def _all_simple_paths(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    *,
    start_place: str | None = None,
) -> list[list[str]]:
    """Find all simple paths (no repeated markings) from *im* to *fm*.

    Returns a list of paths, where each path is a list of transition names.
    Uses covering semantics: a marking *m* satisfies the goal when m >= fm.

    When *start_place* is given, the search starts from a single-token marking
    in that place instead of *im*.
    """
    if start_place is not None:
        place_obj = next((p for p in net.places if p.name == start_place), None)
        if place_obj is None:
            raise KeyError(f"Place '{start_place}' not found in net")
        im = Marking({place_obj: 1})

    def _covers(m: Marking) -> bool:
        return all(m.get(p, 0) >= fm[p] for p in fm)

    paths: list[list[str]] = []

    def _dfs(marking: Marking, path: list[str], visited: set[Marking]):
        if _covers(marking):
            paths.append(list(path))
            return
        for t in pn_semantics.enabled_transitions(net, marking):
            new_marking = pn_semantics.execute(t, net, marking)
            if new_marking not in visited:
                visited.add(new_marking)
                path.append(t.name)
                _dfs(new_marking, path, visited)
                path.pop()
                visited.discard(new_marking)

    _dfs(im, [], {im})
    return paths


def _reachability_set_size(net: PetriNet, marking: Marking) -> int:
    """Count all markings reachable from *marking* (including itself) via BFS."""
    visited: set[Marking] = {marking}
    queue: list[Marking] = [marking]
    while queue:
        m = queue.pop(0)
        for t in pn_semantics.enabled_transitions(net, m):
            m2 = pn_semantics.execute(t, net, m)
            if m2 not in visited:
                visited.add(m2)
                queue.append(m2)
    return len(visited)


def usability(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    agent_mapping: AgentMapping,
    *,
    normalized: bool = True,
    start_place: str | None = None,
) -> dict[str, float]:
    """Compute the usability index for each agent.

    For each simple path from *im* to *fm*, all non-empty prefixes are
    treated as firing sequences.  For each sequence of length *L*, every
    transition contributes ``1 / (L * k)`` to each of the *k* agents that
    can fire it.  Scores are averaged across all sequences.

    usability(a) = (1/|S|) * sum_s sum_{t in s} 1 / (|s| * |A(t)|)
                   for each agent a in A(t)

    where S is the set of all non-empty prefixes of all simple paths to *fm*
    and A(t) is the set of agents that can fire transition t.

    When *normalized* is True (default), scores are rescaled to sum to 1.

    When *start_place* is given, paths are enumerated from a single-token
    marking in that place instead of *im*.
    """
    agents = sorted({a for s in agent_mapping.values() for a in s})
    paths = _all_simple_paths(net, im, fm, start_place=start_place)

    if not paths:
        return {a: 0.0 for a in agents}

    # Expand each path into all non-empty prefixes (firing sequences)
    sequences: list[list[str]] = []
    for path in paths:
        for length in range(1, len(path) + 1):
            sequences.append(path[:length])

    scores: dict[str, float] = {a: 0.0 for a in agents}
    for seq in sequences:
        seq_len = len(seq)
        for t_name in seq:
            capable = agent_mapping.get(t_name, set())
            n_capable = len(capable)
            if n_capable > 0:
                for agent in capable:
                    scores[agent] += 1.0 / (seq_len * n_capable)

    # Average across sequences
    n_seqs = len(sequences)
    scores = {a: v / n_seqs for a, v in scores.items()}

    if normalized:
        total = sum(scores.values())
        if total > 0:
            scores = {a: v / total for a, v in scores.items()}

    return scores


def gatekeeper(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    agent_mapping: AgentMapping,
    *,
    normalized: bool = True,
) -> dict[str, float]:
    """Compute the gatekeeper power index for each agent.

    For each transition at position *p* in a path of length *L*:
    - Position weight: (L − p) / L  (earlier transitions weigh more)
    - Credit is shared equally among the *k* agents that can fire the transition

    Each agent's contribution per transition: (L − p) / (L · k).
    Scores are summed across all paths.

    When *normalized* is True (default), scores are rescaled to sum to 1.
    """
    agents = sorted({a for s in agent_mapping.values() for a in s})
    paths = _all_simple_paths(net, im, fm)

    if not paths:
        return {a: 0.0 for a in agents}

    scores: dict[str, float] = {a: 0.0 for a in agents}
    for path in paths:
        path_len = len(path)
        for position, t_name in enumerate(path):
            position_weight = (path_len - position) / path_len
            capable = agent_mapping.get(t_name, set())
            n_capable = len(capable)
            if n_capable > 0:
                contribution = position_weight / n_capable
                for agent in capable:
                    scores[agent] += contribution

    if normalized:
        total = sum(scores.values())
        if total > 0:
            scores = {a: v / total for a, v in scores.items()}

    return scores


def gatekeeper_reach(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    agent_mapping: AgentMapping,
    *,
    normalized: bool = True,
) -> dict[str, float]:
    """Gatekeeper variant weighted by forward-reachability set size.

    For each transition in a simple path, the weight is the number of markings
    reachable from the marking obtained after firing that transition.  Credit is
    shared equally among the agents that can fire the transition.

    When *normalized* is True (default), scores are rescaled to sum to 1.
    """
    agents = sorted({a for s in agent_mapping.values() for a in s})
    paths = _all_simple_paths(net, im, fm)

    if not paths:
        return {a: 0.0 for a in agents}

    by_name = {t.name: t for t in net.transitions}
    reach_cache: dict[Marking, int] = {}

    scores: dict[str, float] = {a: 0.0 for a in agents}
    for path in paths:
        marking = im
        for t_name in path:
            marking = pn_semantics.execute(by_name[t_name], net, marking)
            if marking not in reach_cache:
                reach_cache[marking] = _reachability_set_size(net, marking)
            weight = reach_cache[marking]
            capable = agent_mapping.get(t_name, set())
            n_capable = len(capable)
            if n_capable > 0:
                contribution = weight / n_capable
                for agent in capable:
                    scores[agent] += contribution

    if normalized:
        total = sum(scores.values())
        if total > 0:
            scores = {a: v / total for a, v in scores.items()}

    return scores


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient of a distribution (0 = equal, 1 = maximally unequal)."""
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


def granularity(agent_mapping: AgentMapping) -> float:
    """Gini index of the supply-degree distribution across transitions.

    supply degree deg(t) = |agent_mapping[t]|  (number of agents that can fire t)
    supply share  p_s(t) = deg(t) / sum(deg)
    granularity   = gini_coefficient([p_s(t) for all t])
    """
    degrees = [float(len(agents)) for agents in agent_mapping.values()]
    total = sum(degrees)
    if total == 0:
        return 0.0
    shares = [d / total for d in degrees]
    return gini_coefficient(shares)


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
