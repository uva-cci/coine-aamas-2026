"""Visualization helpers for Petri nets."""

from pathlib import Path
from typing import Any

import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet


def build_stochastic_decorations(
    stochastic_map: dict[Any, Any],
) -> dict[Any, dict[str, str]]:
    """Build a decorations dict labelling each transition with its weight.

    Accepts a stochastic_map as returned by ``load_pnml_stochastic``.
    Returns a dict suitable for pm4py's ``decorations`` parameter.
    """
    decorations: dict[Any, dict[str, str]] = {}
    for transition, rv in stochastic_map.items():
        weight = rv.get_weight()
        decorations[transition] = {
            "label": f"{transition.label}\nw={weight}",
            "color": "#AAAAFF",
        }
    return decorations


def save_net_png(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    output_path: str | Path,
    decorations: dict[Any, Any] | None = None,
) -> None:
    """Save a Petri net visualization as PNG."""
    pm4py.save_vis_petri_net(
        net, initial_marking, final_marking, str(output_path), decorations=decorations
    )
