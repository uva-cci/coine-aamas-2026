"""Visualization helpers for Petri nets."""

from pathlib import Path

import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet


def save_net_png(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    output_path: str | Path,
) -> None:
    """Save a Petri net visualization as PNG."""
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, str(output_path))
