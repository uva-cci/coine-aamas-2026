"""PNML loading via pm4py."""

from pathlib import Path

import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet


def load_pnml(path: str | Path) -> tuple[PetriNet, Marking, Marking]:
    """Load a Petri net from a PNML file.

    Returns (net, initial_marking, final_marking).
    """
    return pm4py.read_pnml(str(path))
