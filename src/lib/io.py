"""PNML loading via pm4py."""

from pathlib import Path
from typing import Any

import pm4py
from pm4py.objects.petri_net.importer import importer
from pm4py.objects.petri_net.importer.variants.pnml import Parameters
from pm4py.objects.petri_net.obj import Marking, PetriNet


def load_pnml(path: str | Path) -> tuple[PetriNet, Marking, Marking]:
    """Load a Petri net from a PNML file.

    Returns (net, initial_marking, final_marking).
    """
    return pm4py.read_pnml(str(path))


def load_pnml_stochastic(
    path: str | Path,
) -> tuple[PetriNet, Marking, Marking, dict[Any, Any]]:
    """Load a Petri net with stochastic information from a PNML file.

    Returns (net, initial_marking, final_marking, stochastic_map).
    The stochastic_map maps Transition objects to RandomVariable objects
    containing weight, priority, and distribution information.
    """
    return importer.apply(
        str(path),
        parameters={Parameters.RETURN_STOCHASTIC_MAP: True},
    )
