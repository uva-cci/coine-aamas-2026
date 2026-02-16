"""Structural analysis of Petri nets."""

import numpy as np
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import incidence_matrix as im_util
from pm4py.objects.petri_net.utils import reachability_graph as rg_util


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
