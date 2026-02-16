"""Example scenario: load a simple net and print its properties."""

from pathlib import Path

from lib import incidence_matrix, load_pnml

SCENARIO_DIR = Path(__file__).parent


def main() -> None:
    net, im, fm = load_pnml(SCENARIO_DIR / "simple_net.pnml")

    print(f"Places:      {len(net.places)}")
    print(f"Transitions: {len(net.transitions)}")
    print(f"Arcs:        {len(net.arcs)}")
    print()
    print("Incidence matrix:")
    print(incidence_matrix(net))


if __name__ == "__main__":
    main()
