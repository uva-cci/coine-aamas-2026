"""Fuzzball scenario: load and visualize the Petri net."""

from pathlib import Path

from lib import load_pnml, save_net_png

SCENARIO_DIR = Path(__file__).parent


def main() -> None:
    net, im, fm = load_pnml(SCENARIO_DIR / "fuzzball.pnml")
    output_dir = SCENARIO_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_net_png(net, im, fm, output_dir / "fuzzball.png")
    print(f"Saved visualization to {output_dir / 'fuzzball.png'}")


if __name__ == "__main__":
    main()
