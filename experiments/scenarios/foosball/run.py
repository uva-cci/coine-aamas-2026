"""Foosball scenario: load and visualize the stochastic Petri net."""

from pathlib import Path

from lib import build_stochastic_decorations, load_pnml_stochastic, save_net_png

SCENARIO_DIR = Path(__file__).parent


def main() -> None:
    net, im, fm, smap = load_pnml_stochastic(SCENARIO_DIR / "foosball.pnml")

    print("Stochastic weights:")
    for t, rv in sorted(smap.items(), key=lambda x: x[0].name):
        print(f"  {t.name}: weight={rv.get_weight()}")

    decorations = build_stochastic_decorations(smap)

    output_dir = SCENARIO_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_net_png(net, im, fm, output_dir / "foosball.eps", decorations=decorations)
    print(f"Saved visualization to {output_dir / 'foosball.eps'}")


if __name__ == "__main__":
    main()
