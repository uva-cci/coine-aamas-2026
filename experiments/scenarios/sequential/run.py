"""Sequential scenario: power index analysis."""

from pathlib import Path

from lib import banzhaf, load_pnml, shapley_shubik

SCENARIO_DIR = Path(__file__).parent

CONFIGS = {
    "Even": {
        "t0": {"1", "2", "3"},
        "t1": {"1", "2", "3"},
        "t2": {"1", "2", "3"},
    },
    "Bottleneck": {
        "t0": {"1", "2", "3"},
        "t1": {"1", "2", "3"},
        "t2": {"1"},
    },
    "Funnel": {
        "t0": {"1", "2", "3"},
        "t1": {"1", "2"},
        "t2": {"1"},
    },
}


def main() -> None:
    net, im, fm = load_pnml(SCENARIO_DIR / "sequential.pnml")

    for name, agent_mapping in CONFIGS.items():
        print(f"--- {name} ---")
        ss = shapley_shubik(net, im, fm, agent_mapping)
        bz = banzhaf(net, im, fm, agent_mapping)
        print(f"  Shapley-Shubik: {ss}")
        print(f"  Banzhaf:        {bz}")
        print()


if __name__ == "__main__":
    main()
