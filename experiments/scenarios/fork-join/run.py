"""Fork-join scenario: power index analysis."""

from pathlib import Path

from lib import banzhaf, load_pnml, shapley_shubik

SCENARIO_DIR = Path(__file__).parent

CONFIGS = {
    "Even": {
        "fork": {"1", "2", "3"},
        "branch_a": {"1", "2", "3"},
        "branch_b": {"1", "2", "3"},
        "join": {"1", "2", "3"},
    },
    "Skewed": {
        "fork": {"1", "2", "3"},
        "branch_a": {"1", "2"},
        "branch_b": {"3"},
        "join": {"1", "2", "3"},
    },
}


def main() -> None:
    net, im, fm = load_pnml(SCENARIO_DIR / "fork_join.pnml")

    for name, agent_mapping in CONFIGS.items():
        print(f"--- {name} ---")
        ss = shapley_shubik(net, im, fm, agent_mapping)
        bz = banzhaf(net, im, fm, agent_mapping)
        print(f"  Shapley-Shubik: {ss}")
        print(f"  Banzhaf:        {bz}")
        print()


if __name__ == "__main__":
    main()
