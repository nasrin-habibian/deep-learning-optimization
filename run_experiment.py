import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dlopt.experiments import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Compare neural-network optimizers.")
    parser.add_argument("--config", default="config.json", help="Path to JSON config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    run_experiment(config)


if __name__ == "__main__":
    main()
