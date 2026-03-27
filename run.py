from __future__ import annotations

import argparse
import importlib


PIPELINES = {
    "pfnn": "pipelines.pfnn.main",
    "repulsive_shell": "pipelines.repulsive_shell.main",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured research pipeline.")
    parser.add_argument("--pipeline", choices=PIPELINES.keys(), required=True)
    args = parser.parse_args()

    module = importlib.import_module(PIPELINES[args.pipeline])
    module.main()


if __name__ == "__main__":
    main()
