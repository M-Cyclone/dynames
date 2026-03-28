from __future__ import annotations

import argparse
import importlib

from viz.viewer import launch_viewer


PIPELINES = {
    "pfnn": "pipelines.pfnn.main",
    "repulsive_shell": "pipelines.repulsive_shell.main",
    "xpbd_cloth": "pipelines.xpbd_cloth.main",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured research pipeline.")
    parser.add_argument("--pipeline", choices=PIPELINES.keys())
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the viewer tool. With --pipeline, opens that pipeline's viewer when available.",
    )
    args = parser.parse_args()

    if args.viewer and not args.pipeline:
        launch_viewer(title="Dynames Viewer", scene_name="Viewer Tool")
        return

    if not args.pipeline:
        parser.error("please provide --pipeline, or use --viewer for the standalone tool")

    module = importlib.import_module(PIPELINES[args.pipeline])

    if args.viewer and hasattr(module, "launch_viewer"):
        module.launch_viewer()
        return

    module.main()


if __name__ == "__main__":
    main()
