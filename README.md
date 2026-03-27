# Dynastra

Research-oriented motion/control framework with a reusable Python orchestration layer and a high-performance Rust engine.

## Layout

- `engine/`: Rust core and Python bindings
- `core/`: reusable controllers, behaviors, models, and shared utilities
- `sim/`: generic simulator loop and state definitions
- `pipelines/`: paper-specific entry points
- `data/`: loaders, preprocessing, and datasets
- `viz/`: debugging and visualization tools
- `configs/`: shared configuration templates

## Quick Start

1. Create or activate the local virtual environment:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
2. Run a pipeline:
   - `python run.py --pipeline pfnn`
   - `python run.py --pipeline repulsive_shell`

## Notes

- Pipelines should stay thin and compose reusable pieces from `core/` and `sim/`.
- The Rust engine should expose generic numeric/simulation primitives, not paper-specific logic.
