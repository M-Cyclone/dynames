from __future__ import annotations

from typing import TypeAlias

import numpy as np

from latticeon.backend.taichi.fields import ensure_taichi_initialized
from latticeon.core.scene import Scene
from latticeon.data.particles import Particles
from latticeon.systems.xpbd.constraints import DistanceConstraints
from latticeon.systems.xpbd.xpbd_system import XPBDSystem

ClothData: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
ClothGridBuild: TypeAlias = tuple[ClothData, np.ndarray, np.ndarray]


def build_cloth_grid(width: int, height: int, spacing: float) -> ClothGridBuild:
    positions: list[list[float]] = []
    inv_mass = np.ones(width * height, dtype=np.float32)

    for row in range(height):
        for col in range(width):
            positions.append(
                [
                    (col - (width - 1) * 0.5) * spacing,
                    4.5,
                    (row - (height - 1) * 0.5) * spacing,
                ]
            )

    inv_mass[0] = 0.0
    inv_mass[width - 1] = 0.0

    i0: list[int] = []
    i1: list[int] = []
    rest_length: list[float] = []
    compliance: list[float] = []

    def particle_index(row: int, col: int) -> int:
        return row * width + col

    for row in range(height):
        for col in range(width):
            current = particle_index(row, col)
            if col + 1 < width:
                right = particle_index(row, col + 1)
                i0.append(current)
                i1.append(right)
                rest_length.append(spacing)
                compliance.append(1e-6)
            if row + 1 < height:
                below = particle_index(row + 1, col)
                i0.append(current)
                i1.append(below)
                rest_length.append(spacing)
                compliance.append(1e-6)

    return (
        (
            np.asarray(positions, dtype=np.float32),
            inv_mass,
            np.asarray(i0, dtype=np.int32),
            np.asarray(i1, dtype=np.int32),
        ),
        np.asarray(rest_length, dtype=np.float32),
        np.asarray(compliance, dtype=np.float32),
    )


def build_pipeline() -> tuple[Scene, Particles]:
    ensure_taichi_initialized()

    cloth_width = 12
    cloth_height = 12
    spacing = 0.18

    cloth_data, rest_length, compliance = build_cloth_grid(
        cloth_width, cloth_height, spacing
    )
    positions, inv_mass, i0, i1 = cloth_data

    particles = Particles(n=cloth_width * cloth_height)
    particles.set_state(positions=positions, inv_mass=inv_mass)

    constraints = DistanceConstraints(n=len(i0))
    constraints.set_data(i0=i0, i1=i1, rest_length=rest_length, compliance=compliance)

    xpbd = XPBDSystem(particles, constraints, iterations=12)
    scene = Scene([xpbd])
    return scene, particles
