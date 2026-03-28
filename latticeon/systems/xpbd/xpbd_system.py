from __future__ import annotations

import taichi as ti

from latticeon.backend.taichi.kernels.integrate import predict_positions, update_velocities
from latticeon.backend.taichi.kernels.xpbd import reset_lagrange_multipliers, solve_distance_constraints
from latticeon.core.system import System
from latticeon.data.particles import Particles
from latticeon.systems.xpbd.constraints import DistanceConstraints


class XPBDSystem(System):
    def __init__(
        self,
        particles: Particles,
        constraints: DistanceConstraints,
        iterations: int = 10,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        damping: float = 0.995,
    ) -> None:
        self.particles = particles
        self.constraints = constraints
        self.iterations = iterations
        self.gravity = ti.Vector(gravity)
        self.damping = damping

    def step(self, dt: float) -> None:
        predict_positions(
            self.particles.x,
            self.particles.x_prev,
            self.particles.v,
            self.particles.inv_mass,
            self.gravity,
            dt,
        )

        reset_lagrange_multipliers(self.constraints.lambdas)
        for _ in range(self.iterations):
            solve_distance_constraints(
                self.particles.x,
                self.particles.inv_mass,
                self.constraints.i0,
                self.constraints.i1,
                self.constraints.rest_length,
                self.constraints.compliance,
                self.constraints.lambdas,
                dt,
            )

        update_velocities(
            self.particles.x,
            self.particles.x_prev,
            self.particles.v,
            self.damping,
            dt,
        )
