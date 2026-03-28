from __future__ import annotations

import numpy as np
import taichi as ti


class Particles:
    def __init__(self, n: int) -> None:
        self.n = n
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.x_prev = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.inv_mass = ti.field(dtype=ti.f32, shape=n)

    def set_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        inv_mass: np.ndarray | None = None,
    ) -> None:
        positions_array = np.asarray(positions, dtype=np.float32)
        if positions_array.shape != (self.n, 3):
            raise ValueError(f"positions must have shape {(self.n, 3)}, got {positions_array.shape}")

        velocity_array = np.zeros((self.n, 3), dtype=np.float32) if velocities is None else np.asarray(velocities, dtype=np.float32)
        if velocity_array.shape != (self.n, 3):
            raise ValueError(f"velocities must have shape {(self.n, 3)}, got {velocity_array.shape}")

        inv_mass_array = np.ones(self.n, dtype=np.float32) if inv_mass is None else np.asarray(inv_mass, dtype=np.float32)
        if inv_mass_array.shape != (self.n,):
            raise ValueError(f"inv_mass must have shape {(self.n,)}, got {inv_mass_array.shape}")

        self.x.from_numpy(positions_array)
        self.x_prev.from_numpy(positions_array)
        self.v.from_numpy(velocity_array)
        self.inv_mass.from_numpy(inv_mass_array)

    def positions_numpy(self) -> np.ndarray:
        return self.x.to_numpy()

    def velocities_numpy(self) -> np.ndarray:
        return self.v.to_numpy()
