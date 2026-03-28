from __future__ import annotations

import numpy as np
import taichi as ti


class DistanceConstraints:
    def __init__(self, n: int) -> None:
        self.n = n
        self.i0 = ti.field(dtype=ti.i32, shape=n)
        self.i1 = ti.field(dtype=ti.i32, shape=n)
        self.rest_length = ti.field(dtype=ti.f32, shape=n)
        self.compliance = ti.field(dtype=ti.f32, shape=n)
        self.lambdas = ti.field(dtype=ti.f32, shape=n)

    def set_data(
        self,
        i0: np.ndarray,
        i1: np.ndarray,
        rest_length: np.ndarray,
        compliance: np.ndarray,
    ) -> None:
        i0_array = np.asarray(i0, dtype=np.int32)
        i1_array = np.asarray(i1, dtype=np.int32)
        rest_length_array = np.asarray(rest_length, dtype=np.float32)
        compliance_array = np.asarray(compliance, dtype=np.float32)

        expected_shape = (self.n,)
        for name, array in {
            "i0": i0_array,
            "i1": i1_array,
            "rest_length": rest_length_array,
            "compliance": compliance_array,
        }.items():
            if array.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")

        self.i0.from_numpy(i0_array)
        self.i1.from_numpy(i1_array)
        self.rest_length.from_numpy(rest_length_array)
        self.compliance.from_numpy(compliance_array)
        self.lambdas.fill(0.0)
