import taichi as ti


@ti.kernel
def reset_lagrange_multipliers(  # pyright: ignore[reportInvalidTypeForm]
    lambdas: ti.template(),
):
    for i in lambdas:
        lambdas[i] = 0.0


@ti.kernel
def solve_distance_constraints(  # pyright: ignore[reportInvalidTypeForm]
    x: ti.template(),
    inv_mass: ti.template(),
    i0: ti.template(),
    i1: ti.template(),
    rest_length: ti.template(),
    compliance: ti.template(),
    lambdas: ti.template(),
    dt: ti.f32,
):
    for k in i0:
        a = i0[k]
        b = i1[k]

        w_a = inv_mass[a]
        w_b = inv_mass[b]
        w = w_a + w_b
        if w == 0.0:
            continue

        delta = x[a] - x[b]
        dist = delta.norm()
        if dist < 1e-6:
            continue

        constraint = dist - rest_length[k]
        alpha = compliance[k] / (dt * dt)
        delta_lambda = (-constraint - alpha * lambdas[k]) / (w + alpha)
        lambdas[k] += delta_lambda

        direction = delta / dist
        correction = delta_lambda * direction

        x[a] += correction * w_a
        x[b] -= correction * w_b
