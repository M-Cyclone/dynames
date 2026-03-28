import taichi as ti


@ti.kernel
def predict_positions(
    x: ti.template(),
    x_prev: ti.template(),
    v: ti.template(),
    inv_mass: ti.template(),
    gravity: ti.types.vector(3, ti.f32),
    dt: ti.f32,
):
    for i in x:
        x_prev[i] = x[i]
        if inv_mass[i] > 0.0:
            v[i] += gravity * dt
            x[i] += v[i] * dt


@ti.kernel
def update_velocities(
    x: ti.template(),
    x_prev: ti.template(),
    v: ti.template(),
    damping: ti.f32,
    dt: ti.f32,
):
    for i in x:
        v[i] = ((x[i] - x_prev[i]) / dt) * damping
