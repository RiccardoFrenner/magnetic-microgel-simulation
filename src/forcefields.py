from __future__ import annotations

import numpy as np


def exp_force_fun(positions, *, center, equil_distance: float):
    positions = positions.reshape((-1, 3))

    center_to_p_vectors = positions - center
    distances_from_center = np.linalg.norm(center_to_p_vectors, axis=1, keepdims=True)
    # prevents division by zero but return zero force anyways
    distances_from_center[distances_from_center < 1e-12] = 1.0
    force_unit_vectors = center_to_p_vectors / distances_from_center
    # scale = 2*2.7e-3  # Too strong
    # scale = 2*2.7e-4  # Too strong
    # scale = 2*2.7e-5  # Too weak
    scale = 6 * 2.7e-5
    f = -force_unit_vectors * np.exp(distances_from_center / equil_distance) * scale
    return f


def harm_force_fun(positions, *, center, equil_distance: float):
    positions = positions.reshape((-1, 3))

    center_to_p_vectors = positions - center
    distances_from_center = np.linalg.norm(center_to_p_vectors, axis=1, keepdims=True)
    # prevents division by zero but return zero force anyways
    distances_from_center[distances_from_center < 1e-12] = 1.0
    force_unit_vectors = -center_to_p_vectors / distances_from_center

    # Actually too strong as well. This results in a whole in the middle
    # of the constraint
    # scale = 2.7e-2

    # scale = 2.7e-4  # seems OK. Could be a little higher (x2) maybe
    scale = 3.5e-4  # good
    f = (
        force_unit_vectors
        * np.clip((distances_from_center - equil_distance) / equil_distance, -1.0, 1.0)
        * scale
    )
    return f


force_fields = [exp_force_fun, harm_force_fun]


def _field_from_fn(box_size, grid_spacing, f):
    import itertools

    shape = np.array(np.ceil(box_size / grid_spacing), dtype=int) + 2
    origin = -0.5 * grid_spacing

    positions = np.zeros((shape[0], shape[1], shape[2], 3))
    field = np.zeros((shape[0], shape[1], shape[2], 3))

    for i in itertools.product(*map(range, shape)):
        x = origin + np.array(i) * grid_spacing
        positions[i] = x
        field[i] = f(x)

    return field, positions


def _main():
    from functools import partial

    import matplotlib.pyplot as plt

    box_l = np.full(3, 30.0)
    confinement_radius = 12.0
    grid_spacing = np.full(3, box_l[0] / (box_l[0] // (confinement_radius / 12) + 1))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    for ax, force_fun in zip(axs, force_fields):
        ax_title = force_fun.__name__
        force_fun = partial(
            force_fun, center=box_l / 2, equil_distance=confinement_radius
        )

        field_data, positions = _field_from_fn(box_l, grid_spacing, force_fun)
        n = field_data.shape[0]

        ax.set_title(ax_title)
        ax.set_aspect("equal")
        ax.hlines(box_l[1] / 2 + confinement_radius, 0, box_l[0])
        ax.vlines(box_l[0] / 2 + confinement_radius, 0, box_l[1])
        ax.hlines(box_l[1] / 2 - confinement_radius, 0, box_l[0])
        ax.vlines(box_l[0] / 2 - confinement_radius, 0, box_l[1])
        ax.quiver(
            positions[:, :, n // 2, 0],
            positions[:, :, n // 2, 1],
            field_data[:, :, n // 2, 0],
            field_data[:, :, n // 2, 1],
        )

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    for ax, force_fun in zip(axs, force_fields):
        ax_title = force_fun.__name__
        force_fun = partial(
            force_fun, center=box_l / 2, equil_distance=confinement_radius
        )

        field_data, positions = _field_from_fn(box_l, grid_spacing, force_fun)
        n = field_data.shape[0]

        ax.set_title(ax_title)
        ax.plot(positions[:, n // 2, n // 2, 0], field_data[:, n // 2, n // 2, 0])
        # draw a line at the center and two at the confinement radii
        ymax = np.max(field_data[:, n // 2, n // 2, 0])
        print(ymax)
        ax.vlines(box_l[0] / 2, -ymax, ymax, color="black", linestyles="--")
        ax.vlines(
            box_l[0] / 2 + confinement_radius,
            -ymax,
            ymax,
            color="black",
            linestyles="--",
        )
        ax.vlines(
            box_l[0] / 2 - confinement_radius,
            -ymax,
            ymax,
            color="black",
            linestyles="--",
        )
    plt.show()


def compute_energy_difference():
    # compute the amount of energy needed to pull a particle
    # from the center of the exponential force field to the boundary.
    # Why? -> The exponential force field has issues connecting the
    # polymers into a single cluster and we thought one reason may be
    # the the density of agents is far too low at the edges because it
    # would cost too much energy for them to even be there.
    import sympy as sp

    def exp_force(r):
        return 2 * 2.7e-3 * sp.exp(r)

    r = sp.symbols("r")
    energy_required = sp.integrate(exp_force(r), (r, 0, 1))
    print(energy_required.evalf())


if __name__ == "__main__":
    # _main()
    compute_energy_difference()
