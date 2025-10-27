from __future__ import annotations

import numpy as np


def fene_potential(r, k, d_r_max, r_0):
    return -0.5 * k * d_r_max**2 * np.log(1 - (r - r_0) ** 2 / d_r_max**2)


def harmonic_potential(r, k, r_0):
    return 0.5 * k * (r - r_0) ** 2


def wca_potential(r, epsilon, sigma):
    return np.where(
        r < sigma * 2 ** (1.0 / 6.0),
        4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6 + 0.25),
        np.zeros_like(r),
    )
