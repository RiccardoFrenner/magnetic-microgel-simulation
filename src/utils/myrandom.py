from __future__ import annotations

import random
import string
from typing import Sequence, Union

import numpy as np


def random_uniform_ball(*,
                        rng: np.random.Generator,
                        n: int = 100,
                        R: float = 1,
                        center=None,
                        on_surface: bool = False,
                        dim: int = 3,
                        keepdim=False,
                        ):

    if n == 1 and not keepdim:
        shape = (dim, )
    else:
        shape = (dim, n)

    random_directions = rng.normal(size=shape)
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = 1 if on_surface else rng.random(n) ** (1/dim)

    center = _get_default_center(center=center, dim=dim)
    return R * (random_directions * random_radii).T + center


def random_uniform_annulus(*,
                           rng: np.random.Generator,
                           n: int = 100,
                           r: float = 0, R: float = 1,
                           center=None,
                           dim: int = 3,
                           ):

    random_directions = rng.normal(size=(dim, n))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = rng.uniform(
        low=r**dim, high=R**dim, size=(n,)) ** (1/dim)

    center = _get_default_center(center=center, dim=dim)
    return (random_directions * random_radii).T + np.array(center)


def uniform_box(*,
                rng: np.random.Generator,
                n: int,
                box_length: Union[float, Sequence[float]],
                center=None,
                dim: int = 3,
                ):

    points = (rng.random(size=(n, dim)) - 0.5) * np.asarray(box_length)
    return points + _get_default_center(center=center, dim=dim)


def uniform_box_with_exclusion(*,
                               rng: np.random.Generator,
                               box_length: float,
                               n: int = 100,
                               r: float = 0,
                               center=None,
                               dim: int = 3,
                               ):

    center = _get_default_center(center=center, dim=dim)

    if r == 0.0:
        return uniform_box(rng=rng, n=n, box_length=box_length,
                           center=center, dim=dim)

    points = np.empty(shape=(n, dim))
    boxl_half = box_length/2
    r_sqr = r**2
    i = 0
    while i < n:
        p = rng.uniform(low=-boxl_half, high=boxl_half, size=dim)
        if np.sum(p**2) < r_sqr:
            continue
        points[i] = p
        i += 1
    return points + center


def _get_default_center(center, dim: int) -> np.ndarray:
    if center is None:
        return np.zeros(shape=dim)
    if len(center) != dim:
        raise ValueError(f'Dimension of "center" ({len(center)}) does not '
                         f'match the specified dimension ({dim}).')
    return np.array(center)


def random_chars(n: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(n))
