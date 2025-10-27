from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class RadialNumberDensityResult:
    r: np.ndarray
    number_density: np.ndarray
    bins: np.ndarray
    all_distances: np.ndarray


def compute_radial_number_density(
    points: np.ndarray,
    relative_to: Optional[Union[np.ndarray, float]] = None,
    distances: Optional[np.ndarray] = None,
    use_multi_bin_smooth: bool = False,
    **hist_kwargs,
) -> RadialNumberDensityResult:
    if relative_to is None:
        relative_to = np.mean(points, axis=0)
    if distances is None:
        distances = np.linalg.norm(points - relative_to, axis=1)
    assert distances is not None
    hist_kwargs.setdefault("bins", "auto")

    density, bin_edges = np.histogram(distances, **hist_kwargs)
    if use_multi_bin_smooth:
        n = 4
        dx = (bin_edges[1] - bin_edges[0]) * 4
        for i in range(n):
            hist_kwargs["bins"] = bin_edges + dx/2**i
            density += np.histogram(distances, **hist_kwargs)[0]
            hist_kwargs["bins"] = bin_edges - dx/2**i
            density += np.histogram(distances, **hist_kwargs)[0]
        density = density / (1 + 2*n)
    bin_volumes = 4.0 / 3.0 * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    density = density / bin_volumes
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    return RadialNumberDensityResult(
        r=bin_centers,
        number_density=density,
        bins=bin_edges,
        all_distances=distances,
    )


def rnd_from_file(p: Path, **kwargs):
    agent_points = np.load(p)
    com = agent_points.mean(axis=0)
    rnd_result = compute_radial_number_density(agent_points, relative_to=com, **kwargs)
    return rnd_result.r, rnd_result.number_density
