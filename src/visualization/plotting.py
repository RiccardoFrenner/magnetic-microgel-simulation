import itertools
from typing import Any

import numpy as np

import constants


def specie_alpha(species: np.ndarray) -> np.ndarray:
    alphas = np.where(species == constants.BEAD.ptype, 1.0, 0.5)
    return alphas


def specie_size(species: np.ndarray) -> np.ndarray:
    sizes = np.where(species == constants.NANOPARTICLE.ptype, 1000.0, 100.0)
    return sizes


def myscatter(ax, pos: np.ndarray, species: np.ndarray, dim: int = 2, only_beads: bool = False):
    types_to_remove = constants.ION_SPECIES + [constants.CROSSLINK_AGENT.ptype]

    if only_beads:
        types_to_remove = set(np.unique(species))
        types_to_remove -= set([constants.BEAD.ptype])
        types_to_remove = list(types_to_remove)

    should_remove = np.isin(species, types_to_remove)
    pos = pos[~should_remove]
    species = species[~should_remove]

    points = pos[:, :dim]
    ax.scatter(
        *points.T,
        c=species,
        s=specie_size(species),
        alpha=specie_alpha(species),
    )


def plot_box(ax, box_size: float, color: Any = "black", dim: int = 2) -> None:
    cube_points = list(itertools.product([0, box_size], repeat=dim))
    for p1, p2 in itertools.combinations(cube_points, 2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        if np.linalg.norm(p1 - p2) > box_size:
            continue
        ax.plot(*np.array([p1, p2]).T, color=color)
