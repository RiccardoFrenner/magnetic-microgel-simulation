from typing import Any, Literal

import numpy as np
import pandas as pd

import constants
from utils import mymath

NanoparticleClassifierTypes = Literal[
    "ellipsoid",
    "kde",
    "neighbors",
    "np_dens",
]

MINIMUM_ACCEPTED_DENSITY = 1e-8
INSIDE_GEL_DENSITY_FACTOR = 0.9
SURFACE_DENSITY_FACTOR = 0.4


class NanoparticleClassifier:
    LOCATIONS = pd.CategoricalDtype(
        categories=["outside", "surface", "inside"], ordered=True
    )

    def __init__(
        self,
        *,
        bead_positions: np.ndarray,
        np_positions: np.ndarray,
    ):
        self._bead_positions = bead_positions
        self._np_positions = np_positions

        self.bead_pc = mymath.PointCloud(bead_positions)
        self.np_pc = mymath.PointCloud(np_positions)

    def classify(
        self,
        method: NanoparticleClassifierTypes,
        **method_kwargs,
    ) -> tuple[pd.Series, dict[str, Any]]:
        if method == "ellipsoid":
            classes, computation_data = self._classify_nps_by_ellipsoid(**method_kwargs)
        elif method == "KDE":
            classes, computation_data = self._classify_nps_by_kde(**method_kwargs)
        elif method == "neighbors":
            classes, computation_data = self._classify_nps_by_neighbors(**method_kwargs)
        elif method == "np_dens":
            classes, computation_data = self._classify_nps_by_dens(**method_kwargs)
        else:
            raise NotImplementedError()

        assert all(classes.isin(["outside", "surface", "inside"])), classes
        return classes, computation_data

    def _classify_nps_by_kde(
        self, bw_method: float = 0.4
    ) -> tuple[pd.Series, dict[str, Any]]:
        kernel = self.bead_pc.gaussian_kde(
            bw_method=bw_method,
        )
        dens_at_nps = kernel(self._np_positions.T).T
        dens_at_nps[dens_at_nps < MINIMUM_ACCEPTED_DENSITY] = 0
        at_or_below_surf = dens_at_nps > dens_at_nps.mean() * SURFACE_DENSITY_FACTOR
        is_inside = dens_at_nps > dens_at_nps.mean() * INSIDE_GEL_DENSITY_FACTOR

        np_locations = self._default_np_classes()
        np_locations[is_inside] = "inside"
        np_locations[~is_inside * at_or_below_surf] = "surface"

        return np_locations, dict()

    def _classify_nps_by_ellipsoid(
        self,
        tol: float = constants.NANOPARTICLE.radius * 1.75,
        tol_out: float = constants.NANOPARTICLE.radius * 1.0,
    ) -> tuple[pd.Series, dict[str, Any]]:
        abc, vec = self.bead_pc.ellipsoid_axes_of_inertia
        ellipsoid = mymath.Ellipsoid(axes=vec.T, abc=abc, center=self.bead_pc.com)

        np_locations = self._default_np_classes()
        np_locations[ellipsoid.are_points_inside(self.np_pc.points, tol=-tol)] = (
            "inside"
        )
        np_locations[
            ellipsoid.are_points_on_surface(self.np_pc.points, tol=tol, tol_out=tol_out)
        ] = "surface"
        return np_locations, {"ellipsoid": ellipsoid}

    def _classify_nps_by_neighbors(
        self,
        max_bead_np_distance: float = 10.0,
        min_nearby_beads_threshold: int = 20,
    ) -> tuple[pd.Series, dict[str, Any]]:
        
        # bead neighbors of MNPs
        bead_np_neighbors = self.np_pc.kd_tree.query_ball_tree(self.bead_pc.kd_tree, r=max_bead_np_distance)
        n_np_neighbors = np.array([len(beads) for beads in bead_np_neighbors])
        assert len(n_np_neighbors) == self.np_pc.n

        np_locations = self._default_np_classes()
        np_locations[n_np_neighbors >= min_nearby_beads_threshold] = "inside"
        return np_locations, dict()

    def _classify_nps_by_dens(
        self, bw_method: float = 0.4
    ) -> tuple[pd.Series, dict[str, Any]]:
        assert self.np_pc.n > 1
        if np.all(self.np_pc.points[:, 2] == 0.0):
            self.np_pc = mymath.PointCloud(self.np_pc.points[:, :2])
            self.bead_pc = mymath.PointCloud(self.bead_pc.points[:, :2])
        from scipy import spatial

        np_d = constants.NANOPARTICLE.radius

        def volume_fraction(points):
            return spatial.ConvexHull(points).volume / (
                len(points) * mymath.ball_volume(np_d / 2, dimension=self.np_pc.dim)
            )

        mean_bead_com_distance = np.linalg.norm(
            self.bead_pc.points - self.bead_pc.com, axis=1
        ).mean()  # type: ignore
        start_indices = np.flatnonzero(
            np.linalg.norm(
                self.np_pc.points  # type: ignore
                - self.bead_pc.com,
                axis=1,
            )
            < 1.3 * mean_bead_com_distance
        ).tolist()

        unused_indices = set(range(self.np_pc.n)).difference(start_indices)

        vf0 = volume_fraction(self.np_pc.points[list(start_indices)])
        vf1 = 0.0
        while len(unused_indices) > 0:
            for i in unused_indices:
                vf1 = volume_fraction(self.np_pc.points[start_indices + [i]])
                if abs(vf0 - vf1) < 0.002 or vf1 < vf0:
                    start_indices.append(i)
                    unused_indices.remove(i)
                    vf0 = vf1
                    break
            else:
                break


        np_locations = self._default_np_classes()
        np_locations[start_indices] = "inside"

        return np_locations, dict()

    def _default_np_classes(self) -> pd.Series:
        # assume outside per default
        np_locations = pd.Series(
            ["outside"] * len(self._np_positions), dtype=self.LOCATIONS
        )
        return np_locations


def are_nanoparticles_in_gel(
    *,
    bead_positions: np.ndarray,
    np_positions: np.ndarray,
    method: NanoparticleClassifierTypes = "ellipsoid",
) -> np.ndarray:
    npclassi = NanoparticleClassifier(
        bead_positions=bead_positions, np_positions=np_positions
    )

    return (
        npclassi.classify(
            method=method,
            # only valid for nearby beads classifier
            # max_bead_np_distance=10.0,
            # min_nearby_beads_threshold=20,
        )[0]
        == "inside"
    ).to_numpy()
