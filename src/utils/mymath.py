from __future__ import annotations

import functools
from dataclasses import dataclass
from math import gamma, pi
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def sphere_volume(radius):
    return 4.0 / 3.0 * pi * radius**3


def ball_volume(radius, dimension: int = 3):
    return pi ** (dimension / 2.0) / gamma(dimension / 2.0 + 1) * radius**dimension


def ball_radius(volume, dimension: int = 3):
    inv_dim = 1.0 / dimension
    return gamma(dimension / 2.0 + 1) ** inv_dim / pi**0.5 * volume**inv_dim


def rot_a_onto_b_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if abs(c + 1) < 1e-12:
        raise ValueError(
            f"Vectors {list(a)} and {list(b)} point in " "opposite directions."
        )

    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])  # type: ignore
    R = np.eye(3) + Vx + Vx @ Vx / (1 + c)
    return R


def cylinder_points(
    a: np.ndarray, b: np.ndarray, radius: float, nt: int = 100, nv: int = 50
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    axis = b - a  # type: ignore
    height = np.linalg.norm(axis)
    v = np.linspace(0, height, nv)
    theta = np.linspace(0, 2 * np.pi, nt)

    theta, v = np.meshgrid(theta, v)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = v

    points = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    R = rot_a_onto_b_matrix(np.array([0.0, 0.0, 1.0]), axis)
    points = (R @ points).T
    points += a

    x = points[:, 0].reshape(x.shape)
    y = points[:, 1].reshape(y.shape)
    z = points[:, 2].reshape(z.shape)

    return points, (x, y, z)


@dataclass
class BestFitPlaneResult:
    projected_points: (
        np.ndarray
    )  # 2D coordinates of points projected onto the best-fit plane
    plane_normal: np.ndarray  # Normal vector of the best-fit plane
    principal_axes: (
        np.ndarray
    )  # Two main principal axes of the plane (first two singular vectors)
    center_of_mass: np.ndarray  # Center of mass of the original 3D points


def project_points_onto_best_plane(points: np.ndarray) -> BestFitPlaneResult:
    assert points.ndim == 2
    assert points.shape[1] == 3

    points = np.copy(points).T
    com = np.mean(points, axis=1, keepdims=True)
    points -= com
    svd = np.linalg.svd(points)
    left = svd[0]
    plane_normal = left[:, -1]

    projected_points = np.vstack((left[:, 0:1].T @ points, left[:, 1:2].T @ points))
    assert projected_points.shape[1] == points.shape[1]
    assert projected_points.shape[0] == 2

    points += com

    return BestFitPlaneResult(
        projected_points=projected_points.T,
        plane_normal=plane_normal,
        principal_axes=left[:, 0:2].T,
        center_of_mass=com.flatten(),
    )


def plane_slice_mask(
    points: np.ndarray, normal: np.ndarray, origin: np.ndarray, thickness: float
) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    distance_from_plane = np.abs(np.dot(points - origin, normal))
    return distance_from_plane > thickness / 2


def quat_to_director(quat):
    return np.array(
        [
            2 * (quat[1] * quat[3] + quat[0] * quat[2]),
            2 * (quat[2] * quat[3] - quat[0] * quat[1]),
            quat[0] * quat[0]
            - quat[1] * quat[1]
            - quat[2] * quat[2]
            + quat[3] * quat[3],
        ]
    )


@dataclass(frozen=True)
class Ellipsoid:
    axes: np.ndarray
    abc: np.ndarray
    center: np.ndarray

    def _transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points to ellipsoid body frame"""
        return (points - self.center) @ self.axes.T  # type: ignore

    def _equation(self, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
        points = self._transform_points(points)
        return np.sum(points**2 / (self.abc + tol) ** 2, axis=-1)

    def are_points_inside(self, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
        return self._equation(points, tol) <= 1

    def are_points_on_surface(
        self, points: np.ndarray, tol: float = 0.0, tol_out: Optional[float] = None
    ) -> np.ndarray:
        tol_out = -tol if tol_out is None else tol_out
        assert tol_out > -tol
        return self.are_points_inside(points, tol_out) * ~self.are_points_inside(
            points, -tol
        )

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * float(np.prod(self.abc))

    @property
    def unnormalized_axes(self) -> np.ndarray:
        return self.axes * self.abc[..., np.newaxis]

    @functools.cached_property
    def abc_cartesian(self) -> np.ndarray:
        # s = (self.axes / self.abc[..., np.newaxis]).sum(axis=0)
        # return 1 / s
        return np.abs(self.unnormalized_axes).sum(axis=0) / 3

    def plot(self, ax, **kwargs) -> None:
        plot_ellipsoid(ax, *self.unnormalized_axes, center=self.center, **kwargs)


class PointCloud:
    def __init__(self, points: np.ndarray):
        self._points = np.copy(points)

    @property
    def points(self) -> np.ndarray:
        return self._points

    @functools.cached_property
    def aabb(self) -> np.ndarray:
        return np.array(
            [
                np.min(self.points, axis=0),
                np.max(self.points, axis=0),
            ]
        )

    @property
    def n(self) -> int:
        return self.points.shape[0]

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @functools.cached_property
    def com(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @functools.cached_property
    def com_positions(self) -> np.ndarray:
        return self.points - self.com  # type: ignore

    @functools.lru_cache()
    def gaussian_kde(self, bw_method=None):
        return stats.gaussian_kde(
            self.points.T,
            bw_method=bw_method,
        )

    def density_at(self, points: np.ndarray, **kde_kwargs) -> np.ndarray:
        return self.gaussian_kde(**kde_kwargs)(points.T).T

    @functools.cached_property
    def moment_of_inertia_tensor(self) -> np.ndarray:
        if self.dim == 3:
            x, y, z = self.com_positions.T
            Ixx = np.sum(y**2 + z**2)
            Iyy = np.sum(x**2 + z**2)
            Izz = np.sum(x**2 + y**2)
            Ixy = -np.sum(x * y)
            Ixz = -np.sum(x * z)
            Iyz = -np.sum(y * z)
            return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        elif self.dim == 2:
            x, y = self.com_positions.T
            Ixx = np.sum(y**2)
            Iyy = np.sum(x**2)
            Ixy = -np.sum(x * y)
            return np.array([[Ixx, Ixy], [Ixy, Iyy]])
        else:
            raise NotImplementedError()

    @functools.cached_property
    def principal_moments_of_inertia(self) -> np.ndarray:
        return self._moment_of_inertia_tensor_eig_decomp[0]

    @functools.cached_property
    def principal_axes_of_inertia(self) -> np.ndarray:
        return self._moment_of_inertia_tensor_eig_decomp[1]

    @functools.cached_property
    def ellipsoid_axes_of_inertia(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.dim == 3

        eigvals, eigvecs = self._moment_of_inertia_tensor_eig_decomp
        assert np.all(eigvals.sum() - 2 * eigvals >= 0), (eigvals.sum(), 2 * eigvals)
        ellipsoid_axes = np.sqrt(2.5 / self.n * (eigvals.sum() - 2 * eigvals))
        return ellipsoid_axes, eigvecs

    @functools.cached_property
    def ellipsoid_of_inertia(self) -> Ellipsoid:
        assert self.dim == 3

        ellipsoid_axes, eigvecs = self.ellipsoid_axes_of_inertia
        return Ellipsoid(axes=eigvecs, abc=ellipsoid_axes, center=self.com)

    @functools.cached_property
    def kd_tree(self):
        from scipy import spatial

        return spatial.KDTree(self.points)

    @functools.cached_property
    def _moment_of_inertia_tensor_eig_decomp(self) -> Tuple[np.ndarray, np.ndarray]:
        eigvals, eigvecs = np.linalg.eig(self.moment_of_inertia_tensor)
        eigvecs = eigvecs.T

        sort_indices = np.argsort(eigvals)
        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[sort_indices]

        eigvecs *= np.sign(eigvecs[:, 0])[..., np.newaxis]
        assert np.all(eigvecs[:, 0] >= 0)
        return eigvals, eigvecs

    def __hash__(self):
        return hash(id(self))


def uniform_gridpoints(low, high, n, stacked=True) -> np.ndarray:
    dim = len(low)
    try:
        iter(n)
    except TypeError:
        n = [n] * dim

    lines = [np.linspace(a, b, k) for a, b, k in zip(low, high, n, strict=True)]
    XYZ = np.meshgrid(*lines)
    if stacked:
        return np.stack(XYZ, axis=-1).reshape(np.prod(n), dim)
    return XYZ


def random_rotation_matrix() -> np.ndarray:
    rng = np.random.default_rng()
    A = rng.random(size=(3, 3))
    U, _, _ = np.linalg.svd(A, full_matrices=True)
    return U


def plot_ellipsoid(
    ax, v1, v2, v3, center=None, color: Any = "blue", wire: bool = True
) -> None:
    if center is None:
        center = np.zeros(3)
    center = np.array(center)

    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)

    # Define the angles of rotation
    n = 10
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    # Define the x, y, and z coordinates of the ellipsoid surface
    x = a * np.outer(np.cos(u), np.sin(v))  # type: ignore
    y = b * np.outer(np.sin(u), np.sin(v))  # type: ignore
    z = c * np.outer(np.ones_like(u), np.cos(v))  # type: ignore
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    assert points.shape == (n**2, 3)

    # Normalize the vectors to get their unit vectors
    u1 = v1 / a
    u2 = v2 / b
    u3 = v3 / c

    # Rotate points
    rot_matrix = np.array([u1, u2, u3]).T
    points = (rot_matrix @ points.T).T

    # Plot the ellipsoid surface
    points += center
    x = points[:, 0].reshape(n, n)
    y = points[:, 1].reshape(n, n)
    z = points[:, 2].reshape(n, n)
    if wire:
        ax.plot_wireframe(x, y, z, color=color)
    else:
        ax.plot_surface(x, y, z, color=color)


def plot_sphere(ax, radius: float, center=None, color: Any = "blue") -> None:
    v1, v2, v3 = np.eye(3) * radius
    plot_ellipsoid(ax, v1, v2, v3, center=center, color=color, wire=False)


def online_mean(data_iterator):
    """Compute mean iteratively."""
    mean = None
    for n, data in enumerate(data_iterator, start=1):
        if mean is None:
            mean = 0*data

        mean += (data - mean) / n

    return mean


def online_mean_and_ci(data_iterator, alpha) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Initialize mean and variables for variance
    mean = None
    M2 = 0  # Sum of squares of differences from the current mean

    for n, data in enumerate(data_iterator, start=1):
        if mean is None:
            # Initialize on the first iteration
            mean = 0*data
            M2 = 0*data

        # Compute the new mean iteratively
        delta = data - mean
        mean += delta / n
        
        # Update M2 for variance calculation
        M2 += delta * (data - mean)

    assert mean is not None

    # Compute the variance (sample variance)
    variance = M2 / (n - 1)

    # Standard error (SE)
    standard_error = np.sqrt(variance) / np.sqrt(n)

    # 100*(1 - alpha/2)% confidence interval. Two-tailed CI (thus / 2)
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)  
    ci_lower = mean - t_value * standard_error
    ci_upper = mean + t_value * standard_error

    return mean, ci_lower, ci_upper



def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    rng = np.random.default_rng()
    points = rng.random(size=(10000, 3)) - 0.5
    # ax.plot(*points.T, "k.")

    e1 = Ellipsoid(random_rotation_matrix(), np.array([0.1, 0.2, 0.3]), np.zeros(3))
    print(np.count_nonzero(e1.are_points_inside(points)))
    points = points[e1.are_points_inside(points)]

    pc = PointCloud(points)
    e2 = pc.ellipsoid_of_inertia

    ax.plot(*points.T, "g.")
    plot_ellipsoid(ax, *e1.unnormalized_axes, color="red")
    plot_ellipsoid(ax, *e2.unnormalized_axes, color="blue")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


if __name__ == "__main__":
    main()
