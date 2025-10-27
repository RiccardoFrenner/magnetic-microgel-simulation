import numpy as np
import matplotlib.pyplot as plt
import pytest


def center_points(points):
    """Centers a point cloud to have its center of mass at the origin."""
    center = np.mean(points, axis=0)
    return points - center


def generate_directions(num_directions, dimension):
    """Generates uniformly distributed directions on the unit hypersphere."""
    if dimension == 2:
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        directions = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
    else:
        directions = np.random.randn(num_directions, dimension)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def find_plane_distances(points, directions, p):
    """Calculates distances from the origin to planes along given directions."""
    points = center_points(points)
    plane_distances = []
    for direction in directions:
        projections = np.dot(points, direction)
        distance = np.percentile(projections, p)
        plane_distances.append(distance)
    return np.array(plane_distances)


def iterative_plane_fitting(
    points,
    directions,
    p_max=5,
    relative_threshold=0.05,
    min_dx_fraction=0.001,
):
    """Iteratively finds plane positions using adaptive step size (dx)."""

    num_points = points.shape[0]

    # Calculate AABB size
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    aabb_size = np.max(max_bounds - min_bounds)

    # Initialize distances and dx
    plane_distances = find_plane_distances(points, directions, p=0)
    max_plane_distances = find_plane_distances(points, directions, p=p_max)
    dx = aabb_size * 0.1  # Initial step size (relative to AABB)
    min_dx = aabb_size * min_dx_fraction

    for i, direction in enumerate(directions):
        while dx > min_dx:
            current_dist = plane_distances[i]

            # Count points behind the plane before move
            points_behind_before = np.sum(np.dot(points, direction) <= current_dist)

            # Move the plane closer to the center by dx
            new_dist = current_dist + dx

            # Count points behind the plane after move
            points_behind_after = np.sum(np.dot(points, direction) <= new_dist)

            # Calculate relative change in points behind the plane
            change_relative = (points_behind_after - points_behind_before) / num_points

            if change_relative > relative_threshold:
                dx /= 2  # Halve the step size
            else:
                if new_dist > max_plane_distances[i]:
                    break
                plane_distances[i] = new_dist  # Update plane distance

        dx = aabb_size * 0.1  # Reset dx for the next plane

    return plane_distances


def estimate_volume_mc(points, plane_distances, directions, num_samples=10000):
    """Estimates volume using rejection sampling (Monte Carlo)."""
    points = center_points(points)

    dimension = points.shape[1]
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)

    # Generate random points within the AABB (Axis-Aligned Bounding Box)
    random_points = (
        np.random.rand(num_samples, dimension) * (max_bounds - min_bounds) + min_bounds
    )

    # Reject points outside the planes
    are_points_inside = np.full(random_points.shape[0], True)
    for distance, direction in zip(plane_distances, directions):
        are_points_inside &= np.dot(random_points, direction) > distance

    inside_points = random_points[are_points_inside]

    # Estimate volume as fraction of inside points * volume of AABB
    aabb_volume = np.prod(max_bounds - min_bounds)
    volume_estimate = (len(inside_points) / num_samples) * aabb_volume
    return volume_estimate, inside_points


def plot_plane2D(normal, distance, extend=5, ax=None):
    """Plots a plane in 2D given its normal vector and distance from origin."""
    if ax is None:
        ax = plt.gca()

    point_on_plane = normal * distance  # Point on plane along the normal direction
    tangent = np.array([-normal[1], normal[0]])  # Tangent vector to the plane

    start_point = point_on_plane - extend * tangent
    end_point = point_on_plane + extend * tangent

    ax.plot(
        [start_point[0], end_point[0]],
        [start_point[1], end_point[1]],
    )
    ax.scatter(*point_on_plane, color="red")  # Mark the point on plane (optional)


def estimate_volume_hyperplanes(
    points: np.ndarray,
    visualize=False,
    n_directions=None,
    p=2,
    method="percentile",
    p_max=5,
    relative_threshold=0.05,
    min_dx_fraction=0.001,
):
    """Estimates the volume of a point cloud using hyperplanes.

    Args:
        points: The input point cloud as a numpy array.
        visualize: If True, plots the 2D points and planes.
        n_directions: The number of directions to use for generating planes.
                      If None, defaults to 8 * dimension.
        p: The percentile used to determine the plane distance (for percentile method).
        method: The method for fitting the planes ("percentile" or "iterative").
        p_initial: The initial percentile for iterative fitting.
        p_max: The maximum percentile for iterative fitting.
        relative_threshold: The relative threshold for iterative fitting.
        min_dx_fraction: The minimum fraction of AABB size for dx in iterative fitting.

    Returns:
        The estimated volume.
    """
    dimension = points.shape[1]
    if n_directions is None:
        n_directions = 8 * dimension
    directions = generate_directions(n_directions, dimension)

    if method == "percentile":
        plane_distances = find_plane_distances(points, directions, p)
    elif method == "iterative":
        plane_distances = iterative_plane_fitting(
            points,
            directions,
            p_max=p_max,
            relative_threshold=relative_threshold,
            min_dx_fraction=min_dx_fraction,
        )
    else:
        raise ValueError("Invalid method. Choose 'percentile' or 'iterative'.")

    volume, inside_points = estimate_volume_mc(
        points, plane_distances, directions, num_samples=10000
    )

    if dimension == 2 and visualize:
        centered_points = center_points(points)
        plt.figure(figsize=(8, 6))
        plt.plot(0, 0, "kx")
        plt.scatter(
            centered_points[:, 0], centered_points[:, 1], label="Centered Points"
        )
        for distance, direction in zip(plane_distances, directions):
            plot_plane2D(direction, distance, extend=7)
            arrow_points = np.array(
                [(direction * distance), (direction * (distance + 1))]
            )
            plt.plot(*arrow_points.T, "k-")

        plt.axis("equal")
        plt.legend()
        plt.title("Point Cloud and Planes")

        plt.scatter(*inside_points.T, color="red", alpha=0.02)

    return volume


def test_iterative_larger_volume():
    points = np.random.rand(100, 2) * 10 - 5
    p = 2
    volume_percentile = estimate_volume_hyperplanes(
        points, method="percentile", p=p, n_directions=9
    )
    volume_iterative = estimate_volume_hyperplanes(
        points, method="iterative", p_initial=0, p_max=5, n_directions=9
    )
    assert volume_iterative > volume_percentile
