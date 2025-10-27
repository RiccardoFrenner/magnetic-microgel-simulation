"""
This script quantifies the void space (pore size) within the simulated polymer gel structure. It selects dense spatial sampling points using blue noise generation and uses ESPResSo and steepest descent minimization to determine the maximum radius a fixed probe particle can attain at these locations. The results, including the estimated pore sizes and probe coordinates, are saved to the `processed/pores` directory.
"""

import sys
from pathlib import Path

sys.path.append(str((Path(__file__) / "../../../src").resolve()))

import networkx as nx
import numpy as np
import tqdm
from scipy.spatial import KDTree

import espressomd
import espressomd.analyze
import espressomd.interactions
import espressomd.polymer

import analysis
import analysis.network
import common
import constants

if "system" not in globals() and "system" not in locals():
    system = None


def _setup_system(
    system: espressomd.System,
    positions: np.ndarray,
    graph: nx.Graph,
):
    """Sets up the ESPResSo system with particles and bonds."""
    system.time_step = constants.TIME_STEP
    system.cell_system.skin = 0.4
    system.cell_system.set_regular_decomposition(use_verlet_lists=True)
    system.comfixed.types = [0]
    system.part.add(type=np.full(len(positions), 0), pos=positions)
    fene = espressomd.interactions.FeneBond(**constants.FENE_PARAMS)
    system.bonded_inter.add(fene)
    for i, j in graph.edges:
        system.part.by_id(i).add_bond((fene, system.part.by_id(j)))
    system.non_bonded_inter[0, 0].wca.set_params(
        epsilon=constants.EPS, sigma=constants.BEAD.diameter
    )


def _set_probe_size(system: espressomd.System, radius: float):
    """Sets up the probe particle in the ESPResSo system."""

    system.non_bonded_inter[0, 1].wca.set_params(
        epsilon=constants.EPS, sigma=constants.BEAD.radius + radius
    )
    system.non_bonded_inter[1, 1].wca.set_params(
        epsilon=constants.EPS, sigma=2 * radius
    )


def _generate_blue_noise_points(
    n_points: int, positions: np.ndarray, rejection_fun, rng
):
    """Generates blue noise points within the bounding box of positions."""
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    ranges = max_coords - min_coords
    threshold_dist = 0.1 * np.mean(ranges)  # Precompute distance threshold

    points = []
    kdtree = None  # Use k-d tree for efficient neighbor searches

    while len(points) < n_points:
        # Generate candidate points in batches
        candidates = min_coords + rng.uniform(size=(100, len(min_coords))) * ranges

        for new_point in candidates:
            if rejection_fun(new_point):
                continue

            if kdtree is None:  # First point
                points.append(new_point)
                kdtree = KDTree(np.array(points))
                continue

            # Check if the new point satisfies blue noise criteria
            distances, _ = kdtree.query(new_point, k=1)
            if distances > threshold_dist:
                points.append(new_point)
                kdtree = KDTree(np.array(points))  # Update k-d tree

                if len(points) >= n_points:
                    break

    return np.array(points)


def draw_random_spatial_points(
    positions: np.ndarray,
    min_neighbors: int,
    search_radius: float,
    n_points: int,
    rng,
):
    """
    Draws multiple random points in the spatial region of the graph using rejection sampling and KD trees.
    """
    kd_tree = KDTree(positions)

    def rejection_fun(point):
        neighbor_indices = kd_tree.query_ball_point(point, search_radius, p=2)
        neighbors = len(neighbor_indices)

        if neighbors < min_neighbors:
            return True
        return False

    blue_noise_points = _generate_blue_noise_points(
        n_points, positions, rejection_fun, rng
    )

    return blue_noise_points


def _run_steepest_descent(
    system: espressomd.System,
    current_radius: float,
    patience: int = 20,
    steps_per_it: int = 10,
    verbose: int = 0,
    min_delta: float = 0.1,
):
    """Runs steepest descent until convergence or max radius is reached."""

    min_dist_patience_counter = 0
    prev_min_dist = np.inf

    if verbose > 1:
        print(f"  Current radius: {current_radius:.1e}")

    while True:
        system.integrator.run(steps_per_it)

        # Calculate distances between type 0 and type 1 particles
        distances = np.linalg.norm(
            system.part.select(type=0).pos - system.part.select(type=1).pos.squeeze(),
            axis=1,
        )
        current_min_dist = np.min(distances)
        n_overlaps = np.count_nonzero(distances < current_radius)

        if verbose > 2:
            print(f"    Current min distance: {current_min_dist:.1e}")

        if n_overlaps == 0:
            if verbose > 1:
                print("    SD Converged! No overlaps found.")
            return True  # Converged

        if current_min_dist - prev_min_dist < min_delta:
            min_dist_patience_counter += 1
            if min_dist_patience_counter >= patience:
                if verbose > 1:
                    print(
                        f"    SD NOT converged due to overlap with {n_overlaps} particles and no change in min distance."
                    )
                return False  # Max radius reached due to no min distance change
        else:
            min_dist_patience_counter = 0  # Reset count if min distance changes
            prev_min_dist = current_min_dist

        if np.isnan(current_min_dist):
            if verbose > 1:
                print("    Failed due to nan")
            return False


def estimate_pore_size_at(
    system: espressomd.System,
    graph: nx.Graph,
    positions: np.ndarray,
    at: np.ndarray,
    max_radius: float,
    patience: int,
    n_radius_steps: int,
    verbose: int = 0,
):
    """
    Estimates the pore size at a given point by incrementally increasing the radius of a probe particle.
    """

    _setup_system(system, positions, graph)
    system.integrator.set_steepest_descent(
        f_max=0, gamma=0.01, max_displacement=0.01 * constants.BEAD.diameter
    )

    COM = np.mean(positions, axis=0)

    # add probe
    system.part.add(pos=at, type=1, fix=np.full(3, True))
    radius = system.analysis.min_dist(p1=[0], p2=[1])*0.99

    estimated_radius = 0.0
    is_minimization_possible = True
    while is_minimization_possible:
        if radius > max_radius:
            break

        _set_probe_size(system, radius=radius)

        if verbose > 2:
            print(
                f"SD for particle at distance {np.linalg.norm(COM - at):.2f} with size {radius:g}"
            )

        for _ in range(5):
            try:
                system.integrator.run(5)
            except Exception:
                is_minimization_possible = False
                break

            dist = system.analysis.min_dist(p1=[0], p2=[1])
            if dist*0.99 > radius:
                estimated_radius = radius
                radius = dist*0.99
                break
        else:
            is_minimization_possible = False


    return estimated_radius, system.part.select(type=0).pos


def estimate_pore_size(
    graph: nx.Graph,
    positions: np.ndarray,
    min_neighbors: int,
    max_radius: float,
    rng,
    n_points: int = 1,
    patience: int = 20,
    n_radius_steps: int = 10,
    verbose: int = 0,
):
    """
    Estimates the pore size of the graph structure.
    """

    search_radius = constants.BEAD.radius * 10
    probe_points = draw_random_spatial_points(
        positions, min_neighbors, search_radius, n_points, rng
    )
    if probe_points is None:
        raise RuntimeError(
            "Could not find enough suitable points inside the graph's spatial region."
        )

    global system
    pore_sizes = []
    final_part_positions = []
    BOX_L = positions.max() - positions.min()
    for point in probe_points:
        if system is None:
            if verbose > 1:
                print(">>> Recreating system")
            system = espressomd.System(box_l=[BOX_L] * 3)
        system.part.clear()
        size, final_pos = estimate_pore_size_at(
            system,
            graph,
            positions,
            at=point,
            max_radius=max_radius,
            patience=patience,
            n_radius_steps=n_radius_steps,
            verbose=verbose,
        )
        pore_sizes.append(size)
        final_part_positions.append(final_pos)

    return pore_sizes, probe_points, final_part_positions


def main(gel_dir_path: Path, n_points: int, more_points: bool, clear: bool):
    gel_dir = common.GelDir(gel_dir_path)
    assert gel_dir.path.exists()

    out_dir = gel_dir.path / "processed/pores"
    out_dir.mkdir(exist_ok=True, parents=True)
    if clear:
        import shutil

        shutil.rmtree(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

    if not more_points and (out_dir / "pore_sizes.npy").exists():
        print("Already done.")
        return

    gel_proxy = analysis.GelProxy.from_gel_dir(gel_dir.path)
    G = analysis.network.graph_from_gel_proxy(gel_proxy)
    bead_points = np.load(gel_dir.beads_gel_eq()["path"].iloc[-1])

    rng = np.random.default_rng()

    pore_sizes, probe_points, final_part_positions = estimate_pore_size(
        graph=G,
        positions=bead_points,
        min_neighbors=40,
        max_radius=10,
        rng=rng,
        n_points=n_points,
        patience=10,
        n_radius_steps=10,
        verbose=5,
    )

    if more_points and (out_dir / "pore_sizes.npy").exists():
        print(
            "Before:",
            np.shape(pore_sizes),
            np.shape(probe_points),
            np.shape(final_part_positions),
        )
        pore_sizes = np.concatenate([np.load(out_dir / "pore_sizes.npy"), pore_sizes], axis=0)
        probe_points = np.concatenate(
            [np.load(out_dir / "probe_points.npy"), probe_points], axis=0
        )
        final_part_positions = np.concatenate(
            [np.load(out_dir / "final_part_positions.npy"), final_part_positions],
            axis=0,
        )
        print(
            "After:",
            np.shape(pore_sizes),
            np.shape(probe_points),
            np.shape(final_part_positions),
        )

    np.save(out_dir / "pore_sizes.npy", pore_sizes)
    np.save(out_dir / "probe_points.npy", probe_points)
    np.save(out_dir / "final_part_positions.npy", final_part_positions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate pore size from gel data.")
    parser.add_argument("gel_dir_path", type=Path, help="Path to the gel directory.")
    parser.add_argument("--n_points", type=int, default=2)
    parser.add_argument("--more_points", action="store_true", default=False)
    parser.add_argument("--clear", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
