import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

PROJECT_ROOT = (Path(__file__) / "../../..").resolve()
print("=" * 100)
print(PROJECT_ROOT)
print("=" * 100)
HYDRO_RADIUS_EXE = PROJECT_ROOT / "scripts/data_processing/hydro_radius"


class HydroRadiusComputer:
    @staticmethod
    def _tool_backup(file_path, skip=1000):
        # Use only a subset of points to compute the
        # hydrodynamic radius to save memory and computation time
        points = np.load(file_path)[::skip]
        N = len(points)
        rh_inv = sum(
            1.0 / np.linalg.norm(points[j] - points[i])
            for i in range(N)
            for j in range(i + 1, N)
        )

        rh_inv *= 2.0 / (N * (N - 1))

        return (str(file_path), 1.0 / rh_inv)

    @staticmethod
    def _check_tool_availability():
        return HYDRO_RADIUS_EXE.exists()

    @staticmethod
    def _check_gnu_parallel():
        try:
            subprocess.run(
                ["parallel", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def _call_external_tool(file_path):
        try:
            result = subprocess.run(
                [str(HYDRO_RADIUS_EXE), file_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output = result.stdout.decode().strip()
            try:
                file_path, value = output.split(",")
            except ValueError:
                value = output.split(",")[0]
            return file_path, float(value)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error calling external tool: {e}")

    @staticmethod
    def _call_backup_tool(file_path):
        return HydroRadiusComputer._tool_backup(file_path)

    @staticmethod
    def process_file(file_path):
        if HydroRadiusComputer._check_tool_availability():
            return HydroRadiusComputer._call_external_tool(file_path)
        else:
            return HydroRadiusComputer._call_backup_tool(file_path)

    @staticmethod
    def _parallel_process_with_multiprocessing(file_paths):
        print("Using multiprocessing")
        with Pool() as pool:
            results = pool.map(HydroRadiusComputer.process_file, file_paths)
        print(results)
        return pd.DataFrame.from_records(
            index=list(range(len(results))),
            data=results,
            columns=["paths", "hydro_radius"],
        )

    @staticmethod
    def _parallel_process_with_gnu_parallel(file_paths) -> pd.DataFrame:
        print("Using gnu parallel with fast tool")
        file_list = " ".join(map(str, file_paths))
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
            for file in file_list:
                temp.write(file + "\n")
            temp.close()

        command = f"parallel --argfile {temp.name} {str(HYDRO_RADIUS_EXE)}"
        # command = f"parallel {str(HYDRO_RADIUS_EXE)} ::: {file_list}"
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = result.stdout.decode().strip().split("\n")
        result_file_paths = [str(line.split(",")[0]) for line in output]
        values = [float(line.split(",")[1]) for line in output]
        print(result_file_paths)
        print(values)
        return pd.DataFrame({"paths": result_file_paths, "hydro_radius": values})

    @staticmethod
    def process_files(file_paths) -> pd.DataFrame:
        if HydroRadiusComputer._check_tool_availability():
            if HydroRadiusComputer._check_gnu_parallel():
                results = HydroRadiusComputer._parallel_process_with_gnu_parallel(
                    file_paths
                )
            else:
                results = HydroRadiusComputer._parallel_process_with_multiprocessing(
                    file_paths
                )
        else:
            results = HydroRadiusComputer._parallel_process_with_multiprocessing(
                file_paths
            )

        results["timestep"] = list(
            map(lambda s: int(s.split("_")[-1].split(".")[0]), results["paths"])
        )
        return results


class PointCloudVolume:
    @staticmethod
    def bounding_box_volume(points: np.ndarray) -> float:
        """Compute the volume of the axis-aligned bounding box of the points."""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        volume = np.prod(max_coords - min_coords)
        return volume

    @staticmethod
    def convex_hull_volume(points: np.ndarray) -> float:
        """Compute the volume of the convex hull of the points."""
        hull = ConvexHull(points)
        return hull.volume

    @staticmethod
    def pca_extent_volume(points: np.ndarray) -> float:
        """Compute the approximate volume based on PCA extents."""
        pca = PCA(n_components=3)
        pca.fit(points)
        # The variance along each principal axis
        variances = pca.explained_variance_
        # Estimate volume as the product of 2*sqrt(variance) along each axis
        extent_volume = float(np.prod(2 * np.sqrt(variances)))
        return extent_volume

    @staticmethod
    def mean_pairwise_distance(points: np.ndarray) -> float:
        """Compute the mean pairwise distance between points."""
        distances = pdist(points)
        mean_distance = float(np.mean(distances))
        return mean_distance

    @staticmethod
    def max_pairwise_distance(points: np.ndarray) -> float:
        """Compute the maximum pairwise distance between points."""
        return pdist(points).max()

    @staticmethod
    def radius(points, method="max"):
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        if method == "max":
            return distances.max()
        elif method == "average":
            return np.mean(distances)
        elif method == "median":
            return np.median(distances)
        else:
            raise ValueError("Invalid method. Choose 'max', 'average', or 'median'.")

    @staticmethod
    def radius_of_gyration(points: np.ndarray) -> float:
        """Compute the radius of gyration of the points."""
        centroid = np.mean(points, axis=0)
        squared_distances = np.sum((points - centroid) ** 2, axis=1)
        radius_gyration = np.sqrt(np.mean(squared_distances))
        return radius_gyration
    
    @staticmethod
    def hydrodynamic_radius(points: np.ndarray) -> float:
        """Compute the hydrodynamic radius of the points."""
        N = len(points)
        
        OPTIMIZATION_THRESHOLD = 10000

        if N < OPTIMIZATION_THRESHOLD:
            # Optimized method for smaller N using pdist
            distances = pdist(points)
            rh_inv = float(np.sum(1.0 / distances))
            rh_inv *= 2.0 / (N * (N - 1))
        else:
            rh_inv = float(sum(
                1.0 / np.linalg.norm(points[j] - points[i])
                for i in range(N)
                for j in range(i + 1, N)
            ))

            rh_inv *= 2.0 / (N * (N - 1))

        return 1.0 / rh_inv
