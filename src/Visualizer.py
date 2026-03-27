import numpy as np

from src.Cluster import Cluster
from src.Grid import Grid

from src.PolygonalPath2D import PolygonalPath2D

from matplotlib import pyplot as plt




class Visualizer:  # todo: make read file to avoid re-runig VFKM just for visualization

    grid: Grid  # todo: bounding box and resolution - use raw parameters or re-create grid inside this object

    clusters: list[Cluster]

    paths: np.ndarray[PolygonalPath2D]



    def __init__(self, clusters: list[Cluster], paths: list[PolygonalPath2D], grid: Grid):

        self.clusters = clusters
        self.grid = grid

        self.paths = np.array(paths, dtype=PolygonalPath2D)

    def visualize_vector_fields(self, resolution: int) -> None:
        for i, cluster in enumerate(self.clusters):

            # --- Create a new figure for each cluster ---
            plt.figure(figsize=(8, 8))
            plt.title(f"Vector Field {i}")

            # --- Interpolated (fine) grid ---
            U: np.ndarray[float] = np.zeros(shape=resolution * resolution, dtype=float)
            V: np.ndarray[float] = np.zeros(shape=resolution * resolution, dtype=float)

            x = np.arange(0, resolution, 1)
            y = np.arange(0, resolution, 1)
            X, Y = np.meshgrid(x, y)

            for line in range(resolution):
                for col in range(resolution):
                    index = line * resolution + col
                    relative_coordinates: np.ndarray[float] = np.array([line, col]) / resolution
                    U[index], V[index] = cluster.get_vector(relative_coordinates=relative_coordinates, grid=self.grid)[
                        0]

            plt.quiver(X, Y, U, V, label='Interpolation')

            # --- Basis (coarse) grid ---
            U_basis: np.ndarray[float] = cluster.vector_field[0]
            V_basis: np.ndarray[float] = cluster.vector_field[1]

            x_basis = np.arange(0, resolution, (resolution - 1) / (self.grid.get_resolution_x() - 1))
            y_basis = np.arange(0, resolution, (resolution - 1) / (self.grid.get_resolution_y() - 1))
            X_basis, Y_basis = np.meshgrid(x_basis, y_basis)

            # Use 'label' instead of 'legend'
            plt.quiver(X_basis, Y_basis, U_basis, V_basis, color='r', label='Vertices')

            # --- Call plt.legend() with no arguments ---
            # It will automatically find the 'label's you defined.
            plt.legend()

            plt.savefig(f"../output/vf{i}.png")

    def visualize_curves(self) -> None:
        for i, cluster in enumerate(self.clusters):
            plt.figure(figsize=(8, 8))
            plt.title(f"Curves in Cluster {cluster.name}")

            for curve in cluster.curves:
                index: int = curve.index
                path: PolygonalPath2D = self.paths[index]

                x_coords = [point.space[0] for point in path.points]
                y_coords = [point.space[1] for point in path.points]

                plt.plot(x_coords, y_coords)

            plt.xlim(self.grid.x, self.grid.x + self.grid.w)
            plt.ylim(self.grid.y, self.grid.y + self.grid.h)

            # Optional: keep aspect ratio square
            plt.gca().set_aspect("equal", adjustable="box")

            plt.savefig(f"../output/curves{i}.png")
