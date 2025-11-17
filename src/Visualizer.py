"""todo:

create visualizer class
object is like a ProblemSetting object

main recieves bool for Visualize

load np array of all polygonalpaths

access outputfile by index

this class uses:
list of clusters - use VECTOR FIELD and CURVE (could get just the index to plot from input file - avoid reduntant aligned points from tesselation)

from object - must compute position from barycentric coordinates
from file - must normalize coordinate to grid

"""
import numpy as np
from dateutil.relativedelta import relativedelta
from matplotlib.pyplot import legend

from src.Cluster import Cluster
from src.Grid import Grid

import matplotlib
matplotlib.use('Qt5Agg')  # or 'QtAgg', 'Qt5Agg', et
from matplotlib import pyplot as plt




class Visualizer:

    grid: Grid

    clusters: list[Cluster]
    curves_file: str

    def __init__(self, clusters: list[Cluster], curve_file: str, grid: Grid):

        self.clusters = clusters
        self.curves_file = curve_file
        self.grid = grid

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

            plt.show()  # This will show one plot for each cluster, one by one