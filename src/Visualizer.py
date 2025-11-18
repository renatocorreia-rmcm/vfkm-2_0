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

from src.PolygonalPath2D import PolygonalPath2D
from src.main import load_curves

matplotlib.use('Qt5Agg')  # or 'QtAgg', 'Qt5Agg', et
from matplotlib import pyplot as plt




class Visualizer:  # todo: make read file to avoid re-runig VFKM just for visualization

    paths: np.ndarray[PolygonalPath2D]

    amount_clusters: int

    grid: Grid  # todo: bounding box and resolution - use raw parameters or re-create grid inside this object


    def __init__(
            self
    ):

        with open("output/experiment.txt", 'r') as experiment:
            self.paths, bounding_box = load_curves(experiment.readline())
            self.amount_clusters = int(experiment.readline().strip())

        with open("output/vf_r.txt", 'r') as vf:
            self.grid = Grid(
                bounding_box=bounding_box,
                resolution=int(vf.readline())**0.5
            )



    def visualize_vector_fields(self, resolution: tuple[int, int]) -> None:
        for i_cluster in range(self.amount_clusters):

            # --- Create a new figure for each cluster ---
            plt.figure(figsize=(8, 8))
            plt.title(f"Vector Field {i_cluster}")

            # --- Interpolated (fine) grid ---
            U: np.ndarray[float] = np.zeros(shape=resolution * resolution[0], dtype=float)
            V: np.ndarray[float] = np.zeros(shape=resolution * resolution[1], dtype=float)

            x = np.arange(0, resolution, 1)
            y = np.arange(0, resolution, 1)
            X, Y = np.meshgrid(x, y)

            for line in range(resolution[1]):
                for col in range(resolution[0]):
                    index: int = line * resolution[0] + col

                    relative_coordinates: np.ndarray[float] = np.array([line, col]) / np.array(resolution)

                    U[index], V[index] = cluster.get_vector(relative_coordinates=relative_coordinates, grid=self.grid)[0]  # TODO: HOW TO INTERPOLATE

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

    def visualize_curves(self) -> None:
        for cluster in self.clusters:
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

            plt.show()
