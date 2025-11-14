"""

class Visualizer:
visualize curves
visuzalize vector field
visualize cluster

"""

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'QtAgg'
import matplotlib.pyplot as plt
import numpy as np

from Grid import CurveDescription, Grid
from VectorField2D import VectorField2D
from Cluster import Cluster


def visualize_curves(curves: list[CurveDescription], grid: Grid) -> None:
    """
    Plot curves of a cluster

    :param curves: List of CurveDescription objects to visualize.
    """
    print("\n PLOTTING CURVES \n")
    plt.figure()
    for i, curve in enumerate(curves):
        print(f"curve {curve.index}: \n{curve}")
        x = []
        y = []
        # get coordinate from face indices and barycentric coordinates
        for segment in curve.segments:
            coord: np.ndarray[float] = segment.endpoints[0].get_coordinates(grid=grid)
            print(f"segment: \n{coord}")
            x.append(coord[0])
            y.append(coord[1])

        plt.plot(x, y, label=f'Curve {curve.index}')

    plt.title('Curves Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()


def visualize_vector_field(vector_field: VectorField2D, grid: Grid) -> None:
    # 1. Get the 1D component arrays
    x_1d: np.ndarray[float] = vector_field[0]
    y_1d: np.ndarray[float] = vector_field[1]

    # 2. Get grid resolutions
    res_x = grid.get_resolution_x()  # Or grid.res_x
    res_y = grid.get_resolution_y()  # Or grid.res_y

    # 3. Create the "world coordinate" arrays
    # We use linspace to get the actual coordinates of the grid vertices
    x_coords = np.linspace(grid.x, grid.x + grid.w, res_x)
    y_coords = np.linspace(grid.y, grid.y + grid.h, res_y)

    # 4. Create the 2D coordinate grids
    # Xv and Yv will have shape (res_y, res_x)
    Xv, Yv = np.meshgrid(x_coords, y_coords)

    # 5. Reshape the 1D vector data into 2D grids
    # We shape them to (res_y, res_x) to match Xv and Yv
    U = x_1d.reshape(res_y, res_x)
    V = y_1d.reshape(res_y, res_x)

    # color

    color = np.sqrt(U**2 + V**2)

    # 6. Plot
    plt.figure(figsize=(10, 8))
    plt.quiver(Xv, Yv, U, V, angles='xy', scale_units='xy', scale=None)  # 'angles' and 'scale' help
    plt.title("Vector Field Visualization")
    plt.xlabel("World X Coordinate")
    plt.ylabel("World Y Coordinate")
    plt.axis('equal')  # Important for seeing vector directions correctly
    plt.show()


def visualize_cluster(cluster: Cluster, grid: Grid):
    pass