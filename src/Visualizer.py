import copy
from math import inf

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

# basic types

vector_field_type = tuple[list[float], list[float]]  # 2uple of axis
curve_type = tuple[list[float], list[float], list[float]]  # 3ple of x, y, t lists


# todo: estatísticas sobre as velocidades de cada cluster

# todo: include colormesh plots (background of something ?)

# todo: create own dataset variation: translate trajectories my its average position. so on clustering, we get the best of its rotation tendencies
# todo: may test some propposital alisaing on trajectories, reading then with a stepfactor traj = traj[::s]
# may induce some average trajectory


class Visualizer:
    output_directory: str
    current_file_loaded: str
    dataset_path: str
    dataset_name: str
    experiment_name: str

    grid_resolution: tuple[int, int]
    k: int
    smoothness_weight: float

    curves: npt.NDArray[curve_type]  # array of all curves in dataset

    vector_fields: list[vector_field_type]

    clusters_indices: list[list[tuple[int, float]]]  # used only to load self.cluster_curves
    clusters_curves: list[list[tuple[curve_type, float]]]
    clusters_errors_bounds: list[tuple[float, float]]  # min and max error for each cluster

    bounding_box: dict[str, float]

    # CONSTRUCTOR

    def __init__(self, output_directory: str, current_file_loaded: str, experiment_name: str):

        self.output_directory = output_directory
        self.current_file_loaded = current_file_loaded
        self.experiment_name = experiment_name

        self.dataset_path = current_file_loaded

        self.dataset_name: str = current_file_loaded.split('/')[-1][:-4]

        self.experiment_directory = output_directory + current_file_loaded.split('/')[-1][:-4] + f'/{experiment_name}/'
        print(f"Saving images at {self.experiment_directory}")

        with open(self.experiment_directory + 'arguments.txt', 'r') as file:
            line = file.readline().split(': ')[-1].split()
            self.grid_resolution = (int(line[0]), int(line[1]))

            self.k = int(file.readline().split(': ')[-1])

            self.smoothness_weight = float(file.readline().split(': ')[-1])

        # load cluster vector fields

        def load_vector_field(filename: str) -> vector_field_type:
            vector_field = ([], [])

            with open(filename, 'r') as f:
                f.readline()  # discard size value at begging of file
                for line in f:
                    x, y = line.split()
                    vector_field[0].append(float(x))
                    vector_field[1].append(float(y))

            return vector_field

        def load_all_vector_fields() -> list[vector_field_type]:
            vector_fields = []
            for i in range(self.k):
                vector_fields.append(load_vector_field(self.experiment_directory + f"txt/vf_r_{i}.txt"))
            return vector_fields

        self.vector_fields = load_all_vector_fields()

        # load clusters indices

        def load_cluster_indices(filename: str) -> tuple[list[tuple[int, float]], tuple[float, float]]:
            cluster = []

            min_error = float('inf')
            max_error = float('-inf')

            with open(filename, 'r') as f:
                for line in f:
                    line = line.split()
                    error = float(line[1])
                    if error < min_error: min_error = error
                    if error > max_error: max_error = error

                    cluster.append((int(line[0]), error))

            return cluster, (min_error, max_error)

        def load_all_clusters_indices() -> tuple[list[list[tuple[int, float]]], list[tuple[float, float]]]:
            clusters = []
            error_bounds = []

            for i in range(self.k):
                cluster, error_bound = load_cluster_indices(self.experiment_directory + f"txt/curves_r_{i}.txt")

                clusters.append(cluster)
                error_bounds.append(error_bound)

            return clusters, error_bounds

        self.clusters_indices, self.clusters_errors_bounds = load_all_clusters_indices()

        # load all curves

        def load_curves(filename: str) -> tuple[npt.NDArray[curve_type], dict[str, float]]:
            bounding_box: dict[str, float] = {
                "x_min": +inf, "x_max": -inf,
                "y_min": +inf, "y_max": -inf,
                "t_min": +inf, "t_max": -inf,
            }

            with (open(filename, "r") as file):
                # read bounding box
                header: list[str] = file.readline().split()
                if len(header) < 6:
                    raise ValueError("Invalid bounding box line in input file")

                bounding_box["x_min"], bounding_box["x_max"], bounding_box["y_min"], bounding_box["y_max"], \
                bounding_box["t_min"], bounding_box["t_max"] = map(float, header)

                curve: curve_type = ([], [], [])  # curve = x_axis, y_axis, t_axis
                curves: list[curve_type] = []

                for line in file:
                    tokens = [float(i) for i in line.strip().split()]
                    if len(tokens) < 3:  # missing data (coordinate or timestamp)
                        continue

                    x, y, t = map(float, tokens)

                    if x == y == t == 0:  # end of curve (explicit: flag)
                        if len(curve[0]) >= 2:
                            curves.append(copy.deepcopy(curve))
                        for ax in curve:
                            ax.clear()

                    elif (  # end of curve (implicit: Out of bounding box)
                            x < bounding_box["x_min"] or x > bounding_box["x_max"] or
                            y < bounding_box["y_min"] or y > bounding_box["y_max"] or
                            t < bounding_box["t_min"] or t > bounding_box["t_max"]
                    ):
                        if len(curve[0]) >= 2:
                            curves.append(curve)

                        for ax in curve: ax.clear()

                    else:  # valid point

                        if not curve[0]:  # first point in curve
                            curve[0].append(x)
                            curve[1].append(y)
                            curve[2].append(t)
                        elif t == curve[2][-1]:  # repeated timestamp
                            continue
                        elif (  # do not move
                                x == curve[0][-1] and
                                y == curve[1][-1]
                        ):
                            continue
                        else:  # regular point
                            curve[0].append(x)
                            curve[1].append(y)
                            curve[2].append(t)

            return np.array(curves, dtype='object'), bounding_box

        self.curves, self.bounding_box = load_curves(self.dataset_path)

        # load all clusters curves

        def map_clusters_curves() -> list[list[tuple[curve_type, float]]]:

            clusters_curves: list[list[tuple[curve_type, float]]] = [
                [(self.curves[curve[0]], curve[1]) for curve in cluster] for cluster in self.clusters_indices
            ]

            return clusters_curves

        self.clusters_curves = map_clusters_curves()

        # set pyplot resolution

        plt.rcParams['savefig.dpi'] = 300

    # GETTERS

    def resample_vector_fields(self, new_vector_field_resolution: tuple[int, int]):
        """Resample using linear interpolation on triangular grid"""

        resampled_vector_fields: list[vector_field_type] = []

        for vector_field in self.vector_fields:
            U_flat, V_flat = vector_field

            new_w, new_h = new_vector_field_resolution
            old_w = old_h = int(len(U_flat) ** 0.5)

            U = np.array(U_flat).reshape(old_h, old_w)
            V = np.array(V_flat).reshape(old_h, old_w)

            # new grid
            X_new = np.linspace(0, 1, new_w)
            Y_new = np.linspace(0, 1, new_h)
            X_new, Y_new = np.meshgrid(X_new, Y_new)

            # map to old index space
            X = X_new * (old_w - 1)
            Y = Y_new * (old_h - 1)

            def triangle_interp(Z, X, Y):
                x0 = np.floor(X).astype(int)
                y0 = np.floor(Y).astype(int)

                x1 = np.clip(x0 + 1, 0, Z.shape[1] - 1)
                y1 = np.clip(y0 + 1, 0, Z.shape[0] - 1)

                # local coordinates inside cell
                dx = X - x0
                dy = Y - y0

                Z_new = np.zeros_like(X)

                # mask: which triangle?
                lower = (dx + dy <= 1)  # lower-left triangle
                upper = ~lower  # upper-right triangle

                # --- lower triangle (x0,y0), (x1,y0), (x0,y1)
                Z_new[lower] = (
                        (1 - dx[lower] - dy[lower]) * Z[y0[lower], x0[lower]] +
                        dx[lower] * Z[y0[lower], x1[lower]] +
                        dy[lower] * Z[y1[lower], x0[lower]]
                )

                # --- upper triangle (x1,y1), (x1,y0), (x0,y1)
                Z_new[upper] = (
                        (dx[upper] + dy[upper] - 1) * Z[y1[upper], x1[upper]] +
                        (1 - dy[upper]) * Z[y0[upper], x1[upper]] +
                        (1 - dx[upper]) * Z[y1[upper], x0[upper]]
                )

                return Z_new

            U_new = triangle_interp(U, X, Y)
            V_new = triangle_interp(V, X, Y)

            resampled_vector_fields.append((U_new.flatten(), V_new.flatten()))

        return resampled_vector_fields

    def get_plot(self, title: str = None):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_aspect('equal')
        ax.set_xlim(self.bounding_box['x_min'], self.bounding_box['x_max'])
        ax.set_ylim(self.bounding_box['y_min'], self.bounding_box['y_max'])
        if title:
            plt.title(title)

        return fig, ax

    # SAVERS

    def save_vector_fields(self, resolution: tuple[int, int]):

        X = np.linspace(self.bounding_box['x_min'], self.bounding_box['x_max'], resolution[0])
        Y = np.linspace(self.bounding_box['y_min'], self.bounding_box['y_max'], resolution[1])
        X, Y = np.meshgrid(X, Y)

        for i, resampled_vector_field in enumerate(self.resample_vector_fields(resolution)):
            fig, ax = self.get_plot(title=f'vector field {i + 1} of {self.k}')
            ax.set_facecolor('black')

            U, V = resampled_vector_field

            color = np.hypot(U, V)

            ax.quiver(X, Y, U, V, color, cmap='viridis')

            norm = mcolors.Normalize(vmin=np.min(color), vmax=np.max(color))
            cmap = cm.viridis

            # colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = plt.colorbar(sm, ax=ax, label="speed")
            colorbar.set_ticks([np.min(color), np.average(color), np.max(color)])
            colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            plt.savefig(self.experiment_directory + f'vector_field_{i}.png', dpi=150)
            plt.close(fig)

    def save_streams(self, resolution: tuple[int, int]):

        X = np.linspace(self.bounding_box['x_min'], self.bounding_box['x_max'], resolution[0])
        Y = np.linspace(self.bounding_box['y_min'], self.bounding_box['y_max'], resolution[1])
        X, Y = np.meshgrid(X, Y)

        for i, resampled_vector_field in enumerate(self.resample_vector_fields(resolution)):
            fig, ax = self.get_plot(title=f"streamplot {i + 1} of {self.k}")
            ax.set_facecolor('black')

            U, V = resampled_vector_field

            ny, nx = Y.shape

            U = np.array(U).reshape(ny, nx)
            V = np.array(V).reshape(ny, nx)

            color = np.hypot(U, V)

            ax.streamplot(X, Y, U, V, color=color, cmap='viridis')

            norm = mcolors.Normalize(vmin=np.min(color), vmax=np.max(color))
            cmap = cm.viridis

            # colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = plt.colorbar(sm, ax=ax, label="speed")
            colorbar.set_ticks([np.min(color), np.average(color), np.max(color)])
            colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            plt.savefig(self.experiment_directory + f'stream_{i}.png', dpi=200)
            plt.close(fig)

    def save_dataset(self):

        fig, ax = self.get_plot(title=f"{self.dataset_name}")

        # plot in index order
        for curve in self.curves:
            ax.plot(curve[0], curve[1])
        """

        # plot in cluster order
        for cluster in self.clusters_curves:
            for curve in cluster:
                ax.plot(curve[0], curve[1])
        """

        plt.savefig(self.experiment_directory + 'dataset.png')
        plt.close(fig)

    def save_clusters_curves(self, vf_resolution: tuple[int, int] = None):

        resampled_vector_fields: list[vector_field_type] = []
        meshgrid: tuple = ()

        if vf_resolution:
            X = np.linspace(self.bounding_box['x_min'], self.bounding_box['x_max'], vf_resolution[0])
            Y = np.linspace(self.bounding_box['y_min'], self.bounding_box['y_max'], vf_resolution[1])
            meshgrid = np.meshgrid(X, Y)

            resampled_vector_fields = self.resample_vector_fields(vf_resolution)

        for i, cluster in enumerate(self.clusters_curves):

            fig, ax = self.get_plot(f'curves {i + 1} of {self.k}')
            ax.set_facecolor('black')

            error_bounding = self.clusters_errors_bounds[i]

            norm = mcolors.Normalize(vmin=error_bounding[0], vmax=error_bounding[1])
            cmap = cm.viridis

            # colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = plt.colorbar(sm, ax=ax, label="Curve Error")

            for curve in cluster:
                curve_cords = curve[0]
                curve_error: float = curve[1]

                ax.plot(curve_cords[0], curve_cords[1], color=cmap(norm(curve_error)))

            plt.savefig(self.experiment_directory + f'curves_{i}.png')

            if vf_resolution:
                U = resampled_vector_fields[i][0]
                V = resampled_vector_fields[i][1]
                color = np.hypot(U, V)
                ax.quiver(meshgrid[0], meshgrid[1], U, V, color='w', zorder=2)
                plt.title(f'cluster {i + 1} of {self.k}')
                plt.savefig(self.experiment_directory + f'cluster_{i}.png')
            plt.close(fig)

    # ALL

    def save_all(self, vf_resolution: tuple[int, int] = None):
        self.save_vector_fields(vf_resolution)
        self.save_dataset()
        self.save_clusters_curves(vf_resolution)
        self.save_streams(vf_resolution)


if __name__ == '__main__':
    v = Visualizer(current_file_loaded='../data/sperm_xy_rotated.txt', output_directory='../output/',
                   experiment_name='Experiment_4x4_5_0.02')
    v.save_all((10, 7))
