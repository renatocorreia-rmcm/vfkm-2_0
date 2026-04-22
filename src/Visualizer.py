from pathlib import Path

import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter


# todo fix quiver ploter interpolation resolution

# todo: histogramas e graficos sobre velocidade e comprimento das curvas em cada cluster

# todo: track static trajectories. where did they went ?

# todo: colormesh plots (background of something ?)


##########################################################


class VizVectorField:
    x_axis: npt.NDArray[float]
    y_axis: npt.NDArray[float]

    def __init__(self, x_axis: npt.NDArray[float], y_axis: npt.NDArray[float]):
        self.x_axis = x_axis
        self.y_axis = y_axis

    def resample(self, resolution: (int, int)) -> (npt.NDArray[float], npt.NDArray[float]):
        """
        resample vector field axis to given resolution using linear interpolation
        """

        U_flat, V_flat = self.x_axis, self.y_axis

        new_w, new_h = resolution
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

        return U_new.flatten(), V_new.flatten()


class VizCurve:
    index: int

    x_axis: npt.NDArray[float]
    y_axis: npt.NDArray[float]
    t_axis: npt.NDArray[float]

    geometric_length: float
    speed: float

    error: float

    def __init__(self, index: int, x_axis: npt.NDArray[float], y_axis: npt.NDArray[float], t_axis: npt.NDArray[float],
                 geometric_length: float):
        self.index = index

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.t_axis = t_axis

        self.geometric_length = geometric_length

        # error is assigned in cluster file read


##########################################################

class VizCluster:
    # todo: assign lenght and speed bounds
    #   maybe on statistic calculations, considering it will need to iterate over all cluster curves anyway to get its metrics

    vector_field: VizVectorField
    curves: npt.NDArray[VizCurve]

    error_bounds: (float, float)

    # lengths_bounds: (float, float)
    # speeds_bounds: (float, float)

    def __init__(
            self,
            vector_field: VizVectorField,
            curves: npt.NDArray[VizCurve],
            error_bounds: (float, float),
            # lengths_bounds: (float, float),
            # speeds_bounds: (float, float)
    ):
        self.vector_field = vector_field
        self.curves = curves
        self.error_bounds = error_bounds
        # self.lengths_bounds = lengths_bounds
        # self.speeds_bounds = speeds_bounds


##########################################################


class Visualizer:
    # txt parameters

    output_directory: str
    dataset_path: str
    experiment_name: str
    dataset_path: str
    dataset_name: str

    # VFKM objects

    curves: npt.NDArray[VizCurve]

    clusters: npt.NDArray[VizCluster]

    dataset_lengths_bounds: (float, float)
    dataset_speeds_bounds: (float, float)
    dataset_errors_bounds: (float, float)  # todo: assign: just need to bound over clusters bounds

    # VFKM parameters

    grid_resolution: (int, int)
    k: int
    smoothness_weight: float

    bounding_box: {str, float}

    # INIT (LOADERS)

    def __init__(self, output_directory: str, dataset_path: str, experiment_name: str):
        """
        load all data from txt files in experiment directory
        """
        # VISUALIZER

        self.output_directory = output_directory
        self.experiment_name = experiment_name

        self.dataset_path = dataset_path
        self.dataset_name: str = dataset_path.split('/')[-1][:-4]

        self.experiment_directory = output_directory + self.dataset_name + f'/{experiment_name}/'

        with open(self.experiment_directory + 'arguments.txt', 'r') as file:

            line = file.readline().split(': ')[-1].split()
            self.grid_resolution = (int(line[0]), int(line[1]))

            line = file.readline().split(': ')[-1]
            self.k = int(line)

            line = file.readline().split(': ')[-1]
            self.smoothness_weight = float(line)

        # VECTOR FIELDS

        def load_vector_field(filename: str) -> VizVectorField:
            vector_field = ([], [])

            with open(filename, 'r') as f:
                f.readline()  # discard size value at begging of file
                for line in f:
                    x, y = line.split()
                    vector_field[0].append(float(x))
                    vector_field[1].append(float(y))

            return VizVectorField(np.array(vector_field[0]), np.array(vector_field[1]))

        # CURVES

        def load_all_curves() -> (npt.NDArray[VizCurve], {str, float}):
            """
            return array of curves and bounding box
            """

            inf = float('inf')
            bounding_box: {str, float} = {
                "x_min": +inf, "x_max": -inf,
                "y_min": +inf, "y_max": -inf,
                "t_min": +inf, "t_max": -inf,
            }

            min_curve_length = float('inf')
            max_curve_length = float('-inf')
            min_curve_speed = float('inf')
            max_curve_speed = float('-inf')

            with (open(self.dataset_path, "r") as file):
                # read bounding box
                header: [str] = file.readline().split()
                if len(header) < 6:
                    raise ValueError("Invalid bounding box line in input file")

                bounding_box["x_min"], bounding_box["x_max"], bounding_box["y_min"], bounding_box["y_max"], \
                    bounding_box["t_min"], bounding_box["t_max"] = map(float, header)

                curve = [[], [], [], 0]  # x_axis, y_axis, t_axis, geometric_length
                viz_curves: [VizCurve] = []

                for i, line in enumerate(file):

                    tokens = [float(i) for i in line.strip().split()]
                    if len(tokens) < 3:  # missing data (coordinate or timestamp)
                        continue
                    x, y, t = tokens

                    if (  # END OF CURVE
                            # (implicit: Out of bounding box)
                            x < bounding_box["x_min"] or x > bounding_box["x_max"] or
                            y < bounding_box["y_min"] or y > bounding_box["y_max"] or
                            t < bounding_box["t_min"] or t > bounding_box["t_max"]
                            or  # (explicit: flag)
                            x == y == t == 0
                    ):
                        if len(curve[0]) >= 2:  # valid curve - store
                            viz_curve: VizCurve = VizCurve(
                                x_axis=np.array(curve[0]),
                                y_axis=np.array(curve[1]),
                                t_axis=np.array(curve[2]),
                                index=i,
                                geometric_length=curve[-1]
                            )
                            viz_curves.append(viz_curve)

                            # update length bounds
                            if curve[-1] < min_curve_length:
                                min_curve_length = curve[-1]
                            if curve[-1] > max_curve_length:
                                max_curve_length = curve[-1]

                            # updte speed bounds
                            curve_speed = curve[-1] / curve[2][-1]  # total_lenght/total_time
                            if curve_speed < min_curve_speed:
                                min_curve_speed = curve_speed
                            if curve_speed > max_curve_speed:
                                max_curve_speed = curve_speed

                        for ax in curve[:-1]: ax.clear()  # reset coords
                        curve[-1] = 0  # reset length

                    else:  # VALID POINT

                        if not curve[0]:  # FIRST POINT in curve
                            curve[0].append(x)
                            curve[1].append(y)
                            curve[2].append(t)

                        elif t == curve[2][-1]:  # REPEATED timestamp
                            continue

                        # want to consider static trajectories from now
                        # elif (  # do not move
                        #         x == curve[0][-1] and
                        #         y == curve[1][-1]
                        # ):
                        #     continue

                        else:  # regular point
                            curve[-1] += np.hypot(
                                curve[0][-1] - x, curve[1][-1] - y
                            )
                            curve[0].append(x)
                            curve[1].append(y)
                            curve[2].append(t)

            self.dataset_lengths_bounds = min_curve_length, max_curve_length
            self.dataset_speeds_bounds = min_curve_speed, max_curve_speed
            return np.array(viz_curves, dtype='object'), bounding_box

        self.curves, self.bounding_box = load_all_curves()

        # CLUSTERS

        def load_cluster_indices(filename: str) -> ([(int, float)], (float, float)):
            """
            return cluster indices, errors and errors bounds
            """
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

        def load_all_clusters() -> npt.NDArray[VizCluster]:

            clusters: [VizCluster] = []

            for cluster_index in range(self.k):

                # vector_field
                vector_field: VizVectorField = load_vector_field(
                    self.experiment_directory + f"txt/vf_r_{cluster_index}.txt")

                # curves
                cluster_curves: [VizCurve] = []

                indices_errors, error_bounds = load_cluster_indices(
                    self.experiment_directory + f"txt/curves_r_{cluster_index}.txt")

                for index, error in indices_errors:
                    self.curves[index].error = error
                    cluster_curves.append(self.curves[index])

                # create cluster

                clusters.append(
                    VizCluster(
                        vector_field=vector_field,
                        curves=np.array(cluster_curves, dtype='object'),
                        error_bounds=error_bounds
                    )
                )

            return np.array(clusters)

        self.clusters = load_all_clusters()

    # GETTERS

    def get_plot(self, title: str = None):
        """
        get empty plot with bounding box limits and title
        """
        fig, ax = plt.subplots(constrained_layout=True)

        ax.set_aspect('equal')

        ax.set_xlim(self.bounding_box['x_min'], self.bounding_box['x_max'])
        ax.set_ylim(self.bounding_box['y_min'], self.bounding_box['y_max'])

        if title:
            plt.title(title)

        ax.set_facecolor('black')

        return fig, ax

    # SAVERS

    def save_all_fields(self, resolution: tuple[int, int]):
        """
        save all vector fields as quiver and streamplot, with shared global color normalization
        """

        X = np.linspace(self.bounding_box['x_min'], self.bounding_box['x_max'], resolution[0])
        Y = np.linspace(self.bounding_box['y_min'], self.bounding_box['y_max'], resolution[1])
        X, Y = np.meshgrid(X, Y)

        resampled_vector_fields = [c.vector_field.resample(resolution) for c in self.clusters]

        all_colors = [
            np.hypot(U, V)
            for U, V in resampled_vector_fields
        ]

        colors_min = min(np.min(c) for c in all_colors)
        colors_max = max(np.max(c) for c in all_colors)

        # normalização global compartilhada - cor de cada vetor é relativa a todos os campos vetorias, e não so o que ele pertence
        global_norm = mcolors.Normalize(vmin=colors_min, vmax=colors_max)

        cmap = cm.viridis

        for i, (U_raw, V_raw) in enumerate(resampled_vector_fields):

            # magnitude deste campo
            color_quiver = np.hypot(U_raw, V_raw)

            # =======================
            # QUIVER
            # =======================
            fig, ax = self.get_plot(title=f'vector field {i + 1} of {self.k}')

            ax.quiver(X, Y, U_raw, V_raw, color_quiver, cmap=cmap, norm=global_norm)

            sm = cm.ScalarMappable(norm=global_norm, cmap=cmap)
            sm.set_array([])

            colorbar = plt.colorbar(sm, ax=ax, label="speed")
            colorbar.set_ticks([colors_min, np.mean(color_quiver), colors_max])
            colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            plt.savefig(self.experiment_directory + f'vector_field_{i}.png', dpi=150)
            plt.close(fig)

            # =======================
            # STREAM
            # =======================
            ny, nx = Y.shape
            U = np.array(U_raw).reshape(ny, nx)
            V = np.array(V_raw).reshape(ny, nx)

            color_stream = np.hypot(U, V)

            fig, ax = self.get_plot(title=f'streamplot {i + 1} of {self.k}')

            ax.streamplot(X, Y, U, V, color=color_stream, cmap=cmap, norm=global_norm)

            sm = cm.ScalarMappable(norm=global_norm, cmap=cmap)

            colorbar = plt.colorbar(sm, ax=ax, label="speed")
            colorbar.set_ticks([colors_min, np.mean(color_stream), colors_max])
            colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            plt.savefig(self.experiment_directory + f'stream_{i}.png', dpi=200)
            plt.close(fig)

    def save_dataset(self):
        # todo: map color to length
        # todo: map color to speed
        # todo: plot in length decrescent order (avoid full overwriting of short curves by long ones)

        fig, ax = self.get_plot(title=f"{self.dataset_name}")

        # plot in index order
        for curve in self.curves:
            ax.plot(curve.x_axis, curve.y_axis)

        """
        # plot in cluster order
        for cluster in self.clusters_curves:
            for curve in cluster:
                ax.plot(curve[0], curve[1])
        """

        plt.savefig(self.output_directory + self.dataset_name + '/dataset.png', dpi=300)
        plt.close(fig)

    def save_clusters_curves(self, vf_resolution: tuple[int, int] = None):

        if vf_resolution is None:
            # no interpolation, raw vertices vectors
            vf_resolution = self.grid_resolution

        meshgrid: tuple = ()
        if vf_resolution:
            X = np.linspace(self.bounding_box['x_min'], self.bounding_box['x_max'], vf_resolution[0])
            Y = np.linspace(self.bounding_box['y_min'], self.bounding_box['y_max'], vf_resolution[1])
            meshgrid = np.meshgrid(X, Y)

        for i, cluster in enumerate(self.clusters):

            fig, ax = self.get_plot(f'curves {i + 1} of {self.k}')

            # error color
            # error_bounding = self.clusters_errors_bounds[i]
            #
            # norm = mcolors.Normalize(vmin=error_bounding[0], vmax=error_bounding[1])
            # cmap = cm.viridis
            #
            # # colorbar
            # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            # colorbar = plt.colorbar(sm, ax=ax, label="Curve error")
            #
            # for c in cluster:
            #     curve = c[0]
            #     curve_error: float = c[1]
            #
            #     ax.plot(curve[0], curve[1], color=cmap(norm(curve_error)))

            # length color
            # norm = mcolors.Normalize(vmin=self.lengths_bounds[0], vmax=self.lengths_bounds[1])
            # cmap = cm.viridis
            # for c in cluster:
            #     curve = c[0]
            #     curve_length = curve[-1]
            #
            #     ax.plot(curve[0], curve[1], color=cmap(norm(curve_length)))
            # # colorbar
            # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            # colorbar = plt.colorbar(sm, ax=ax, label="Curve length")

            # speed color

            norm = mcolors.Normalize(vmin=self.dataset_speeds_bounds[0], vmax=self.dataset_speeds_bounds[1])
            cmap = cm.viridis

            for curve in cluster.curves:
                # todo: take statics HERE
                curve_speed = curve.geometric_length/curve.t_axis[-1]

                ax.plot(curve.x_axis, curve.y_axis, color=cmap(norm(curve_speed)))

            # colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            plt.colorbar(sm, ax=ax, label="average speed")

            plt.savefig(self.experiment_directory + f'curves_{i}.png', dpi=300)

            # OVERLAY VECTOR FIELD

            U, V = cluster.vector_field.resample(vf_resolution)

            ax.quiver(meshgrid[0], meshgrid[1], U, V, color='w', zorder=2)
            plt.title(f'cluster {i + 1} of {self.k}')

            plt.savefig(self.experiment_directory + f'cluster_{i}.png', dpi=300)

            plt.close(fig)

    def save_all(self, vf_resolution: tuple[int, int] = None):

        if not Path(self.output_directory + self.dataset_name + '/dataset.png').exists():
            print('saving dataset')
            self.save_dataset()

        self.save_all_fields(vf_resolution)
        self.save_clusters_curves(vf_resolution)


if __name__ == '__main__':
    v = Visualizer(
        output_directory='../output/',
        dataset_path='../data/sperm_xy_rotated_centered.txt',
        experiment_name='Experiment_3x3_3_0.0150'
    )
    v.save_all((10, 7))
