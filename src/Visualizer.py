import copy
from math import inf

import matplotlib.pyplot as plt
import numpy as np


def load_curves(filename: str) -> tuple[list[tuple[list[float], list[float], list[float]]], dict[str, float]]:
    bounding_box: dict[str, float] = {
        "x_min": +inf, "x_max": -inf,
        "y_min": +inf, "y_max": -inf,
        "t_min": +inf, "t_max": -inf,
    }

    with (open(filename, "r") as file):  # (create temp file to store real indices)
        # read bounding box
        header: list[str] = file.readline().split()
        if len(header) < 6:
            raise ValueError("Invalid bounding box line in input file")

        bounding_box["x_min"], bounding_box["x_max"], bounding_box["y_min"], bounding_box["y_max"], bounding_box[
            "t_min"], bounding_box["t_max"] = map(float, header)

        curve: tuple[list[float], list[float], list[float]] = ([], [], [])  # curve = x_axis, y_axis, t_axis
        curves: list[tuple[list[float], list[float], list[float]]] = []

        real_index: int = 0

        for line in file:
            tokens = [float(i) for i in line.strip().split()]
            if len(tokens) < 3:  # missing data (coordinate or timestamp)
                continue

            x, y, t = map(float, tokens)

            if x == y == t == 0:  # end of curve (explicit: flag)
                if len(curve[0]) >= 2:
                    curves.append(copy.deepcopy(curve))
                real_index += 1
                for ax in curve:
                    ax.clear()

            elif (  # end of curve (implicit: Out of bounding box)
                    x < bounding_box["x_min"] or x > bounding_box["x_max"] or
                    y < bounding_box["y_min"] or y > bounding_box["y_max"] or
                    t < bounding_box["t_min"] or t > bounding_box["t_max"]
            ):
                if len(curve[0]) >= 2:
                    curves.append(curve)
                # real_index is NOT incremented here
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

    return curves, bounding_box


def load_cluster(filename: str) -> list[int]:
    cluster = []
    with open(filename, 'r') as f:
        for line in f:
            cluster.append(int(line.split()[0]))

    return cluster


def load_vector_field(filename: str) -> tuple[list[float], list[float]]:
    vector_field = ([], [])

    with open(filename, 'r') as f:
        f.readline()  # discard size value at begging of file
        for line in f:
            x, y = line.split()
            vector_field[0].append(float(x))
            vector_field[1].append(float(y))

    return vector_field


def resample_vf(vector_field, new_vector_field_resolution: tuple[int, int]):
    """Resample using linear interpolation on triangular grid"""

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
        upper = ~lower          # upper-right triangle

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

    return [U_new.flatten(), V_new.flatten()]


def visualize_solution(k: int, dataset: str, vf_resolution: tuple[int, int]):  # todo: remove need of parameter k
    # LOAD CURVES COORDINATES

    curves, bounding_box = load_curves(dataset)

    # LOAD CLUSTER CURVES INDICES

    clusters = []
    for i in range(k):
        clusters.append(load_cluster(f"../output/curves_r_{i}.txt"))

    # LOAD VECTOR FIELDS

    vector_fields = []
    for i in range(k):
        vector_fields.append(load_vector_field(f"../output/vf_r_{i}.txt"))

    # PLOT

    # setup subplots

    fig, axes = plt.subplots(nrows=k+1, ncols=3, constrained_layout=True)

    for line in axes:
        for ax in line:
            ax.set_aspect('equal')
            ax.set_xlim(bounding_box['x_min'], bounding_box['x_max'])
            ax.set_ylim(bounding_box['y_min'], bounding_box['y_max'])

    # plot curves

    for i, curve in enumerate(curves):

        for j, cluster in enumerate(clusters):
            axes[0][0].plot(curve[0], curve[1])
            if i in cluster:
                axes[j + 1][0].plot(curve[0], curve[1])
                axes[j + 1][1].plot(curve[0], curve[1])
                break

    # plot vector fields

    for i, vector_field in enumerate(vector_fields):
        X = np.linspace(bounding_box['x_min'], bounding_box['x_max'], vf_resolution[0])
        Y = np.linspace(bounding_box['y_min'], bounding_box['y_max'], vf_resolution[1])
        X, Y = np.meshgrid(X, Y)

        resampled_vector_field = resample_vf(vector_field, vf_resolution)

        axes[i + 1][2].quiver(X, Y, resampled_vector_field[0], resampled_vector_field[1])

        axes[i + 1][1].quiver(X, Y, resampled_vector_field[0], resampled_vector_field[1], zorder=2)

    plt.show()


visualize_solution(k=2, dataset='../data/synthetic.txt', vf_resolution=(10, 10))  # todo: write k and dataset values in output/file
