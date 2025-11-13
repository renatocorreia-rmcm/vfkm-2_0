import numpy as np

from src.Cluster import Cluster
from src.Grid import Grid
from src.Point2D import Point2D
from src.PolygonalPath2D import PolygonalPath2D


import sys
from math import inf

from src.PolygonalPath2D import PolygonalPath2D as PolygonalPath
from src.VFKM import VFKM
from src.VectorField2D import VectorField2D


def load_curves(filename: str) -> tuple[list[PolygonalPath], dict[str, float]]:
    """
    params:
        filename: str - path to input file
    returns:
        number of curves read: int
        bounding box: dict with keys x_min, x_max, y_min, y_max,
    """

    print("loading curves")

    paths: list[PolygonalPath] = []

    bounding_box: dict[str, float] = {
        "x_min": +inf, "y_min": +inf, "t_min": +inf,
        "x_max": -inf, "y_max": -inf, "t_max": -inf
    }

    try:
        with (open(filename, "r") as file, open("/tmp/real_indices.txt", "w") as real_indices):  # (create temp file to store real indices)
            # read bounding box
            header: list[str] = file.readline().split()
            if len(header) < 6:
                raise ValueError("Invalid bounding box line in input file")

            bounding_box["x_min"], bounding_box["x_max"], bounding_box["y_min"], bounding_box["y_max"], bounding_box["t_min"], bounding_box["t_max"] = map(float, header)

            curve_contents: list[Point2D] = []
            real_index: int = 0

            for line in file:
                tokens = [float(i) for i in line.strip().split()]
                if len(tokens) < 3:  # missing data (coordinate or timestamp)
                    continue

                x, y, t = map(float, tokens)

                if x == y == t == 0:  # end of curve (explicit - flag)
                    if len(curve_contents) >= 2:
                        real_indices.write(f"{real_index}\n")
                        paths.append(PolygonalPath(curve_contents))
                    real_index += 1
                    curve_contents.clear()

                elif( # end of curve (implicit - Out of bounding box)
                        x < bounding_box["x_min"] or x > bounding_box["x_max"] or
                        y < bounding_box["y_min"] or y > bounding_box["y_max"] or
                        t < bounding_box["t_min"] or t > bounding_box["t_max"]
                ):
                    if len(curve_contents) >= 2:
                        real_indices.write(f"{real_index}\n")
                        paths.append(PolygonalPath(curve_contents))
                    # real_index is NOT incremented here
                    curve_contents.clear()

                else:  # valid point
                    new_point = Point2D(point2d=(np.array([x, y]), t))

                    if not curve_contents:  # first point in curve
                        curve_contents.append(new_point)
                    elif t == curve_contents[-1].time:  # repeated timestamp
                        continue
                    elif (  # do not move
                        x == curve_contents[-1].space[0]
                        and y == curve_contents[-1].space[1]
                    ):
                        continue
                    else:  # regular point
                        curve_contents.append(new_point)

    except FileNotFoundError:
        print(f"Unable to open file {filename}", file=sys.stderr)
        return paths, bounding_box

    # Output read data to file
    with open("read_curves.txt", "w") as outfile:
        outfile.write(
            f"{bounding_box['x_min']} {bounding_box['x_max']} "
            f"{bounding_box['y_min']} {bounding_box['y_max']} "
            f"{bounding_box['t_min']} {bounding_box['t_max']}\n"
        )

        for path in paths:
            for point in path.points:
                outfile.write(f"{point.space} {point.time}\n")
            outfile.write("0 0 0\n")

    # Optional debug section
    DEBUG = True
    if DEBUG:
        print(f"numberOfPathsRead = {len(paths)}")
        for i, path in enumerate(paths):
            print(f"Path {i} \n {path}")

    return paths, bounding_box


import os


def save_experiment(directory: str, current_file_loaded: str, root_cluster: Cluster):

    # Create experiment file
    experiment_path = os.path.join(directory, "experiment.txt")
    with open(experiment_path, "w") as experiment_file:
        experiment_file.write(current_file_loaded + "\n")

        # Initialize queue using a list
        nodes_to_process = [root_cluster]

        # Mapping from cluster to its string path
        map_cluster_path = {root_cluster: "r"}

        experiment_file.write("-1 r\n")

        while nodes_to_process:
            # Take next cluster (FIFO)
            c = nodes_to_process[0]
            nodes_to_process = nodes_to_process[1:]

            cluster_name = map_cluster_path[c]

            # --- Write curve indices file ---
            curve_filename = os.path.join(directory, f"curves_{cluster_name}.txt")
            with open(curve_filename, "w") as curve_indices_file:
                number_of_curves = len(c.indices)
                assert number_of_curves == len(c.curveErrors)

                for idx, err in zip(c.indices, c.curveErrors):
                    curve_indices_file.write(f"{idx} {err}\n")

            # --- Write vector field file ---
            vector_field_filename = os.path.join(directory, f"vf_{cluster_name}.txt")
            with open(vector_field_filename, "w") as vector_field_file:
                x_component = c.vectorField[0]
                y_component = c.vectorField[1]
                grid_dimension = x_component.getDimension()

                vector_field_file.write(f"{grid_dimension}\n")

                for i in range(grid_dimension):
                    vector_field_file.write(f"{x_component[0][i]} {y_component[0][i]}\n")

            # --- Process children ---
            for i, child in enumerate(c.children):
                child_name = f"{cluster_name}_{i}"
                map_cluster_path[child] = child_name

                experiment_file.write(f"{cluster_name} {child_name}\n")
                nodes_to_process.append(child)




def init_experiment(
        filename: str,
        grid_resolution: int
) -> tuple[list[PolygonalPath2D], Grid, Cluster]:
    """
    Initialize paths, grid , and rootcluster
    """

    bounding_box: dict[str, float]
    paths: list[PolygonalPath2D]
    paths, bounding_box = load_curves(filename)

    # Initialize grid (square)
    g: Grid = Grid(
        bounding_box=bounding_box,
        resolution=grid_resolution
    )

    # Initialize root cluster
    root_cluster = Cluster(
        name = str(len(paths)),
        vector_field=VectorField2D([
        np.zeros(shape=g.resolution_x*g.resolution_y),
        np.zeros(shape=g.resolution_x * g.resolution_y)
        ])
    )

    #todo: solve: original implementation load errors as 0 and curves indices as all curves

    return paths, g, root_cluster


def main():
    """
    arguments:
        trajectoryFile gridResolution numberOfVectorFields smoothnessWeight outputDirectory
    """

    # check arguments
    right_number_of_parameters = 6
    if len(sys.argv) != right_number_of_parameters:
        print("./vfkm trajectoryFile gridResolution numberOfVectorFields smoothnessWeight outputDirectory")
        return

    # load arguments
    filename = sys.argv[1]
    grid_resolution = int(sys.argv[2])
    number_of_vector_fields = int(sys.argv[3])
    smoothness_weight = float(sys.argv[4])
    output_directory = sys.argv[5]

    # initialize parameters
    paths: list[PolygonalPath]
    root_cluster: Cluster
    grid: Grid

    paths, grid, root_cluster = init_experiment(
        filename=filename,
        grid_resolution=grid_resolution
    )

    # OPTIMIZE
    print("Optimizing...")
    # initialize current cluster
    current_cluster = root_cluster

    # optimize
    clusters: list[Cluster] = VFKM.optimize_implicit_fast_with_weights(
        grid=grid,
        paths=paths,
        number_of_vector_fields=number_of_vector_fields,
        smoothness_weight=smoothness_weight
    )


if __name__ == "__main__":
    main()
