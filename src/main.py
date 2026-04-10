import shutil
from pathlib import Path

import numpy as np

from Cluster import Cluster
from Grid import Grid
from Point2D import Point2D
from PolygonalPath2D import PolygonalPath2D

import sys
from math import inf

from PolygonalPath2D import PolygonalPath2D as PolygonalPath
from VFKM import VFKM
from src.Visualizer import Visualizer


def load_curves(filename: str) -> tuple[list[PolygonalPath], dict[str, float]]:
    """
	params:
		filename: str - path to input file
	returns:
		polygonalpaths
		bounding box: dict with keys x_min, x_max, y_min, y_max,
	"""

    paths: list[PolygonalPath] = []

    bounding_box: dict[str, float] = {
        "x_min": +inf, "y_min": +inf, "t_min": +inf,
        "x_max": -inf, "y_max": -inf, "t_max": -inf
    }

    with (open(filename, "r") as file):  # (create temp file to store real indices)
        # read bounding box
        header: list[str] = file.readline().split()
        if len(header) < 6:
            raise ValueError("Invalid bounding box line in input file")

        bounding_box["x_min"], bounding_box["x_max"], bounding_box["y_min"], bounding_box["y_max"], bounding_box[
            "t_min"], bounding_box["t_max"] = map(float, header)

        curve_contents: list[Point2D] = []
        real_index: int = 0

        for line in file:
            tokens = [float(i) for i in line.strip().split()]
            if len(tokens) < 3:  # missing data (coordinate or timestamp)
                continue

            x, y, t = map(float, tokens)

            if x == y == t == 0:  # end of curve (explicit - flag)
                if len(curve_contents) >= 2:
                    paths.append(PolygonalPath(curve_contents))
                real_index += 1
                curve_contents.clear()

            elif (  # end of curve (implicit - Out of bounding box)
                    x < bounding_box["x_min"] or x > bounding_box["x_max"] or
                    y < bounding_box["y_min"] or y > bounding_box["y_max"] or
                    t < bounding_box["t_min"] or t > bounding_box["t_max"]
            ):
                if len(curve_contents) >= 2:
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

    return paths, bounding_box


import os


def save_experiment(
        experiment_name: str,
        k: int, grid_resolution: tuple[int, int], smoothness_weight: float,
        output_directory: str, current_file_loaded: str, root_cluster: Cluster
):
    """
    note: an experiment is the running of a dataset with a specif set of arguments

    save files in experiment_directory = output_directory/<current_file_loaded_name>/<experiment_name>/
    this allows to keep multiple experiments results at the same time
    """

    experiment_directory = output_directory + current_file_loaded.split('/')[-1][:-4] + f'/{experiment_name}/'

    # delete experiment_path directory
    if Path(experiment_directory).exists():
        print(f"overwriting (recreating) already existent {experiment_directory}")
        shutil.rmtree(Path(experiment_directory))
    else:
        print(f"Saving experiment at {experiment_directory}")

    # (re)create experiment_path directory
    Path(experiment_directory+'txt/').mkdir(parents=True, exist_ok=True)

    with open(experiment_directory+'arguments.txt', 'w') as visualizer_file:
        # assume there's no need to store output_directory and filename(dataset) in here,
        # because to found this file you need this info already
        visualizer_file.write(f'grid_resolution: {grid_resolution[0]} {grid_resolution[1]}\n')
        visualizer_file.write(f'amount_of_clusters: {k}\n')
        visualizer_file.write(f'smoothness_weight: {smoothness_weight}\n')

    # Create experiment file
    experiment_path = experiment_directory + 'txt/experiment.txt'
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
            curve_filename = os.path.join(experiment_directory, f"txt/curves_{cluster_name}.txt")
            with open(curve_filename, "w") as curve_indices_file:
                number_of_curves = len(c.curves)
                assert number_of_curves == len(c.curve_errors)

                for i in range(len(c.curves)):
                    curve_indices_file.write(f"{c.curves[i].index} {c.curve_errors[i]}\n")

            # --- Write vector field file ---
            vector_field_filename = os.path.join(experiment_directory, f"txt/vf_{cluster_name}.txt")
            with open(vector_field_filename, "w") as vector_field_file:
                x_component = c.vector_field[0]
                y_component = c.vector_field[1]
                grid_dimension = x_component.shape[0]

                vector_field_file.write(f"{grid_dimension}\n")

                for i in range(grid_dimension):
                    vector_field_file.write(f"{x_component[i]} {y_component[i]}\n")

            # --- Process children ---
            for i, child in enumerate(c.children):
                child_name = f"{cluster_name}_{i}"
                map_cluster_path[child] = child_name

                experiment_file.write(f"{cluster_name} {child_name}\n")
                nodes_to_process.append(child)  # include file with data for initialize


def init_experiment(
        filename: str,
        grid_resolution: int
) -> tuple[list[PolygonalPath2D], Grid, Cluster]:
    """
	Initialize paths, grid , and rootcluster
	"""

    paths, bounding_box = load_curves(filename)  # list[PolygonalPath2D], dict[str, float]

    # Initialize grid (square)
    grid: Grid = Grid(
        bounding_box=bounding_box,
        resolution=grid_resolution
    )

    # Initialize root cluster
    root_cluster = Cluster(
        name=str(len(paths)),
        grid=grid
    )

    root_cluster.curves = []  # set it here if needed
    root_cluster.curve_errors = []

    return paths, grid, root_cluster


# todo: implement hierarquical clustering
def main(
        filename: str,
        grid_resolution: int,
        number_of_vector_fields: int,
        smoothness_weight: float,
        output_directory: str
):
    """
	arguments:
		trajectoryFile gridResolution numberOfVectorFields smoothnessWeight outputDirectory
	"""

    # CHECK IF EXPERIMENT ALREADY EXISTS
    experiment_name = f"Experiment_{grid_resolution}x{grid_resolution}_{number_of_vector_fields}_{smoothness_weight:.4f}"
    experiment_directory = output_directory + filename.split('/')[-1][:-4] + f'/{experiment_name}/'

    # delete experiment_path directory
    if Path(experiment_directory).exists():
        print(f"\nFOUND EXPERIMENT {experiment_directory} ALREADY\n SKKIPING TO NEXT ONE\n")
        return


    """
    # check arguments
    right_number_of_parameters = 6
    if len(sys.argv) != right_number_of_parameters:  # aslo check type and file existence
        print("./main.py trajectoryFile gridResolution numberOfVectorFields smoothnessWeight outputDirectory")
        return

    # load arguments
    filename = sys.argv[1]
    grid_resolution = int(sys.argv[2])
    number_of_vector_fields = int(sys.argv[3])
    smoothness_weight = float(sys.argv[4])
    output_directory = sys.argv[5]
    """

    # initialize parameters
    paths: list[PolygonalPath]
    root_cluster: Cluster  # until now, is not being accessed, just updated
    grid: Grid

    paths, grid, root_cluster = init_experiment(
        filename=filename,
        grid_resolution=grid_resolution
    )

    # OPTIMIZE

    # initialize current cluster

    # optimize
    clusters: list[Cluster] = VFKM.optimize_implicit_fast_with_weights(
        grid=grid,
        paths=paths,
        number_of_vector_fields=number_of_vector_fields,
        smoothness_weight=smoothness_weight
    )

    root_cluster.children = clusters

    save_experiment(
        experiment_name=experiment_name,

        grid_resolution=(grid.get_resolution_x(), grid.get_resolution_y()),
        k=number_of_vector_fields,
        smoothness_weight=smoothness_weight,

        output_directory=output_directory,
        current_file_loaded=filename,
        root_cluster=root_cluster  # first cluster is root
    )

    #print("Loading Visualizer...")
    v = Visualizer(output_directory, filename, experiment_name)
    v.save_all((10, 10))  # todo: softcode this


""" debug arguments: ../data/synthetic.txt 3 2 0.05 ../output/

os endereços usados na modularização desse código 
foram escritos para IDEs, como PyCharm, onde cada arquivo acessa o endereço importado a partir da root do projeto,
e não editores de texto, como VScode, que acessam o endereço a partir do arquivo atual.

rodar no VScode exige reescrever as importações em cada arquivo
"""

# todo: IMPLEMENT AND APPLY HIERARQUICAL CLUSTERING
# todo: test: trajectories_CENTERED: k=12: [slow, fast] x [foward, backward] x [horizontal, vertical, circular]
#       test: trajectories_ROTADED_CENTERED: k=4: [slow, fast] x [horizontal, circular]
#       desenvolver esse raciocinio pra preparar apresentação
#       relacionar com cluster hierarquico
#           k = a*b pode ser alcaçado com k=a*b ou k=a de depois herda com k=b

from multiprocessing import Pool, cpu_count


def run_experiment(args):
    exp_id, number_of_experiments, resolution, k, smoothness_weight = args

    print('\n' + '#' * 100)
    print(f'STARTING EXPERIMENT {exp_id} of {number_of_experiments}:')
    print(f'resolution = {resolution}x{resolution}')
    print(f'k = {k}')
    print(f'smoothness_weight = {smoothness_weight:.4f}')

    main(filename='../data/sperm_xy_rotated_centered.txt',
         grid_resolution=resolution,
         number_of_vector_fields=k,
         smoothness_weight=smoothness_weight,
         output_directory='../output/')

    main(filename='../data/sperm_xy_centered.txt',
         grid_resolution=resolution,
         number_of_vector_fields=k,
         smoothness_weight=smoothness_weight,
         output_directory='../output/')

    main(filename='../data/sperm_xy.txt',
         grid_resolution=resolution,
         number_of_vector_fields=k,
         smoothness_weight=smoothness_weight,
         output_directory='../output/')

    main(filename='../data/sperm_xy_translate_modified.txt',
         grid_resolution=resolution,
         number_of_vector_fields=k,
         smoothness_weight=smoothness_weight,
         output_directory='../output/')

    main(filename='../data/sperm_xy_rotated.txt',
         grid_resolution=resolution,
         number_of_vector_fields=k,
         smoothness_weight=smoothness_weight,
         output_directory='../output/')


if __name__ == "__main__":

    # SEARCHING RESULTS IN PARAMETER SPACE
    resolution_space = [3, 4, 5, 6, 7, 8, 9, 10]
    k_space = [2, 3, 4, 5, 6, 7]
    smoothness_weight_space = [0.0001, 0.0005, 0.0015, 0.0050, 0.0100, 0.0250, 0.0400, 0.0700, 0.1000]

    parameter_space = [
        (r, k, s)
        for r in resolution_space
        for k in k_space
        for s in smoothness_weight_space
    ]

    total = len(parameter_space)

    # attach experiment IDs
    tasks = [
        (i + 1, total, r, k, s)
        for i, (r, k, s) in enumerate(parameter_space)
    ]

    # use all cores
    with Pool(cpu_count()) as p:
        p.map(run_experiment, tasks)
