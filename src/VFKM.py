""""""

# todo: use cluster object


import numpy as np

from scipy.sparse.linalg import cg  # conjugate gradient solver
from scipy.sparse.linalg import LinearOperator  # class to incorporate original "multiplyByA()"

from functools import partial

from Grid import Grid, CurveDescription
from src.Cluster import Cluster
from src.PolygonalPath2D import PolygonalPath2D
from src.VectorField2D import VectorField2D


class ProblemSettings:
    """
    Grid and curves
    """

    grid: Grid
    curve_indices: list[int]  # respective to cluster
    curve_descriptions: list[CurveDescription]
    total_curve_length: float  # summed length of all curves (used for normalization)
    smoothness_weight: float

    def __init__(
            self,
            grid: Grid,
            curve_indices: list[int],
            curve_descriptions: list[CurveDescription],
            total_curve_length: float,
            smoothness_weight: float
    ):
        self.grid = grid
        self.curve_indices = curve_indices
        self.curve_descriptions = curve_descriptions
        self.total_curve_length = total_curve_length
        self.smoothness_weight = smoothness_weight


class VFKM:

    @staticmethod
    def multiply_by_A(
            vec: np.ndarray[float],
            problem: ProblemSettings
    ) -> np.ndarray[float]:
        """
        Compute A*v without setting up the matrix

        A*v=b is the derivative of error formula (a quadratic equation derivative results in a linear systems of equations)
        A is it second order derivative

        computed as  A_fit-error * x + A_smooth-error * x
        """

        grid: Grid = problem.grid
        num_vertices: int = grid.get_resolution_x() * grid.get_resolution_y()

        result_x: np.ndarray[float] = np.zeros(shape=num_vertices, dtype=float)
        smoothness_weight = problem.smoothness_weight

        # FIT PENALTY

        curve_indices = problem.curve_indices
        curve_descriptions = problem.curve_descriptions

        total_curve_length = problem.total_curve_length
        k_fit = (1.0 - smoothness_weight) / total_curve_length  # normalization factor for FIT error

        for k in range(len(curve_indices)):  # For each curve in cluster
            i = curve_indices[k]  # Get its index
            curve = curve_descriptions[i]  # Load the curve

            # Add FIT contribution of this curve to result
            curve.add_cTcx(result_x, vec, k_fit)

        # SMOOTH PENALTY

        Ax = vec.copy()  # Ax = v (initial copy of v)

        # DISCRETIZED LAPLACIAN
        grid.multiply_by_laplacian2(Ax)  # Ax = L*x
        grid.multiply_by_laplacian2(Ax)  # Ax = L^T * L * x (laplaciano discretizado é uma matriz simétrica L^T = L)

        # Normalization
        k_smooth = smoothness_weight / num_vertices  # normalization factor for SMOOTH error
        Ax *= k_smooth

        # Add SMOOTH contribution of this curve to result
        result_x += Ax

        return result_x

    @staticmethod
    def optimize_implicit_fast_with_weights(  # todo: pass list of cluster
            grid: Grid,
            number_of_vector_fields: int,
            paths: list[PolygonalPath2D],
            final_vector_fields: list[VectorField2D],
            map_curve_to_vector_field: list[int],
            map_curve_to_error: list[float],

            smoothness_weight: float
    ):
        """
        Performs a K-means-like alternating optimization for clustering curves
        into a fixed number of vector fields. Alternates between optimizing the
        vector fields (M-step) and reassigning curves (E-step) until convergence
        or reaching the iteration limit.
        """

        # --- Initialization ---
        # Build curve descriptions and compute total curve length (float)
        curve_descriptions, total_curve_length = set_constraints(paths, grid)

        # Create vector fields (list of two component arrays per field)
        vector_fields: list[VectorField2D] = [
            VectorField2D([
                np.zeros(shape=grid.get_resolution_x()*grid.get_resolution_y()),
                np.zeros(shape=grid.get_resolution_x()*grid.get_resolution_y())
            ])
            for _ in range(number_of_vector_fields)
        ]

        # First assignment
        f_first: tuple[list[int], list[list[int]]] = compute_first_assignment(
            grid, number_of_vector_fields, curve_descriptions, total_curve_length, smoothness_weight
        )
        first_assignments, map_vector_field_curves = f_first

        # Copy initial mapping
        for i in range(len(first_assignments)):
            map_curve_to_vector_field[i] = first_assignments[i]

        # --- Optimization Loop ---
        number_of_iterations = 0
        total_error = 1e20  # infinity

        while number_of_iterations < 100:
            total_change = [0]

            print(f"Before optimization: {total_error}")

            optimize_all_vector_fields(
                vector_fields, grid, map_vector_field_curves,
                curve_descriptions, total_curve_length, smoothness_weight
            )

            total_error = get_total_error(
                curve_descriptions, vector_fields, map_curve_to_vector_field,
                total_curve_length, smoothness_weight, grid
            )
            print(f"After optimization: {total_error}")

            optimize_assignments(
                total_change, [total_error],
                map_curve_to_vector_field, map_vector_field_curves,
                map_curve_to_error, vector_fields,
                curve_descriptions, total_curve_length, smoothness_weight, grid
            )

            total_error = get_total_error(
                curve_descriptions, vector_fields, map_curve_to_vector_field,
                total_curve_length, smoothness_weight, grid
            )

            print(f"After assignment: {total_error} changes: {total_change[0]}")

            repopulate_empty_cluster(map_vector_field_curves, map_curve_to_vector_field, vector_fields)

            number_of_iterations += 1
            if total_change[0] == 0:  # convergence
                break

        # --- Finalize results ---
        for i in range(number_of_vector_fields):
            final_vector_fields[i][0].set_values(vector_fields[i][0])
            final_vector_fields[i][1].set_values(vector_fields[i][1])


def compute_error_implicit(
        vector_field: VectorField2D,
        curve: CurveDescription,
        total_curve_length: float,
        smoothness_weight: float
) -> float:
    """
    error of a single curve in a vector field
    """

    x_component: np.ndarray[float] = vector_field[0]
    y_component: np.ndarray[float] = vector_field[1]

    error: float = 0.0

    vx = np.zeros(2 * len(curve.segments), dtype=float)
    vy = np.zeros(2 * len(curve.segments), dtype=float)

    for i in range(len(curve.segments)):
        # Segment provides generic add_cx/add_cTx for scalar components; use for both axes
        curve.segments[i].add_cx(vx, x_component)
        curve.segments[i].add_cx(vy, y_component)

    vx -= curve.rhsx
    vy -= curve.rhsy

    # LT . L = [[1/3 1/6] [1/6 1/3]]
    for i in range(0, len(vx), 2):
        this_error_x = (vx[i] * vx[i] + vx[i] * vx[i + 1] + vx[i + 1] * vx[i + 1]) / 3.0
        this_error_y = (vy[i] * vy[i] + vy[i] * vy[i + 1] + vy[i + 1] * vy[i + 1]) / 3.0
        error += (this_error_x + this_error_y) * curve.length

    assert error >= 0.0

    return error * (1.0 - smoothness_weight) / total_curve_length


def optimize_vector_field_with_weights(  # todo: may be a method of cluster class
        grid: Grid,
        cluster: Cluster,
        total_curve_length: float,
        smoothness_weight: float
) -> None:
    """
    optimize a single vector field (using smoothness)

    Given an initial guess for the vector field components,
    construct the RHS from curve constraints
    and solve two independent linear systems (one per component) using CG.

    The solution overwrites the provided initialGuessX/Y vectors.
    """

    initial_guess_x: np.ndarray[float] = cluster.vector_field[0]
    initial_guess_y: np.ndarray[float] = cluster.vector_field[1]

    number_of_vertices = grid.get_resolution_x() * grid.get_resolution_y()

    # independent terms //  rhs terms //  b_x, b_y
    indepx = np.zeros(number_of_vertices, dtype=float)
    indepy = np.zeros(number_of_vertices, dtype=float)
    # load values into independent terms
    for curve in cluster.curves:  # for each curve

        for j in range(len(curve.segments)):  # for each segment in curve
            # Sum contributions into the RHS vectors.
            k_factor = (1.0 - smoothness_weight) * (curve.segments[j].timestamps[1] - curve.segments[j].timestamps[0]) / total_curve_length  # weighting factor  # Each segment's influence is weighted by the [ (1 - smoothness_weight) data-term factor ] and [ its relative curve length ].
            curve.segments[j].add_cTx(indepx, curve.rhsx, k_factor)
            curve.segments[j].add_cTx(indepy, curve.rhsy, k_factor)

    problem = ProblemSettings(grid, cluster.curves_indices, cluster.curves, total_curve_length, smoothness_weight)

    # initial guesses: previous vector fields
    x0 = initial_guess_x.copy()
    y0 = initial_guess_y.copy()

    # solve linear system using Conjugate Gradient (cg_solve)
    x, x_exit_code = cg_solve(problem, indepx, x0)
    y, y_exit_code = cg_solve(problem, indepy, y0)

    # update previous vector vield values
    initial_guess_x[:] = x
    initial_guess_y[:] = y

def cg_solve(
        problem: ProblemSettings,
        b: np.ndarray[float],
        x0: np.ndarray[float]  # initial guess
) -> tuple[np.ndarray, int]:
    """
    interface for scipy cg solver

    return exit code
    """

    # SET SCIPY CG SOLVER PARAMETERS

    shape_A = (
        problem.grid.get_resolution_x() * problem.grid.get_resolution_y(),
        problem.grid.get_resolution_x() * problem.grid.get_resolution_y()
    )
    matvec_A = partial(VFKM.multiply_by_A, problem=problem)  # pre-set 'problem' parameter
    linear_operator_A = LinearOperator(shape=shape_A, matvec=matvec_A, dtype=float)

    # CALL SCIPY CG SOLVER

    x, exit_code = cg(A=linear_operator_A, b=b, x0=x0)

    return x, exit_code


def compute_first_assignment(
    grid: Grid,
    number_of_vector_fields: int,
    curves: list[CurveDescription],
    total_curve_length: float,
    smoothness_weight: float
) -> tuple[list[int], list[list[int]]]:  # mapCurveToVectorField, mapVectorFieldTo Curves
    """
    Generate an initial clustering of curves into vector fields.
    Strategy:
        - For each vector field, pick the currently worst-fitted curve,
        - Optimize a vector field for that single-curve seed,
        - Then assign every curve to its best candidate among the generated vector fields.
    Returns:
        (mapCurveToVectorField, mapVectorFieldCurves)
    """

    # Initialize per-curve errors to a large value
    errors: list[float] = [1e10] * len(curves)

    # Each vector field is a pair (xComponent, yComponent)
    vector_fields: list[VectorField2D] = []

    for i in range(number_of_vector_fields):
        curve_indices: list[int] = []

        # Seed each vector field with the currently worst-fitting curve

        worst_index: int = int(np.argmax(errors))
        curve_indices.append(worst_index)

        # Initialize the vector field components
        num_vertices: int = grid.get_resolution_x()*grid.get_resolution_y()
        x_component: np.ndarray[float] = np.zeros(shape=num_vertices, dtype=float)
        y_component: np.ndarray[float] = np.zeros(shape=num_vertices, dtype=float)

        # Optimize the vector field
        optimize_vector_field_with_weights(
            grid=grid,
            initial_guess_x=x_component, initial_guess_y=y_component,
            curve_indices=curve_indices,
            curve_descriptions=curves,
            total_curve_length=total_curve_length,
            smoothness_weight=smoothness_weight
        )

        # Store optimized components
        vector_fields.append(VectorField2D([x_component, y_component]))

        # Update per-curve error estimates
        for j, curve in enumerate(curves):
            new_error = compute_error_implicit(
                vector_field=vector_fields[i],
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight,
                curve=curve
            )
            errors[j] = min(errors[j], new_error)

    # Assign each curve to the best vector field
    result: list[int] = []
    result_indices: list[list[int]] = [[] for i in range(number_of_vector_fields)]

    for i, curve in enumerate(curves):

        curve_errors = [
            compute_error_implicit(
                vector_field=vf,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight,
                curve=curve
            )
            for vf in vector_fields
        ]

        best_index: int = int(np.argmin(curve_errors))
        result_indices[best_index].append(i)
        result.append(best_index)

    return result, result_indices


def set_constraints(
        polygonal_paths: list[PolygonalPath2D],
        grid: Grid
):
    """
    Prepare per-curve CurveDescription objects from raw PolygonalPath inputs.
    Also accumulates the total length used for normalization in optimization weightings.

    Parameters:
        polygonal_paths (list[PolygonalPath2D]): raw curve paths
        grid (Grid): grid object with clip_line() and curve_description() methods

    Returns:
        tuple[list[CurveDescription], float]: (curve_descriptions, total_curve_length)
    """

    total_curve_length = 0.0
    curve_descriptions: list[CurveDescription] = []

    number_of_curves = len(polygonal_paths)

    for i in range(number_of_curves):
        pp:PolygonalPath2D = polygonal_paths[i]

        # Validate that input times are non-decreasing.
        for j in range(pp.number_of_points() - 1):
            if pp.get_point(j + 1).time < pp.get_point(j).time:
                print("Line is broken, has backward time.", flush=True)

        bad_break = False

        # Clip / tessellate the path to the grid
        grid.clip_line(pp)

        # Verify tessellation didn't introduce non-monotonic times.
        for j in range(pp.number_of_points() - 1):
            if pp.get_point(j + 1).time < pp.get_point(j).time:
                print(f"{i} - Line clipper is broken, introduced backward time: "
                      f"{pp.get_point(j + 1).time} {pp.get_point(j).time}", flush=True)
                bad_break = True
                exit(1) # todo: solve appending null curve
                break


        curve: CurveDescription = CurveDescription(path=pp, grid=grid)
        total_curve_length += curve.length

        curve_descriptions.append(curve)

    return curve_descriptions, total_curve_length


def optimize_all_vector_fields(  # todo: change to optmize clusters. maybe make a cluster method
    grid: Grid,
    clusters: list[Cluster],
    total_curve_length: float,
    smoothness_weight: float
):
    """
    Optimize each vector field independently using its assigned curves.
    Equivalent to the M-step in an EM/clustering process: given assignments
    (map_vector_field_curves), optimize the vector field parameters to minimize error.
    """
    for cluster in clusters:

        current_vector_field: VectorField2D = cluster.vector_field
        curve_indices = cluster.curves_indices

        x_component, y_component = current_vector_field

        # Solve for the best vector field given assigned curves
        optimize_vector_field_with_weights(
            grid=grid,
            cluster=cluster,
            total_curve_length=total_curve_length,
            smoothness_weight=smoothness_weight
        )


def get_total_error(
        clusters: list[Cluster],
        total_curve_length: float,
        smoothness_weight: float,
        grid: Grid
):
    """
    Compute the overall energy being minimized.
    It is the sum of per-curve data-fitting errors (computed with compute_error_implicit)
    plus the smoothness penalty for each vector field
    (scaled by the fraction of total curve length assigned to that field).
    """

    total_error: float = 0.0

    for i, cluster in enumerate(clusters):

        cluster_length: float = 0  # used as normalization factor in smoothness error

        # FIT ERROR
        for curve in cluster.curves:
            cluster_length+=curve.length
            total_error += compute_error_implicit(
                vector_field=cluster.vector_field, curve=curve,
                total_curve_length=total_curve_length, smoothness_weight=smoothness_weight
            )

        # SMOOTHNESS ERROR
        current_vector_field: VectorField2D = cluster.vector_field
        vector_field_copy: VectorField2D = current_vector_field.copy()
        grid.multiply_by_laplacian(vector_field=vector_field_copy)
        weight_factor = smoothness_weight * (cluster_length / total_curve_length)

        total_error += np.linalg.norm(vector_field_copy[0]) * weight_factor
        total_error += np.linalg.norm(vector_field_copy[1]) * weight_factor

    return total_error


def optimize_assignments(  # ASSIGN STEP  # todo: return list of clusters
    total_change: list[int],
    total_error: list[float],

    map_curve_to_vector_field: list[int],
    map_vector_field_curves: list[list[int]],
    map_curve_to_error: list[float],

    vector_fields: list[VectorField2D],
    curves: list[CurveDescription],

    total_curve_length: float,
    smoothness_weight: float
):
    """
    Assign each curve to the vector field that minimizes its contribution
    to the objective, then update the per-vector-field curve lists.
    Also returns the total number of changes and total error.
    """

    total_error[0] = 0.0
    total_change[0] = 0

    number_of_curves = len(curves)
    number_of_vector_fields = len(vector_fields)

    # E-step: update cluster assignments for each curve
    for i in range(number_of_curves):  # todo: simplify: generate array of map vf to error and get argmin
        change = False
        current_curve = curves[i]

        vector_field_index = map_curve_to_vector_field[i]
        new_vector_field_index = vector_field_index

        current_vector_field = vector_fields[vector_field_index]
        error = compute_error_implicit(
            vector_field=current_vector_field,
            total_curve_length=total_curve_length,
            smoothness_weight=smoothness_weight,
            curve=current_curve
        )

        for j in range(number_of_vector_fields):
            if j == vector_field_index:
                continue

            vector_field = vector_fields[j]
            current_error = compute_error_implicit(
                vector_field=vector_field,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight,
                curve=current_curve
            )

            if current_error < error:
                new_vector_field_index = j
                error = current_error
                change = True

        total_change[0] += int(change)
        total_error[0] += error

        # Assign best vector field to this curve and store error
        map_curve_to_vector_field[i] = new_vector_field_index
        map_curve_to_error[i] = error

    # Update map_vector_field_curves
    for container in map_vector_field_curves:
        container.clear()

    for i in range(number_of_curves):
        vector_field_index = map_curve_to_vector_field[i]
        map_vector_field_curves[vector_field_index].append(i)



def repopulate_empty_cluster(  # todo: pass list of clusters
        map_vector_field_curves: list[list[int]],
        map_curve_to_vector_field: list[int],
        vector_fields: list[VectorField2D]
):
    """
    If a cluster has no assigned curves, re-populate it by splitting the
    largest existing cluster into two. This avoids empty clusters and
    keeps the number of vector fields constant.
    """

    number_of_vector_fields = len(vector_fields)

    for i in range(number_of_vector_fields):
        container = map_vector_field_curves[i]

        if len(container) == 0:
            # Reset vector field components to zero before refill
            vector_fields[i][0].set_values(0.0)
            vector_fields[i][1].set_values(0.0)

            max_index = -1
            max_size = 0

            # Find the largest cluster
            for j in range(number_of_vector_fields):
                if len(map_vector_field_curves[j]) > max_size:
                    max_size = len(map_vector_field_curves[j])
                    max_index = j

            n1, n2 = [], []

            # Split the largest cluster into two by alternating assignment
            for j in range(max_size):
                curve = map_vector_field_curves[max_index][j]
                if j % 2 == 1:
                    n1.append(curve)
                    map_curve_to_vector_field[curve] = i
                else:
                    n2.append(curve)
                    map_curve_to_vector_field[curve] = max_index

            map_vector_field_curves[i] = n1
            map_vector_field_curves[max_index] = n2
