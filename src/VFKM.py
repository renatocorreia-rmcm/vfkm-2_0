""""""

import numpy as np

from scipy.sparse.linalg import cg  # conjugate gradient solver
from scipy.sparse.linalg import LinearOperator  # class to incorporate original "multiplyByA()"

from functools import partial

from Grid import Grid, CurveDescription
from src.Cluster import Cluster
from src.PolygonalPath2D import PolygonalPath2D
from src.VectorField2D import VectorField2D


class ProblemSettings:  # used only by MULTIPLY_BY_A
    """
    Description of the Ax=b problem
    """

    grid: Grid
    curve_descriptions: list[CurveDescription]
    total_curve_length: float  # summed length of all curves (used for normalization)
    smoothness_weight: float

    def __init__(
            self,
            grid: Grid,
            curve_descriptions: list[CurveDescription],
            total_curve_length: float,
            smoothness_weight: float
    ):
        self.grid = grid
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

        curve_descriptions = problem.curve_descriptions

        total_curve_length = problem.total_curve_length
        k_fit = (1.0 - smoothness_weight) / total_curve_length  # normalization factor for FIT error

        for curve in curve_descriptions:
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
    def optimize_implicit_fast_with_weights(
            grid: Grid,
            paths: list[PolygonalPath2D],
            number_of_vector_fields: int,
            smoothness_weight: float
    ) -> list[Cluster]:
        """  TOP LEVEL OPTIMIZATION (full process)

        Performs a K-means-like alternating optimization for clustering curves
        into a fixed number of vector fields. Alternates between optimizing the
        vector fields (M-step) and reassigning curves (E-step) until convergence
        or reaching the iteration limit.
        """

        # TESSELATION
        curve_descriptions: list[CurveDescription]
        total_curve_length: float
        curve_descriptions, total_curve_length = set_constraints(paths, grid)

        # FIRST ASSIGNMENT
        clusters: list[Cluster] = compute_first_assignment(
            grid=grid,
            number_of_vector_fields=number_of_vector_fields,
            curves=curve_descriptions,
            total_curve_length=total_curve_length,
            smoothness_weight=smoothness_weight
        )

        # --- Optimization Loop ---
        number_of_iterations: int = 100
        total_error: float = float('inf')


        for i in range(number_of_iterations):
            print(f"Before optimization: {total_error}")


            """
                OPTMIZE
            """
            optimize_all_clusters_vector_fields(
                grid=grid,
                clusters=clusters,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight
            )
            total_error = get_total_error(
                grid=grid,
                clusters=clusters,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight
            )
            print(f"After optimization: {total_error}")


            """
                ASSIGN
            """
            total_error, total_change = optimize_all_clusters_assignments(  # this total_error consider only the FIT error
                clusters=clusters,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight
            )

            total_error = get_total_error(  # this total_error consider both FIT and SMOOTH error
                grid=grid,
                clusters=clusters,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight
            )

            print(f"After assignment: {total_error} changes: {total_change}")

            repopulate_all_empty_cluster_by_random(
                clusters=clusters
            )

            if total_change == 0:  # convergence
                print(f"Converged in {i} iterations.")
                return clusters

        print(f"iteration limit reached ({number_of_iterations} iterations).")
        return clusters


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
) -> list[Cluster]:
    """  # CREATE LIST OF CLUSTERS

    Generate an initial clustering of curves
        - For each vector field, pick the currently worst-fitted curve,
        - Optimize a vector field for that single-curve seed,
        - Then assign every curve to its best candidate among the generated vector fields.
    """

    # Initialize per-curve errors to infinite
    errors: list[float] = [float('inf')] * len(curves)

    num_vertices: int = grid.get_resolution_x() * grid.get_resolution_y()
    # create the k clusters
    clusters: list[Cluster] = [
        Cluster(
            vector_field=VectorField2D([
                np.zeros(shape=num_vertices, dtype=float),
                np.zeros(shape=num_vertices, dtype=float)
            ])
        ) for _ in range(number_of_vector_fields)
    ]


    # --- FIT ---
    for cluster in clusters:

        # Seed each vector field with the currently worst-fitting curve
        worst_index: int = int(np.argmax(errors))

        cluster.curves.append(curves[worst_index])

        # Optimize the vector field
        cluster.optimize_vector_field(
            grid=grid,
            total_curve_length=total_curve_length,
            smoothness_weight=smoothness_weight
        )

        # Update per-curve error estimates
        for j, curve in enumerate(curves):
            new_error = compute_error_implicit(
                vector_field=cluster.vector_field,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight,
                curve=curve
            )
            errors[j] = min(errors[j], new_error)


    for cluster in clusters:
        cluster.clear_curves()
    # -- ASSIGN --
    for i, curve in enumerate(curves):

        curve_errors = [
            compute_error_implicit(
                vector_field=cluster.vector_field,
                curve=curve,
                total_curve_length=total_curve_length,
                smoothness_weight=smoothness_weight
            )
            for cluster in clusters
        ]
        best_index: int = int(np.argmin(curve_errors))

        clusters[best_index].curves.append(curve)

    return clusters


def set_constraints(
        polygonal_paths: list[PolygonalPath2D],
        grid: Grid
) -> tuple[list[CurveDescription], float]:
    """
    TOP-LEVEL TESSELATION + TOTAL-PATH-SIZE COUNT

    Prepare per-curve CurveDescription objects from raw PolygonalPath inputs.
    Also accumulates the total length used for normalization in optimization weightings.

    Parameters:
        polygonal_paths (list[PolygonalPath2D]): raw curve paths
        grid (Grid): grid object with clip_line() and curve_description() methods

    Returns:
        tuple[list[CurveDescription], float]: (curve_descriptions, total_curve_length)
    """

    total_curve_length: float = 0
    curve_descriptions: list[CurveDescription] = []

    number_of_curves: int = len(polygonal_paths)

    for i in range(number_of_curves):
        pp: PolygonalPath2D = polygonal_paths[i]

        # Validate that input times are non-decreasing.
        for j in range(pp.number_of_points() - 1):
            if pp.get_point(j + 1).time < pp.get_point(j).time:
                raise Exception("Line is broken, has backward time.")

        # Clip / tessellate the path to the grid
        grid.clip_line(pp)

        # Verify tessellation didn't introduce non-monotonic times.
        # Verify tessellation didn't introduce non-monotonic times.
        for j in range(pp.number_of_points() - 1):
            if pp.get_point(j + 1).time < pp.get_point(j).time:
                raise Exception(f"{i} - Line clipper is broken, introduced backward time: "
                      f"{pp.get_point(j + 1).time} {pp.get_point(j).time}")

        curve: CurveDescription = CurveDescription(path=pp, grid=grid)
        curve.index = i
        total_curve_length += curve.length

        curve_descriptions.append(curve)

    return curve_descriptions, total_curve_length


def optimize_all_clusters_vector_fields(
        grid: Grid,
        clusters: list[Cluster],
        total_curve_length: float,
        smoothness_weight: float
):
    """
    Optimize each cluster vector field independently using its assigned curves.
    Equivalent to the M-step in an EM/clustering process: given assignments
    , optimize the vector field parameters to minimize error.
    """
    for cluster in clusters:
        cluster.optimize_vector_field(
            grid=grid,
            total_curve_length=total_curve_length,
            smoothness_weight=smoothness_weight
        )


def get_total_error(
        clusters: list[Cluster],
        total_curve_length: float,
        smoothness_weight: float,
        grid: Grid
) -> float:
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
            cluster_length += curve.length
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


def optimize_all_clusters_assignments(  # ASSIGN STEP
        clusters: list[Cluster],
        total_curve_length: float,
        smoothness_weight: float
) -> tuple[float, int]:
    """
    Assign each curve to the vector field that minimizes its contribution
    to the objective, then update the per-vector-field curve lists.
    Also returns the total number of changes and total error.

    return total_error and total_change
    """

    total_error: float = 0.0
    total_change: int = 0

    # copy all cluster-curves assignments
    curves_by_cluster: list[list[CurveDescription]] = [cluster.curves[:] for cluster in clusters]

    for cluster in clusters:
        cluster.clear_curves()

    # for each cluster, check where its curves fit better
    for i_cluster, cluster in enumerate(clusters):

        for curve in curves_by_cluster[i_cluster]:
            changed = False

            errors = [  # error of curve in each cluster
                compute_error_implicit(
                    curve=curve,
                    vector_field=cluster2.vector_field,
                    total_curve_length=total_curve_length,
                    smoothness_weight=smoothness_weight
                )
                for cluster2 in clusters
            ]

            i_min_cluster = np.argmin(errors)
            if i_min_cluster != i_cluster:
                changed = True

            error = errors[i_min_cluster]

            total_change += int(changed)
            total_error += error

            clusters[i_min_cluster].curves.append(curve)
            clusters[i_min_cluster].curve_errors.append(error)

    return total_error, total_change


def repopulate_all_empty_cluster_by_random(  # todo: implement repopulate_all_empty_cluster_by_error
        clusters: list[Cluster]
) -> None:
    """
    If a cluster has no assigned curves, re-populate it by splitting the
    largest existing cluster into two. This avoids empty clusters and
    keeps the number of vector fields constant.

    could split bigger cluster by error
    - repopulate_cluster_random
    - repopulate_cluster_by_error  # may not work well for outliers
    """

    for cluster in clusters:

        if len(cluster.curves) == 0:  # 'cluster' now is the empty one

            # Reset vector field components to zero before refill
            cluster.vector_field[0].set_values(0.0)
            cluster.vector_field[1].set_values(0.0)

            len_clusters = [len(cluster.curves) for cluster in clusters]
            index_bigger = np.argmax(len_clusters)

            # Split the largest cluster into two
            cluster.curves = clusters[index_bigger].curves[1::2]
            clusters[index_bigger].curves = clusters[index_bigger].curves[0::2]
