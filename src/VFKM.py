""""""

import numpy as np

from scipy.sparse.linalg import cg  # conjugate gradient solver
from scipy.sparse.linalg import LinearOperator  # class to incorporate original "multiplyByA()"

from functools import partial

from Grid import Grid, CurveDescription
from src.VectorField2D import VectorField2D


class ProblemSettings:
    """
    struct carrying problem data for solvers.
    """

    grid: Grid
    curve_indices: np.ndarray[int]  # respective to cluster
    curve_descriptions: np.ndarray[CurveDescription]
    total_curve_length: float  # summed length of all curves (used for normalization)
    smoothness_weight: float

    def __init__(
            self,
            grid: Grid,
            curve_indices: np.ndarray[int],
            curve_descriptions: np.ndarray[CurveDescription],
            total_curve_length: float,
            smoothness_weight: float
    ):
        self.grid = grid
        self.curve_indices = curve_indices
        self.curve_descriptions = curve_descriptions
        self.total_curve_length = total_curve_length
        self.smoothness_weight = smoothness_weight


class VFKM:

    def __init__(
            self,
            size: int
    ):
        """
        initialize new VF axis
        """

        ax = np.zeros(shape=size)
        ay = np.zeros(shape=size)


def compute_error_implicit(
        grid: Grid,
        x_component: np.ndarray[float],
        y_component: np.ndarray[float],
        total_curve_length: float,
        smoothness_weight: float,
        curve: CurveDescription
) -> float:
    error: float = 0.0

    vx = np.zeros(2 * len(curve.segments), dtype=float)
    vy = np.zeros(2 * len(curve.segments), dtype=float)

    for i in range(len(curve.segments)):
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


def optimize_vector_field_with_weights(
        grid: Grid,
        initial_guess_x: np.ndarray[float], initial_guess_y: np.ndarray[float],
        curve_indices: list[int],
        curve_descriptions: list[CurveDescription],
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

    number_of_vertices = grid.get_resolution_x() * grid.get_resolution_y()

    # independent terms //  rhs terms //  b_x, b_y
    indepx = np.zeros(number_of_vertices, dtype=float)
    indepy = np.zeros(number_of_vertices, dtype=float)

    for k in range(len(curve_indices)):  # for each curve
        i = curve_indices[k]
        curve: CurveDescription = curve_descriptions[i]

        for j in range(len(curve.segments)):  # for each segment in curve
            # Sum contributions into the RHS vectors.
            k_factor = (1.0 - smoothness_weight) * (curve.segments[j].timestamps[1] - curve.segments[j].timestamps[0]) / total_curve_length  # weighting factor  # Each segment's influence is weighted by the [ (1 - smoothness_weight) data-term factor ] and [ its relative curve length ].
            curve.segments[j].add_cTx(indepx, curve.rhsx, k_factor)
            curve.segments[j].add_cTx(indepy, curve.rhsy, k_factor)

    problem = ProblemSettings(grid, curve_indices, curve_descriptions, total_curve_length, smoothness_weight)

    # initial guesses: previous vector fields
    x0 = initial_guess_x.copy()
    y0 = initial_guess_y.copy()

    # solve linear system using Conjugate Gradient (cg_solve)
    x, x_exit_code = cg_solve(problem, indepx, x0)
    y, y_exit_code = cg_solve(problem, indepy, y0)

    # update previous vector vield values
    initial_guess_x[:] = x
    initial_guess_y[:] = y


def multiply_by_A(
        v: np.ndarray[float],
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
        curve.add_cTcx(result_x, v, k_fit)


    # SMOOTH PENALTY

    Ax = v.copy()  # Ax = v (initial copy of v)

    # DISCRETIZED LAPLACIAN
    grid.multiply_by_laplacian2(Ax)  # Ax = L*x
    grid.multiply_by_laplacian2(Ax)  # Ax = L^T * L * x (laplaciano discretizado é uma matriz simétrica L^T = L)

    # Normalization
    k_smooth = smoothness_weight / num_vertices  # normalization factor for SMOOTH error
    Ax *= k_smooth

    # Add SMOOTH contribution of this curve to result
    result_x += Ax


    return result_x


def cg_solve(
        problem: ProblemSettings,
        b: np.ndarray[float],
        x0: np.ndarray[float]  # intial guess
) -> (np.ndarray[float], int):
    """
    interface for scipy cg solver

    return exit code
    """

    # SET SCIPY CG SOLVER PARAMETERS

    shape_A = (
        problem.grid.resolution_x * problem.grid.resolution_y,
        problem.grid.resolution_x * problem.grid.resolution_y
    )
    matvec_A = partial(multiply_by_A, problem=problem)  # pre-set 'problem' parameter
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
    vector_fields: np.ndarray[VectorField2D] = np.empty(number_of_vector_fields, dtype=VectorField2D)

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
        vector_fields[i] = (x_component, y_component)

        # Update per-curve error estimates
        for j, curve in enumerate(curves):
            new_error = compute_error_implicit(
                grid=grid,
                x_component=x_component, y_component=y_component,
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
                grid=grid,
                x_component=vf[0], y_component=vf[1],
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
