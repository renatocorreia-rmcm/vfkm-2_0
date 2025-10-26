""""""

import numpy as np

from scipy.sparse.linalg import cg  # conjugate gradient solver
from scipy.sparse.linalg import LinearOperator  # class to incorporate original "multiplyByA()"

from functools import partial

from Grid import Grid, CurveDescription


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

    def __init__(self, size: int):
        """
		initialize new VF axis
		"""

        ax = np.zeros(shape=size)
        ay = np.zeros(shape=size)


def compute_error_implicit(
        grid: Grid,
        vf_x_component: np.ndarray[float],
        vf_y_component: np.ndarray[float],
        total_curve_length: float,
        smoothness_weight: float,
        curve: CurveDescription
) -> float:
    error: float = 0.0

    vx = np.zeros(2 * len(curve.segments), dtype=float)
    vy = np.zeros(2 * len(curve.segments), dtype=float)

    for i in range(len(curve.segments)):
        curve.segments[i].add_cx(vx, vf_x_component)
        curve.segments[i].add_cx(vy, vf_y_component)

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
        curve_indices: np.ndarray[int],
        curve_descriptions: np.ndarray[CurveDescription],
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

    indepx = np.zeros(number_of_vertices, dtype=float)
    indepy = np.zeros(number_of_vertices, dtype=float)

    for k in range(len(curve_indices)):  # for each curve
        i = curve_indices[k]
        curve: CurveDescription = curve_descriptions[i]

        for j in range(len(curve.segments)):  # for each segment in curve
            # Sum contributions into the RHS vectors.
            k_factor = (1.0 - smoothness_weight) * (curve.segments[j].time[1] - curve.segments[j].time[
                0]) / total_curve_length  # weighting factor  # Each segment's influence is weighted by the [ (1 - smoothness_weight) data-term factor ] and [ its relative curve length ].
            curve.segments[j].add_cTx(indepx, curve.rhsx, k_factor)
            curve.segments[j].add_cTx(indepy, curve.rhsy, k_factor)

    problem = ProblemSettings(grid, curve_indices, curve_descriptions, total_curve_length, smoothness_weight)

    # aux vars for calling cg_solve()
    x = initial_guess_x.copy()
    y = initial_guess_y.copy()

    # solve linear system using Conjugate Gradient (cg_solve)
    cg_solve(problem, indepx, x)
    cg_solve(problem, indepy, y)

    initial_guess_x = x.copy()
    initial_guess_y = y.copy()


def multiply_by_laplacian(Ax: np.ndarray[float]) -> None:
	"""
	Overwrites AX

	:return:
	"""


def multiply_by_A(v: np.ndarray[float], problem: ProblemSettings) -> np.ndarray[float]:
    """
	Compute A*v without setting up whole matrix

	"""

    # todo: implement

    return v


def cg_solve(
        problem: ProblemSettings,
        b: np.ndarray[float],
        x: np.ndarray[float]
) -> int:
    """
	call scipy cg solver

	return exit code
	"""

    # SET SCIPY CG SOLVER PARAMETERS

    shape_A = (
        2 * problem.grid.resolution_x * problem.grid.resolution_y,
        2 * problem.grid.resolution_x * problem.grid.resolution_y
    )
    matvec_A = partial(multiply_by_A, problem=problem)  # pre-set 'problem' parameter
    linear_operator_A = LinearOperator(shape=shape_A, matvec=matvec_A, dtype=float)

    # CALL SCIPY CG SOLVER

    x, exit_code = cg(linear_operator_A, b)

    return exit_code
