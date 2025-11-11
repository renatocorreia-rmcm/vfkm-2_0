from __future__ import annotations
from typing import Optional

from VectorField2D import VectorField2D
from src.Grid import CurveDescription, Grid
import numpy as np


class Cluster:
    """
    represents a set of trajectories + a vector field

    """

    name: str  # f"{name of parent}:{amount of curves in this cluster}"

    # HIERARCHY
    parent: Optional[Cluster]
    children: list[Cluster]

    # VECTOR FIELD
    vector_field: VectorField2D

    # CURVES
    curves_indices: list[int]  # indices of curves belonging to this cluster
    curves: list[CurveDescription]

    # ERRORS
    curve_errors: list[float]  # for each indice in self.curves_indices
    total_error: float  # total error of cluster
    max_error: float = 0.0  # among all curves in this cluster.

    def __init__(
            self,
            vector_field: VectorField2D,
            name: str = "",
            parent: Optional[Cluster] = None
    ):
        """
        :param name: f"{name of parent}:{amount of curves in this cluster}"
        :param parent:
        :param vector_field: The parent cluster of this cluster. Defaults to None for the root cluster

        """

        self.name = name  # in 1.0, it is computed outside initializer call (and passed as arg). Could this be done by the own Cluster.__init__ ?

        self.parent = parent
        self.children = []

        self.vector_field = vector_field

        self.curves_indices = []
        self.curves = []

        self.curve_errors = []
        self.total_error = 0.0
        self.max_error = 0.0

    def clear_children(self) -> None:
        self.children.clear()

    def add_child(self, child_cluster: Cluster) -> None:
        self.children.append(child_cluster)
        child_cluster.parent = self

    def clear_curves(self):
        """
		used in assign step
		"""
        self.curves_indices.clear()
        self.curves.clear()
        self.curve_errors.clear()
        self.total_error = 0
        self.max_error = 0

    def optimize_vector_field(
            self,
            grid: Grid,
            smoothness_weight: float,
            total_curve_length: float
    ):
        """
        optimize this cluster vector field (using smoothness)

        Construct the RHS from curve constraints
        and solve two independent linear systems (one per component) using CG.
        """

        number_of_vertices: int = grid.get_resolution_x() * grid.get_resolution_y()

        # independent terms //  rhs terms //  b_x, b_y
        indepx = np.zeros(number_of_vertices, dtype=float)
        indepy = np.zeros(number_of_vertices, dtype=float)

        # load values into independent terms
        for curve in self.curves:  # for each curve
            for j in range(len(curve.segments)):  # for each segment in curve
                k_factor: float = (1.0 - smoothness_weight) * (
                        curve.segments[j].timestamps[1] - curve.segments[j].timestamps[
                    0]) / total_curve_length  # weighting factor
                # Sum contributions into the RHS vectors.
                curve.segments[j].add_cTx(indepx, curve.rhsx, k_factor)
                curve.segments[j].add_cTx(indepy, curve.rhsy, k_factor)

        # import here to avoid circular import at module load time
        from src.VFKM import ProblemSettings, cg_solve

        problem = ProblemSettings(grid, self.curves, total_curve_length, smoothness_weight)

        # initial guesses: previous vector fields
        x0 = self.vector_field[0].copy()
        y0 = self.vector_field[1].copy()

        # solve linear system using Conjugate Gradient (cg_solve)
        x, x_exit_code = cg_solve(problem, indepx, x0)
        y, y_exit_code = cg_solve(problem, indepy, y0)

        # update previous vector field values
        self.vector_field[0][:] = x
        self.vector_field[1][:] = y
