from __future__ import annotations
from typing import Optional

from numpy.ma.core import absolute
from scipy.interpolate import barycentric_interpolate

from VectorField2D import VectorField2D
from Grid import CurveDescription, Grid
import numpy as np

from src.Grid_aux import TriangularFace, PointLocation


class Cluster:
    """
    represents a set of trajectories + a vector field
    """

    name: str  # f"{name of parent}:{amount of curves in this cluster}"

    # HIERARCHY
    parent: Optional[Cluster]
    children: list[Cluster]

    # VECTOR FIELD
    vector_field: Optional[VectorField2D]  # todo: make possible to pass grid resolution and auto create vf with np.zeros

    # CURVES
    curves: list[CurveDescription]

    # ERRORS
    curve_errors: list[float]  # for each indice in self.curves_indices
    total_error: float  # total error of cluster
    max_error: float = 0.0  # among all curves in this cluster.

    def __init__(
            self,
            grid: Grid,
            name: str = "",
            parent: Optional[Cluster] = None
    ):
        """
        :param name: f"{name of parent}:{amount of curves in this cluster}"
        :param parent: The parent cluster of this cluster. Defaults to None for the root cluster

        """

        self.name = name  # in 1.0, it is computed outside initializer call (and passed as arg). Could this be done by the own Cluster.__init__ ?

        self.parent = parent
        self.children = []

        self.vector_field = VectorField2D([
            np.zeros(shape=grid.resolution_x * grid.resolution_y),
            np.zeros(shape=grid.resolution_x * grid.resolution_y)
        ])

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
        self.curves.clear()
        self.curve_errors.clear()
        self.total_error = 0
        self.max_error = 0

    def optimize_vector_field(  # todo: to test: its not generating same output than original
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
            j=0
            for segment in curve.segments:  # for each segment in curve

                # todo: SOLVE: Ks ARE DIFFERENT. SEEMS TO KEEP 1/2 PROPORTION

                k_factor: float = (1.0 - smoothness_weight) * (segment.timestamps[1] - segment.timestamps[0]) / total_curve_length

                k_cpp = (1.0 - smoothness_weight) * (curve.segments[j].timestamps[1] - curve.segments[j].timestamps[0]) / total_curve_length


                # Sum contributions into the RHS vectors.
                segment.add_cTx(indepx, curve.rhsx, k_factor)
                segment.add_cTx(indepy, curve.rhsy, k_factor)

                print(f"k       {k_factor}")
                print(f"kcpp    {k_cpp}")
                print(f"c       {curve.index}")
                print(f"segment {segment.index}")

                print("\nINDEP X\n")
                print(indepx)
                print("\nINDEP Y\n")
                print(indepy)

                # todo: solve: indep[2] is always 0

                input("\n\nINPUT ANYTHING TO CONTINUE\n\n")
                j+=1

        """ DEBUG RSH's
        
        print("\nINDEP X\n")
        print(indepx)
        print(indepy)
        print("\nINDEP Y\n")
        """

        # import here to avoid circular import at module load time
        from VFKM import ProblemSettings, cg_solve

        problem = ProblemSettings(grid, self.curves, total_curve_length, smoothness_weight)

        # initial guesses: previous vector fields
        x0 = self.vector_field[0].copy()
        y0 = self.vector_field[1].copy()

        # solve linear system using Conjugate Gradient (cg_solve)
        x = cg_solve(problem, indepx, x0)
        y = cg_solve(problem, indepy, y0)

        # update previous vector field values
        self.vector_field[0][:] = x
        self.vector_field[1][:] = y


    def get_vector(self, relative_coordinates: np.ndarray[float], grid:Grid) -> np.ndarray[float]:
        """
        pass relative coordinates ([0,1], [0,1])
        return vector this position
        """

        grid_coordinates: np.ndarray[float] = relative_coordinates * np.array([grid.get_resolution_x()-1, grid.get_resolution_y()-1])  # todo: check if coordinate transformation is correct

        # get face
        face: TriangularFace = grid.get_face_where_point_lies(grid_coordinates)
        # get barycentric coordinates
        point_location: PointLocation = PointLocation(face=face)
        grid.locate_point(point_loc = point_location, point_coordinates=grid_coordinates)
        barycentric_cords: np.ndarray[float] = point_location.barycentric_cords
        # get vertex vectors
        vertex_vectors: list[np.ndarray[float]] = [
            np.array([
                [ self.vector_field[0][face.indices[i]], self.vector_field[1][face.indices[i]]]
            ]) for i in range(3)
        ]

        return  barycentric_cords[0] * vertex_vectors[0] + \
                barycentric_cords[1] * vertex_vectors[1] + \
                barycentric_cords[2] * vertex_vectors[2]
