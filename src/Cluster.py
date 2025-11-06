from __future__ import annotations

from typing import Optional

from VectorField2D import VectorField2D


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

    # ERRORS
    curve_errors: list[float]  # for each indice in self.curves_indices
    error: float  # total error of cluster
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

        self.parent: Optional[Cluster] = parent
        self.children: list[Cluster] = []

        self.vector_field: Optional[VectorField2D] = vector_field

        self.curves_indices: list[int] = []

        self.curve_errors: list[float] = []
        self.error: float = 0.0
        self.max_error: float = 0.0


    def clear_children(self) -> None:
        self.children.clear()

    def add_child(self, child_cluster: Cluster) -> None:
        self.children.append(child_cluster)
        child_cluster.parent = self
