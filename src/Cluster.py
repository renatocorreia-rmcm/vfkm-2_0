import numpy as np
from typing import Optional, List

from VectorField2D import VectorField2D


class Cluster:
	"""
	represents a set of trajectories(indices) + a vector field

	"""

	def __init__(
			self,
			name: str = "",
			parent: Optional['Cluster'] = None,
			vector_field: VectorField2D = None
	):
		"""

		:param name: f"{name of parent}:{amount of curves in this cluster}"
		:param parent:
		:param vector_field: The parent cluster of this cluster. Defaults to None for the root cluster
		"""

		# CLUSTER NAME
		self.name: str = name  # in 1.0, the "{name of parent}:{amount of curves in this cluster}" is computed outside initializer call (and passed as arg). Could this be done by the own Cluster.__init__ ?

		# HIERARCHY
		self.parent: Optional['Cluster'] = parent
		self.children: List['Cluster'] = []

		# VECTOR FIELD
		self.vector_field: Optional[VectorField2D] = vector_field

		# CURVES
		self.curves_indices: List[int] = []  # indices of curves bellonging to this cluster

		# ERRORS
		self.curve_errors: List[float] = []  # for each indice in self.curves_indices
		self.error: float = 0.0  # total error of cluster
		self.max_error: float = 0.0  # among all curves in this cluster.

	def clear_children(self) -> None:
		self.children.clear()

	def add_child(self, child_cluster: 'Cluster') -> None:
		self.children.append(child_cluster)
		child_cluster.parent = self

	def __str__(self) -> str:
		"""
		:return: easy-to-debug string representation
		"""

		vf_status = "Computed" if (self.vector_field is not None) else "Not computed"
		return (
			f"Cluster(name={self.name}, "
			f"error={self.error:.2f}, "
			f"num_curves={len(self.curves_indices)}, "
			f"num_children={len(self.children)}, "
			f"vector_field={vf_status})"
		)
