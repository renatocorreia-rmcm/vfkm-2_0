import numpy as np
from typing import Optional, List

from VectorField import VectorField


class Cluster:
	"""
	Represents a set of trajectories + a vector field

	"""

	def __init__(self, name: str = "", parent: Optional['Cluster'] = None,  vector_field: VectorField = None):
		"""
		Initializes a new Cluster instance.

		Args:
			name (str): The name or identifier for the cluster.  f"{name of parent}:{amount of curves in this cluster}
			parent (Optional['Cluster']): The parent cluster of this cluster.
										  Defaults to None for the root cluster.
		"""

		self.name: str = name

		# HIERARCHY
		self.parent: Optional['Cluster'] = parent
		self.children: List['Cluster'] = []

		# VECTOR FIELD
		self.vector_field: Optional[VectorField] = vector_field

		# CURVES
		self.curves_indices: List[int] = []  # indices of curves bellonging to this cluster

		# ERRORS
		self.curve_errors: List[float] = []
		self.error: float = 0.0  # total error of cluster
		self.max_error: float = 0.0  # among all curves in this cluster.

	def clear_children(self):
		self.children.clear()

	def add_child(self, child_cluster: 'Cluster'):
		self.children.append(child_cluster)
		child_cluster.parent = self

	def __str__(self):
		vf_status = "Computed" if self.vector_field is not None else "Not computed"
		return (
			f"Cluster(name={self.name}, "
			f"error={self.error:.2f}, "
			f"num_curves={len(self.curves_indices)}, "
			f"num_children={len(self.children)}, "
			f"vector_field={vf_status})"
		)

	def __repr__(self):
		"""
		Returns the 'official' string representation of the object.
		"""
		return self.__str__()



