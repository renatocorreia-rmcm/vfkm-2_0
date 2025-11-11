from __future__ import annotations

import numpy as np

class VectorField2D:
    """
	list of axis

	"""

    vector_field: list[np.ndarray[float]]  # a np array for each axis  # calculated during optimization step

    def __init__(
            self,
            components: list[np.ndarray[float]]
    ):
        self.vector_field = components

    def __getitem__(self, item):
        return self.vector_field[item]

    def copy(self) -> VectorField2D:
        """
        Returns a deep copy of this VectorField2D.
        The underlying NumPy arrays are duplicated to avoid shared memory.
        """

        # Deep copy of the NumPy arrays
        copied_field: list[np.ndarray[float]] = [np.copy(component) for component in self.vector_field]
        return VectorField2D(copied_field)
