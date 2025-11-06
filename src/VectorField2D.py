from __future__ import annotations

import numpy as np
from typing import Optional


class VectorField2D:
    """
	consisted by a np.2Darray that contains:
	a nd array for x-axis,
	a nd array for y-axis

	"""

    vector_field: list[np.ndarray[float]]  # a np array for each axis  # calculated during optimization step

    def __init__(
            self,
            vector_field: list[np.ndarray[float]]
    ):
        self.vector_field = vector_field

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
