from __future__ import annotations

import numpy as np
from typing import Optional


class VectorField2D:
    """
	consisted by a np.2Darray that contains:
	a nd array for x-axis,
	a nd array for y-axis

	"""

    vector_field: Optional[np.ndarray[np.ndarray[float]]]  # a np array for each axis  # calculated during optimization step

    def __init__(
            self,
            vector_field: Optional[np.ndarray[np.ndarray[float]]] = None
    ):
        self.vector_field = vector_field

    def __getitem__(self, item):
        return self.vector_field[item]

    def copy(self) -> VectorField2D:
        """
        Returns a deep copy of this VectorField2D.
        The underlying NumPy arrays are duplicated to avoid shared memory.
        """
        if self.vector_field is None:
            return VectorField2D(None)

        # Deep copy of the NumPy arrays
        copied_field: np.ndarray[float] = np.array([np.copy(comp) for comp in self.vector_field], dtype=float)
        return VectorField2D(copied_field)
