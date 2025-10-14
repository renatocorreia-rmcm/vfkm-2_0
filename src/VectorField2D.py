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
