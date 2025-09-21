import numpy as np
from typing import Optional, Tuple


class VectorField2D:
    """
	consited by a 2-uple that contains:
	a nd array for x-axis,
	a nd array for y-axis

	adapt to VectorFieldND: use a ND array instead of a 2-uple
	"""

    vector_field: Optional[Tuple[np.ndarray, np.ndarray]]  # a np array for each axis  # calculated during optimization step

    def __init__(
            self,
            vector_field: Optional[Tuple[np.ndarray, np.ndarray]]
    ):
        self.vector_field = vector_field
