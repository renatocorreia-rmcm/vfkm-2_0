import numpy as np
from typing import Optional, Tuple

class VectorField:
    """
    Set of axis values

    """

    def __init__(self, vector_field: Optional[Tuple[np.ndarray, np.ndarray]]):
        self.vector_field = vector_field

    vector_field: Optional[Tuple[np.ndarray, np.ndarray]]  # a np array for each axis  # calculated during optimization step
