from __future__ import annotations  # allow to reference Grid type inside the Grid implementation

from typing import Optional

import numpy as np



"""  DATA CLASSES

(store its data directly, without computing anything)

optional arguments on constructor (so they can be assigned after creation)
"""


class TriangularFace:
	"""
    3-uple of indices

    """

	indices: np.ndarray[int]

	def __init__(self, indices: Optional[np.ndarray[int]] = None):
		if indices is None:
			indices = np.zeros(3, dtype=int)

		self.indices = indices


