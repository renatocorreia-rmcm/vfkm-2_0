from typing import Optional
import numpy as np

Vector = np.ndarray


class Point2D:
	"""
	2D space coordinate + float timestamp,
	although it takes a space vector as a nd array
	"""

	def __init__(self, point2D: Optional[tuple[Vector, float]] = None):  # could use space and time parameters separately instead of in a 2-uple
		self.point2D = point2D

	point2D: tuple[Vector, float]  # space cooordinates, time

	def space(self) -> Vector:
		"""
		Get vector of space coordinates
		"""
		return self.point2D[0]

	def time(self) -> float:
		"""
		Get float of timestamp
		"""
		return self.point2D[1]
