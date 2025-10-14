import numpy as np
from typing import Optional

from Point2D import Point2D

Vector = np.ndarray[float]


class PolygonalPath2D:
    """
    Represents a trajectory as a sequence of time-stamped 2D points

    """

    points: list[Point2D]
    tangents: list[Vector]


    def __init__(
            self,
            points: Optional[list[Point2D]] = None,
            tangents: Optional[list[Vector]] = None
    ):
        """
        Args:
            points: A list of PointNDs

            tangents: An optional list of pre-calculated tangent vectors (NumPy arrays).
            If None, they will be calculated automatically.
        """

        if points is None:
            points = []
        if tangents is None:
            tangents = self._calculate_tangents()


        self.points = points
        self.tangents = tangents

    def number_of_points(self) -> int:
        return len(self.points)

    def get_point(self, index: int) -> Point2D:
        return self.points[index]

    def get_tangent(self, index: int) -> Vector:
        return self.tangents[index]

    def _calculate_tangents(self) -> list[Vector]:
        """
        Calculates the tangent vector for each segment of the path.
        (The velocity vector, encodes speed and direction)

        """

        calculated_tangents = []

        zero_vector = np.array([0.0, 0.0])

        if len(self.points) <= 1:  # path has a single or none point
            calculated_tangents.append(zero_vector)  # theres no velocity w/out displacement

        else:
            for j in range(1, len(self.points)):

                # raw coordinates and time of points
                p1_vec, p1_time = self.points[j - 1].point2D
                p2_vec, p2_time = self.points[j].point2D

                # delta of coordinates and time of points
                delta_space = p2_vec - p1_vec
                delta_time = p2_time - p1_time

                if delta_time > 0:
                    tangent = delta_space / delta_time
                    calculated_tangents.append(tangent)

                else:
                    calculated_tangents.append(zero_vector)

        return calculated_tangents

    def add_point(self, index: int, new_point: Point2D, tangent: Vector) -> None:
        self.points.insert(index, new_point)
        self.tangents.insert(index, tangent)