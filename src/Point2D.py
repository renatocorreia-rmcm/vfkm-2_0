import numpy as np

Vector = np.ndarray  # in this class, will represent the space coordinates of some point


class Point2D:
    """
    2D space coordinate + float timestamp,

    theoretically is already a ND point since it takes a nd array of coordinates
    """

    point2D: tuple[Vector, float]  # space coordinates, time


    def __init__(
            self,
            point2d: tuple[Vector, float]
    ):
        self.point2D = point2d

    @property
    def space(self) -> Vector:
        """
        Get vector of space coordinates
        for that point

        """
        return self.point2D[0]

    @property
    def time(self) -> float:
        """
        get float of time stamp
        for that point

        """

        return self.point2D[1]

    def __repr__(self):
        return f"({self.space}, {self.time})"

    def __iter__(self):
        return self.point2D.__iter__()