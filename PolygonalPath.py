import numpy as np

type Point = tuple[np.ndarray[float], float]  # a point is a coordinate in space and time

class PolygonalPath:
    points: np.ndarray[Point]
    tangents: np.ndarray[np.ndarray[float]]  # each point in Points has its velocity vector

    def __init__(self, dimensions: int, points: np.ndarray[Point] = None, tangents: np.ndarray[np.ndarray[float]] = None):
        if points == tangents == None:
            self.points = np.ndarray[Point](shape=dimensions)
        elif tangents == None:
            pass  # todo: implement
        else:
            pass  # todo: implement
