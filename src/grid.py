from typing import Optional


class TriangularFace:
    """
    3-uple of indices
    """

    indices: tuple[int, int, int]

    def __init__(self, indices: Optional[tuple[int, int, int]] = None):
        self.indices = indices


class PointLocation:
    """
    reference to the point's face object
    3-uple of barycentric coordinates
    """

    face: TriangularFace
    barycentricCords: tuple[float, float, float]


class Segment:

    endpoints: tuple[PointLocation, PointLocation]
    timestamps: tuple[float, float]  # respective to the endpoints

    index: int

    