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

    def add_cx(self, summand, field) -> None:
        """
        Compute vectors of this segment endpoints using its face and its barycentric coordinates (linear interpolation)

        """

        v1: float  # endpoint 1
        v2: float  # endpoint 2

        # v = sum of  <verticeVector_i * barycentricCord_i>  for each vertex of face ([0,1,2])

        # linear interpolation to get vector of endpoint 1
        v1 = \
            field[self.endpoints[0].face.indices[0]] * self.endpoints[0].barycentricCords[0] + \
            field[self.endpoints[0].face.indices[1]] * self.endpoints[0].barycentricCords[1] + \
            field[self.endpoints[0].face.indices[2]] * self.endpoints[0].barycentricCords[2]

        # linear interpolation to get vector of endpoint 2
        v2 = \
            field[self.endpoints[1].face.indices[0]] * self.endpoints[1].barycentricCords[0] + \
            field[self.endpoints[1].face.indices[1]] * self.endpoints[1].barycentricCords[1] + \
            field[self.endpoints[1].face.indices[2]] * self.endpoints[1].barycentricCords[2]

        summand[self.index]   += v1
        summand[self.index+1] += v2


    def add_cTx(self) -> None:
        exit(1)
        #  todo: implement


class CurveDescription:
    # todo: implement

class Intersection:
    # todo: implement

class Grid:
    # todo: implement