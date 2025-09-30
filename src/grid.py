from typing import Optional
import numpy as np

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
        C * v
        Gathering
        Interpolates the vector field
        from grid vertices to trajectory points
        to calculate the fitting error.

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


    def add_ctx(self, resulting_field, v, w=1.0) -> None:
        """
        C^T * v
        Scattering
        Distributes the influence of trajectory tangents
        back to the grid vertices
        to set up the optimization problem.

        Does the reverse process of add_cx
        For some vector inside a face, decompose it to each vertex

        "At each grid vertex,
        this is the aggregate velocity we'd like the vector field to have,
        based on all the trajectories passing nearby."
        """

        v1 = v[self.index], v2 = v[self.index + 1]

        resulting_field[self.endpoints[0].face.indices[0]] += w * v1 * self.endpoints[0].barycentricCords[0] / 3.0
        resulting_field[self.endpoints[0].face.indices[1]] += w * v1 * self.endpoints[0].barycentricCords[1] / 3.0
        resulting_field[self.endpoints[0].face.indices[2]] += w * v1 * self.endpoints[0].barycentricCords[2] / 3.0

        resulting_field[self.endpoints[0].face.indices[0]] += w * v2 * self.endpoints[0].barycentricCords[0] / 6.0
        resulting_field[self.endpoints[0].face.indices[1]] += w * v2 * self.endpoints[0].barycentricCords[1] / 6.0
        resulting_field[self.endpoints[0].face.indices[2]] += w * v2 * self.endpoints[0].barycentricCords[2] / 6.0

        resulting_field[self.endpoints[1].face.indices[0]] += w * v1 * self.endpoints[1].barycentricCords[0] / 6.0
        resulting_field[self.endpoints[1].face.indices[1]] += w * v1 * self.endpoints[1].barycentricCords[1] / 6.0
        resulting_field[self.endpoints[1].face.indices[2]] += w * v1 * self.endpoints[1].barycentricCords[2] / 6.0

        resulting_field[self.endpoints[1].face.indices[0]] += w * v2 * self.endpoints[1].barycentricCords[0] / 3.0
        resulting_field[self.endpoints[1].face.indices[1]] += w * v2 * self.endpoints[1].barycentricCords[1] / 3.0
        resulting_field[self.endpoints[1].face.indices[2]] += w * v2 * self.endpoints[1].barycentricCords[2] / 3.0


class CurveDescription:
    """
    Array of Segments

    """

    segments: np.ndarray[Segment]
    index: int
    length: float
    # right hand side vectors
    rhsx: np.ndarray
    rhsy: np.ndarray

    def add_cTcx(self, result_x: np.ndarray, x: np.ndarray, k_global:float=1):
        """
        chains Segment.add_cx and Segment.add_cTx for each segment in this curve
        """

        v = np.zeros(2 * len(self.segments))  # the "summand" parameter for add_cx and add_cTx methods

        for segment in self.segments:  # for each segment in curve

            k = k_global * (segment.time[1] - segment.time[0])  # segment-specific weight  # the time interval (duration) of the segment scaled by a global constant.

            segment.add_cx(v, x)  # calculate C*x for this segment and store it in 'v'.
            segment.add_cTx(result_x, v, k)  # calculate C^T * (the result from step 1) and add it to the final result vector



class Intersection:
    location: PointLocation

    indexV1: int
    indexV2: int

    lambda_: float  # interpolation factor in (1-alpha) v1 + alpha v2
    dirAtIntersection: tuple[float, float]

    distanceBetweenCurveVertices: float


class Grid:
    # todo: implement