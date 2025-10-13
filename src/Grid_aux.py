from __future__ import annotations  # allow to reference Grid type inside the Grid implementation

from Point2D import Point2D

from typing import Optional

import numpy as np

from PolygonalPath2D import PolygonalPath2D  # for CurveDescription.__init__()

from grid import Grid


"""  DATA CLASSES

(store its data directly, without computing anything)

optional arguments on constructor (so they can be assigned after creation)
"""

class TriangularFace:
    """
    3-uple of indices

    used only in Grid.get_face_where_point_lies()
    """

    indices: np.ndarray[int]

    def __init__(self, indices: Optional[np.ndarray[int]] = None):

        if indices is None:
            indices = np.zeros(3, dtype=int)

        self.indices = indices


class PointLocation:
    """
    reference to the point's face object
    3-uple of barycentric coordinates
    """

    face: TriangularFace
    barycentric_cords: np.ndarray[float]  # counter clock wise order

    def __init__(self, face: Optional[TriangularFace] = None, barycentric_cords: Optional[np.ndarray[float]] = None):

        if face is None:
            face = TriangularFace()
        if barycentric_cords is None:
            barycentric_cords = np.zeros(3)

        self.face = face
        self.barycentric_cords = barycentric_cords


class Segment:
    """
    2 endpoints contained inside a single face
    """

    endpoints: np.ndarray[PointLocation]
    timestamps: np.ndarray[float]  # respective to the endpoints

    index: int

    def __init__(self, endpoints: Optional[np.ndarray[PointLocation]] = None, timestamps: Optional[np.ndarray[float]] = None):

        if endpoints is None:
            endpoints = np.array([PointLocation(), PointLocation()])
        if timestamps is None:
            timestamps = np.array([0.0, 00.0])

        self.endpoints = endpoints
        self.timestamps = timestamps

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
            field[self.endpoints[0].face.indices[0]] * self.endpoints[0].barycentric_cords[0] + \
            field[self.endpoints[0].face.indices[1]] * self.endpoints[0].barycentric_cords[1] + \
            field[self.endpoints[0].face.indices[2]] * self.endpoints[0].barycentric_cords[2]

        # linear interpolation to get vector of endpoint 2
        v2 = \
            field[self.endpoints[1].face.indices[0]] * self.endpoints[1].barycentric_cords[0] + \
            field[self.endpoints[1].face.indices[1]] * self.endpoints[1].barycentric_cords[1] + \
            field[self.endpoints[1].face.indices[2]] * self.endpoints[1].barycentric_cords[2]

        summand[self.index] += v1
        summand[self.index + 1] += v2

    def add_ctx(self, resulting_field: np.ndarray[float], v: np.ndarray[float], w: float=1.0) -> None:
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

        v1 = v[self.index]
        v2 = v[self.index + 1]

        resulting_field[self.endpoints[0].face.indices[0]] += w * v1 * self.endpoints[0].barycentric_cords[0] / 3.0
        resulting_field[self.endpoints[0].face.indices[1]] += w * v1 * self.endpoints[0].barycentric_cords[1] / 3.0
        resulting_field[self.endpoints[0].face.indices[2]] += w * v1 * self.endpoints[0].barycentric_cords[2] / 3.0

        resulting_field[self.endpoints[0].face.indices[0]] += w * v2 * self.endpoints[0].barycentric_cords[0] / 6.0
        resulting_field[self.endpoints[0].face.indices[1]] += w * v2 * self.endpoints[0].barycentric_cords[1] / 6.0
        resulting_field[self.endpoints[0].face.indices[2]] += w * v2 * self.endpoints[0].barycentric_cords[2] / 6.0

        resulting_field[self.endpoints[1].face.indices[0]] += w * v1 * self.endpoints[1].barycentric_cords[0] / 6.0
        resulting_field[self.endpoints[1].face.indices[1]] += w * v1 * self.endpoints[1].barycentric_cords[1] / 6.0
        resulting_field[self.endpoints[1].face.indices[2]] += w * v1 * self.endpoints[1].barycentric_cords[2] / 6.0

        resulting_field[self.endpoints[1].face.indices[0]] += w * v2 * self.endpoints[1].barycentric_cords[0] / 3.0
        resulting_field[self.endpoints[1].face.indices[1]] += w * v2 * self.endpoints[1].barycentric_cords[1] / 3.0
        resulting_field[self.endpoints[1].face.indices[2]] += w * v2 * self.endpoints[1].barycentric_cords[2] / 3.0



"""  NON DATA CLASSES

objects are computed, so arguments must be passed on construction
"""

class CurveDescription:
    """
    Array of Segments

    """

    segments: list[Segment]
    index: int
    length: float
    # right hand side vectors
    rhsx: np.ndarray[float]
    rhsy: np.ndarray[float]

    def __init__(self, path: PolygonalPath2D, grid: Grid):
        """
        Create a CurveDescription object (array of segments)  # notice how a segment is relative to its grid
        from a PolygonalPath one (sequence of timestamped 2D points)
        with respect to the given grid object parameters

        """

        self.segments = []
        self.length = 0
        self.rhsx = np.array([])  # Initialize as empty
        self.rhsy = np.array([])

        number_of_points: int = path.number_of_points()
        if number_of_points < 2:
            return


        self.rhsx = np.zeros(2 * (number_of_points - 1))
        self.rhsy = np.zeros(2 * (number_of_points - 1))


        for i in range(number_of_points-1):

            current_point: np.ndarray[float] = grid.to_grid(path.get_point(i).space)
            next_point: np.ndarray[float] = grid.to_grid(path.get_point(i + 1).space)
            mid_point: np.ndarray[float] = (current_point + next_point) * 0.5

            desired_tangent: np.ndarray[float] = path.get_tangent(i)

            location: PointLocation = PointLocation()

            location.barycentric_cords[0] = -1
            location.barycentric_cords[1] = -1
            location.barycentric_cords[2] = -1

            location.face = grid.get_face_where_point_lies(mid_point)


            segment: Segment = Segment()

            segment.endpoints[0] = location
            segment.endpoints[1] = location

            grid.locate_point(segment.endpoints[0], current_point)
            grid.locate_point(segment.endpoints[1], next_point)

            segment.timestamps[0] = path.get_point(i).time
            segment.timestamps[1] = path.get_point(i + 1).time

            segment.index = 2 * i

            self.length += (segment.timestamps[1] - segment.timestamps[0])
            self.segments.append(segment)

            self.rhsx[2 * i] = desired_tangent[0]
            self.rhsx[2 * i + 1] = desired_tangent[0]
            self.rhsy[2 * i] = desired_tangent[1]
            self.rhsy[2 * i + 1] = desired_tangent[1]

    def add_ctcx(self, result_x: np.ndarray[float], x: np.ndarray, k_global: float = 1):
        """
        chains Segment.add_cx and Segment.add_cTx for each segment in this curve
        """

        v: np.ndarray[float] = np.zeros(2 * len(self.segments))  # the "summand" parameter for add_cx and add_cTx methods

        for segment in self.segments:  # for each segment in curve

            k = k_global * (segment.timestamps[1] - segment.timestamps[0])  # segment-specific weight  # the time interval (duration) of the segment scaled by a global constant.

            segment.add_cx(v, x)  # calculate C*x for this segment and store it in 'v'.
            segment.add_ctx(result_x, v,k)  # calculate C^T * (the result from step 1) and add it to the final result vector
