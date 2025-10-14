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
            barycentric_cords = np.zeros(3, dtype=float)

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

    def add_cx(self, summand: np.ndarray[float], field: np.ndarray[float]) -> None:
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

