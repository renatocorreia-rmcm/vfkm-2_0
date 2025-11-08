from __future__ import annotations  # allow to reference Grid type inside the Grid implementation

from Point2D import Point2D
from VectorField2D import VectorField2D

import numpy as np

from Grid_aux import TriangularFace, PointLocation, Segment
from PolygonalPath2D import PolygonalPath2D  # for Grid.clip_line()


class CurveDescription:  # todo: it contain its index. No need to array of curve indices in functions arguments
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
        self.rhsx = np.array([], dtype=float)  # Initialize as empty
        self.rhsy = np.array([], dtype=float)

        number_of_points: int = path.number_of_points()
        if number_of_points < 2:
            return

        self.rhsx = np.zeros(2 * (number_of_points - 1))
        self.rhsy = np.zeros(2 * (number_of_points - 1))

        for i in range(number_of_points - 1):
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

    def add_cTcx(self, result_x: np.ndarray[float], x: np.ndarray[float], k_global: float = 1):
        """
		chains Segment.add_cx and Segment.add_cTx for each segment in this curve
		"""

        v: np.ndarray[float] = np.zeros(
            2 * len(self.segments))  # the "summand" parameter for add_cx and add_cTx methods

        for segment in self.segments:  # for each segment in curve

            k = k_global * (segment.timestamps[1] - segment.timestamps[
                0])  # segment-specific weight  # the time interval (duration) of the segment scaled by a global constant.

            segment.add_cx(v, x)  # calculate C*x for this segment and store it in 'v'.
            segment.add_cTx(result_x, v,
                            k)  # calculate C^T * (the result from step 1) and add it to the final result vector


"""
    GRID
"""


class Grid:
    #  number of vertices along the axis
    resolution_x: int
    resolution_y: int

    #  bottom left coordinate of grid
    x: float
    y: float

    #  total width and height of grid
    w: float
    h: float

    #  distance between adjacent grid vertices
    delta_x: float
    delta_y: float

    def __init__(self, x: float, y: float, w: float, h: float, resolution_x: int, resolution_y: int):

        # m_* flags member variable (attribute)
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        # escalas   (cord. grid) / (cord. mundo real)
        self.delta_x = self.w / (self.resolution_x - 1)
        self.delta_y = self.h / (self.resolution_y - 1)

    def get_vertex_index(self, x: int, y: int) -> int:
        """
			get linear index based on (x, y) coordinate
		"""
        return y * self.resolution_x + x

    def get_face_where_point_lies(self, v: np.ndarray[float]) -> TriangularFace:
        """
			REQUIRES POINT IN GRID COORDINATE SYSTEM

			create and return a new `face` object with the respective properties
		"""
        if (  # coordinates out of range
                (v[0] < 0 or v[0] > (self.resolution_x - 1.0))
                or
                (v[1] < 0 or v[1] > (self.resolution_y - 1.0))
        ):
            raise ValueError("Point out of grid bounds")

        face: TriangularFace = TriangularFace()

        # square cells indices
        square_x: int = int(v[0])
        square_y: int = int(v[1])
        # floating cord inside square cell
        fx: float = v[0] - square_x
        fy: float = v[1] - square_y

        if fx > fy:  # bottom triangle
            face.indices = np.array([  # counter clock wise order
                self.get_vertex_index(square_x, square_y),
                self.get_vertex_index(square_x + 1, square_y),
                self.get_vertex_index(square_x + 1, square_y + 1)
            ])

        else:  # top triangle
            face.indices = np.array([  # counter clock wise order
                self.get_vertex_index(square_x, square_y),
                self.get_vertex_index(square_x + 1, square_y + 1),
                self.get_vertex_index(square_x, square_y + 1)
            ])

        return face

    # resolution getters
    def get_resolution_x(self) -> int:
        return self.resolution_x

    def get_resolution_y(self) -> int:
        return self.resolution_y

    # coordinate converters
    def to_grid(self, world_point: np.ndarray[float]) -> np.ndarray[float]:
        """
		convert geographic ("world") coordinates into grid coordinates
		"""
        return np.array([
            (world_point[0] - self.x) / self.w * (self.resolution_x - 1.0),
            (world_point[1] - self.y) / self.h * (self.resolution_y - 1.0)
        ])

    def to_world(self, grid_point: np.ndarray[float]) -> np.ndarray[float]:
        """
		convert grid coordinates into geographic ("world") coordinates

		"""
        return np.array([
            grid_point[0] / (self.resolution_x - 1.0) * self.w + self.x,
            grid_point[1] / (self.resolution_y - 1.0) * self.h + self.y
        ])

    def get_grid_vertex(self, index: int) -> np.ndarray[int]:
        """
		take linear index, return correspondent (x, y) coordinates
		"""
        return np.array([
            index % self.resolution_x,
            index // self.resolution_x
        ])

    def locate_point(self, point_loc: PointLocation, point: np.ndarray[float]) -> None:
        """
		Calculates the barycentric coordinates of a point within a triangular face.

		If face is fixed, set only barycentric coordinates within face.
		"""
        # todo: chatgpt says this calculation is wrong. Investigate.

        # Retrieve the three vertices of the triangular face
        vertices: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            self.get_grid_vertex(point_loc.face.indices[0]),
            self.get_grid_vertex(point_loc.face.indices[1]),
            self.get_grid_vertex(point_loc.face.indices[2])
        )

        # det tells triangle area
        det: float = np.linalg.det(
            np.array([
                [vertices[0][0], vertices[1][0]],
                [vertices[0][1], vertices[1][1]]
            ])
        )

        if det == 0:
            raise ValueError("Degenerate triangle with zero determinant in locate_point")

        # each barycentric coordinate is a ratio of the area of a sub-triangle to the area of the main triangle.

        beta = (
                       (vertices[0][0] - vertices[2][0]) * (point[1] - vertices[2][1]) -
                       (vertices[0][1] - vertices[2][1]) * (point[0] - vertices[2][0])
               ) / det

        gamma = (
                        (vertices[1][0] - vertices[0][0]) * (point[1] - vertices[0][1]) -
                        (vertices[1][1] - vertices[0][1]) * (point[0] - vertices[0][0])
                ) / det

        alpha = 1.0 - gamma - beta

        # store the results in the PointLocation object
        point_loc.barycentric_cords[0] = alpha
        point_loc.barycentric_cords[1] = beta
        point_loc.barycentric_cords[2] = gamma

    def multiply_by_laplacian(self, vector_field: VectorField2D) -> None:
        """
		multiply both vector field components by the laplacian

		"""

        first_component: np.ndarray[float] = vector_field[0]
        second_component: np.ndarray[float] = vector_field[1]

        if (  # check dimensions
                self.resolution_x * self.resolution_y != first_component.size or
                self.resolution_x * self.resolution_y != second_component.size
        ):
            print("Error while multiplying grid by vector. Incompatible dimensions.")
            exit(1)

        number_of_vectors: int = self.resolution_x * self.resolution_y

        new_first_component: np.ndarray[float] = np.zeros(number_of_vectors)
        new_second_component: np.ndarray[float] = np.zeros(number_of_vectors)

        horizontal_cotangent_weight: float = self.delta_x / self.delta_y
        vertical_cotangent_weight: float = self.delta_y / self.delta_x

        for i in range(
                number_of_vectors):  # number_of_vectors == numberOfVertices in the grid == resolution * resolution

            row = i // self.resolution_x
            col = i % self.resolution_x

            # considering grid constraints
            can_move_left: bool = col > 0
            can_move_down: bool = row > 0
            can_move_right: bool = col < self.resolution_x - 1
            can_move_up: bool = row < self.resolution_y - 1

            degree = 0.0
            accum1 = 0.0
            accum2 = 0.0

            # LEFT
            if can_move_left:
                neigh_index = i - 1
                coef = 0.0

                if can_move_up:
                    coef += vertical_cotangent_weight
                if can_move_down:
                    coef += vertical_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                accum2 += coef * second_component[neigh_index]
                degree += coef

            # RIGHT
            if can_move_right:
                neigh_index = i + 1
                coef = 0.0

                if can_move_down:
                    coef += vertical_cotangent_weight
                if can_move_up:
                    coef += vertical_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                accum2 += coef * second_component[neigh_index]
                degree += coef

            # DOWN
            if can_move_down:
                neigh_index = i - self.resolution_x
                coef = 0.0

                if can_move_left:
                    coef += horizontal_cotangent_weight
                if can_move_right:
                    coef += horizontal_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                accum2 += coef * second_component[neigh_index]
                degree += coef

            # UP
            if can_move_up:
                neigh_index = i + self.resolution_x
                coef = 0.0

                if can_move_left:
                    coef += horizontal_cotangent_weight
                if can_move_right:
                    coef += horizontal_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                accum2 += coef * second_component[neigh_index]
                degree += coef

            # update new components
            new_first_component[i] = accum1 - degree * first_component[i]
            new_second_component[i] = accum2 - degree * second_component[i]

        # Update the field
        first_component[:] = new_first_component
        second_component[:] = new_second_component

    def multiply_by_laplacian2(self, first_component: np.ndarray[float]) -> None:
        """
		multiply only one vector field component by the laplacian

		"""

        if self.resolution_x * self.resolution_y != first_component.size:
            print("Error while multiplying grid by vector. Incompatible dimensions.")
            exit(1)

        number_of_vectors = self.resolution_x * self.resolution_y

        new_first_component = np.zeros(number_of_vectors)

        horizontal_cotangent_weight = self.delta_x / self.delta_y
        vertical_cotangent_weight = self.delta_y / self.delta_x

        for i in range(number_of_vectors):

            row = i // self.resolution_x
            col = i % self.resolution_x

            can_move_left = col > 0
            can_move_down = row > 0
            can_move_right = col < self.resolution_x - 1
            can_move_up = row < self.resolution_y - 1

            degree = 0.0
            accum1 = 0.0

            # LEFT
            if can_move_left:
                neigh_index = i - 1
                coef = 0.0

                if can_move_up:
                    coef += vertical_cotangent_weight
                if can_move_down:
                    coef += vertical_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                degree += coef

            # RIGHT
            if can_move_right:
                neigh_index = i + 1
                coef = 0.0

                if can_move_down:
                    coef += vertical_cotangent_weight
                if can_move_up:
                    coef += vertical_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                degree += coef

            # DOWN
            if can_move_down:
                neigh_index = i - self.resolution_x
                coef = 0.0

                if can_move_left:
                    coef += horizontal_cotangent_weight
                if can_move_right:
                    coef += horizontal_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                degree += coef

            # UP
            if can_move_up:
                neigh_index = i + self.resolution_x
                coef = 0.0

                if can_move_left:
                    coef += horizontal_cotangent_weight
                if can_move_right:
                    coef += horizontal_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                degree += coef

            # update component
            new_first_component[i] = accum1 - degree * first_component[i]

        # update original field in place
        first_component[:] = new_first_component

    class Inter:
        """
		Intersection point.

		"""

        grid_point: np.ndarray[float]  # grid coordinate (world coordinate normalized to grid)
        u: float  # barycentric coordinate along segment

        kind: int  # "enum-like" constants inside the class
        Vertical = 1
        Horizontal = 2
        Diagonal = 3
        EndPoint = 4

    def clip_against_horizontal_lines(self, g1: Grid.Inter, g2: Grid.Inter) -> list[Grid.Inter]:
        """
		Take endpoints of a segment and return a list of its intersections with horizontal grid lines.
		(For each horizontal line between the endpoints, calculate intersection using linear interpolation.)
		"""

        result = []  # list of intersections

        # segment is perfectly horizontal
        if g1.grid_point[1] == g2.grid_point[1]:
            return result  # parallel lines do not cross

        # inverse of slope = run / rise (safe, since rise != 0)
        inv_slope = (g2.grid_point[0] - g1.grid_point[0]) / (g2.grid_point[1] - g1.grid_point[1])

        # if inv_slope >= 0: normal case
        if inv_slope >= 0:
            y1, y2 = g1.grid_point[1], g2.grid_point[1]

            this_y = y1
            next_y = int(this_y) + 1  # floor(this_y) + 1

            while next_y < y2:
                u = (next_y - y1) / (y2 - y1)

                p: Grid.Inter = Grid.Inter()

                p.grid_point = (1 - u) * g1.grid_point + u * g2.grid_point  # linear interpolation

                p.grid_point[1] = next_y  # force exact integer coordinate
                p.u = u
                p.kind = Grid.Inter.Horizontal

                result.append(p)

                this_y = next_y
                next_y = this_y + 1

            return result

        else:  # inv_slope < 0
            # flip the grid vertically
            t = self.get_resolution_y() - 1

            reverse_g1 = Grid.Inter()
            reverse_g2 = Grid.Inter()

            reverse_g1.grid_point = np.array([g1.grid_point[0], t - g1.grid_point[1]])
            reverse_g2.grid_point = np.array([g2.grid_point[0], t - g2.grid_point[1]])

            reverse_g1.u = g1.u
            reverse_g2.u = g2.u

            # compute with flipped inputs (recursive call)
            reverse_result = self.clip_against_horizontal_lines(reverse_g1, reverse_g2)

            # unflip back
            for p in reverse_result:
                p.grid_point[1] = t - p.grid_point[1]

            return reverse_result

    def clip_against_vertical_lines(self, g1: Grid.Inter, g2: Grid.Inter) -> list[Grid.Inter]:
        """
		Flip coordinates so vertical lines become horizontal,
		solve the problem using clip_against_horizontal_lines,
		then transform back.
		"""

        # flip coordinates (x <-> y)
        flipped_g1 = Grid.Inter()
        flipped_g2 = Grid.Inter()

        flipped_g1.grid_point = np.array([g1.grid_point[1], g1.grid_point[0]])
        flipped_g2.grid_point = np.array([g2.grid_point[1], g2.grid_point[0]])
        flipped_g1.u = g1.u
        flipped_g2.u = g2.u

        # compute flipped result using horizontal clipping
        flipped_result = self.clip_against_horizontal_lines(flipped_g1, flipped_g2)

        # unflip and mark as vertical intersections
        for p in flipped_result:
            p.grid_point = np.array([p.grid_point[1], p.grid_point[0]])
            p.kind = Grid.Inter.Vertical

        return flipped_result

    @staticmethod
    def get_u_from_points(v1: np.ndarray[float], v2: np.ndarray[float], u: np.ndarray[float]) -> float:
        """
		"inverse" of lerp:
		takes endpoints and a midpoint,
		return barycentric coordinate of midpoint
		"""

        if v2[0] != v1[0]:
            return (u[0] - v1[0]) / (v2[0] - v1[0])
        else:
            return (u[1] - v1[1]) / (v2[1] - v1[1])

    def clip_line(self, path1: PolygonalPath2D):
        """
		This performs tessellation of 'path1' by finding all intersection points.

		insert new vertices wherever the path intersects with a grid line

		divide path1 into his segments
		then clipAgainstHorizontalLines and clipAgainstVerticalLines do the actual tesselation
		"""

        current_vertex_index = 0

        while current_vertex_index < path1.number_of_points() - 1:  # "for each vertex (segment)"

            # start point - position and time
            from_pos, from_time = path1.get_point(current_vertex_index)

            # end point - position and time
            to_pos, to_time = path1.get_point(current_vertex_index + 1)

            # segment tangent
            tangent = path1.get_tangent(current_vertex_index)

            inters: list[Grid.Inter] = []

            # endpoints
            e1 = self.Inter()
            e2 = self.Inter()

            # Convert coordinates World -> Grid
            e1.grid_point = self.to_grid(from_pos)
            e2.grid_point = self.to_grid(to_pos)

            # set barycentric cords
            e1.u = 0.0  # start
            e2.u = 1.0  # end

            # set kind
            e1.kind = e2.kind = self.Inter.EndPoint

            # append to intersections list
            inters.append(e1)

            # --- Horizontal intersections ---

            horiz = self.clip_against_horizontal_lines(e1, e2)
            horiz.append(e2)

            for h in horiz:

                # --- Vertical intersections --- between last intersection and h
                vert: list[Grid.Inter] = self.clip_against_vertical_lines(inters[-1], h)

                # Update u for each vertical intersection
                for v in vert:
                    v.u = Grid.get_u_from_points(e1.grid_point, e2.grid_point, v.grid_point)

                inters.extend(vert)
                inters.append(h)

            # --- Resolve diagonal intersections ---
            i = 0
            while i < len(inters) - 1:
                x_square = min(int(inters[i].grid_point[0]), int(inters[i + 1].grid_point[0]))
                y_square = min(int(inters[i].grid_point[1]), int(inters[i + 1].grid_point[1]))

                u1 = inters[i].grid_point[0] - x_square
                v1 = inters[i].grid_point[1] - y_square

                u2 = inters[i + 1].grid_point[0] - x_square
                v2 = inters[i + 1].grid_point[1] - y_square

                du = u2 - u1
                dv = v2 - v1

                s1 = np.sign(u1 - v1)
                s2 = np.sign(u2 - v2)

                # Check if signs differ -> diagonal intersection
                if s1 != s2:
                    # Solve system:
                    # x = y = (v2 * du - u2 * dv) / (du - dv)
                    x = (v2 * du - u2 * dv) / (du - dv)
                    y = x

                    new_inter = self.Inter()
                    new_inter.grid_point = np.array([x_square + x, y_square + y])
                    new_inter.u = self.get_u_from_points(e1.grid_point, e2.grid_point, new_inter.grid_point)
                    new_inter.kind = self.Inter.Diagonal

                    inters.insert(i + 1, new_inter)
                    i += 1  # skip newly inserted point
                i += 1

            # --- Insert intersection points into path ---
            for i in range(1, len(inters) - 1):
                world_point = self.to_world(inters[i].grid_point)
                time_value = inters[i].u * to_time + (1 - inters[i].u) * from_time

                path1.add_point(current_vertex_index + 1, Point2D((world_point, time_value)), tangent)

                current_vertex_index += 1

            current_vertex_index += 1
