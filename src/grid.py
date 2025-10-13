from __future__ import annotations
from typing import Optional
import numpy as np

from PolygonalPath2D import PolygonalPath2D  # for Grid.clip_line()
from Point2D import Point2D


# todo: modularizar; isolar grid dessas outras subclasses

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
    barycentricCords: tuple[float, float, float]  # counter clock wise order


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

        summand[self.index] += v1
        summand[self.index + 1] += v2

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

    def __init__(self):
        # todo: implement
        """

                CurveDescription Grid::curve_description(const PolygonalPath &path) const
        {
            CurveDescription result;
            Vector &right_hand_side_x = result.rhsx;
            Vector &right_hand_side_y = result.rhsy;
            result.length = 0;
            int numberOfPoints = path.numberOfPoints();
            if(numberOfPoints < 2){
                return result;
            }

            right_hand_side_x = Vector(2*(numberOfPoints-1));
            right_hand_side_y = Vector(2*(numberOfPoints-1));

            for(int i = 0 ; i < numberOfPoints - 1 ; ++i){
                Vector2D current = toGrid(path.getPoint(i).first);
                Vector2D next = toGrid(path.getPoint(i+1).first);
                Vector2D midpoint = (current + next) * 0.5;
                Vector2D desired_tangent = path.getTangent(i);
                PointLocation location;
            //initialize to avoid compilation warnings
            location.barycentric_coords[0] = -1;
            location.barycentric_coords[1] = -1;
            location.barycentric_coords[2] = -1;
                //
            location.face = getFaceWherePointLies(midpoint);
                Segment segment;
                segment.endpoint[0] = location;
                segment.endpoint[1] = location;
                locate_point(segment.endpoint[0], current);
                locate_point(segment.endpoint[1], next);
                segment.time[0] = path.getPoint(i).second;
                segment.time[1] = path.getPoint(i+1).second;
                segment.index = 2 * i;
                result.length += 2(segment.time[1] - segment.time[0]);
                result.segments.push_back(segment);
                right_hand_side_x[2*i]   = desired_tangent.X();
                right_hand_side_x[2*i+1] = desired_tangent.X();
                right_hand_side_y[2*i]   = desired_tangent.Y();
                right_hand_side_y[2*i+1] = desired_tangent.Y();
            }
            return result;
        }

        """

    def add_ctcx(self, result_x: np.ndarray, x: np.ndarray, k_global: float = 1):
        """
        chains Segment.add_cx and Segment.add_cTx for each segment in this curve
        """

        v = np.zeros(2 * len(self.segments))  # the "summand" parameter for add_cx and add_cTx methods

        for segment in self.segments:  # for each segment in curve

            k = k_global * (segment.time[1] - segment.time[
                0])  # segment-specific weight  # the time interval (duration) of the segment scaled by a global constant.

            segment.add_cx(v, x)  # calculate C*x for this segment and store it in 'v'.
            segment.add_cTx(result_x, v,
                            k)  # calculate C^T * (the result from step 1) and add it to the final result vector


class Intersection:
    location: PointLocation

    indexV1: int
    indexV2: int

    lambda_: float  # interpolation factor in (1-alpha) v1 + alpha v2
    dirAtIntersection: tuple[float, float]

    distanceBetweenCurveVertices: float


def sign(n: float) -> int:
    if n == 0:
        return 0
    if n > 0:
        return 1
    else:
        return -1

class Grid:
    #  number of vertices along the axis
    resolutionX: int
    resolutionY: int

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
        self.resolutionX = resolution_x
        self.resolutionY = resolution_y
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        # escalas   (cord. grid) / (cord. mundo real)
        self.delta_x = self.w / (self.resolutionX - 1)
        self.delta_y = self.h / (self.resolutionY - 1)

    def get_vertex_index(self, x: int, y: int) -> int:
        """
            get linear index based on (x, y) coordinate
        """
        return y * self.resolutionX + x

    def get_face_where_point_lies(self, v: np.ndarray[float]) -> TriangularFace:
        """
            REQUIRES POINT IN GRID COORDINATE SYSTEM

            create and return a new `face` object with the respective properties
        """

        if (  # coordinates out of range
                (v[0] < 0 or v[0] > (self.resolutionX - 1.0))
                or
                (v[1] < 0 or v[1] > (self.resolutionY - 1.0))
        ):
            print("BAD POINT")
            exit(1)

        face: TriangularFace = TriangularFace()

        # square cells indices
        square_x: int = int(v[0])
        square_y: int = int(v[1])
        # floating cord inside square cell
        fx: float = v[0] - square_x
        fy: float = v[1] - square_y

        if fx > fy:  # bottom triangle
            face.indices = (  # counter clock wise order
                self.get_vertex_index(square_x, square_y),
                self.get_vertex_index(square_x + 1, square_y),
                self.get_vertex_index(square_x + 1, square_y + 1)
            )

        else:  # top triangle
            face.indices = (  # counter clock wise order
                self.get_vertex_index(square_x, square_y),
                self.get_vertex_index(square_x + 1, square_y + 1),
                self.get_vertex_index(square_x, square_y + 1)
            )

        return face

    # resolution getters
    def get_resolution_x(self) -> int:
        return self.resolutionX

    def get_resolution_y(self) -> int:
        return self.resolutionY

    # coordinate converters
    def to_grid(self, world_point: np.ndarray) -> np.ndarray[float]:
        """
        convert geographic ("world") coordinates into grid coordinates
        """
        return np.array([
            (world_point[0] - self.x) / self.w * (self.resolutionX - 1.0),
            (world_point[1] - self.y) / self.h * (self.resolutionY - 1.0)
        ])

    def to_world(self, grid_point: np.ndarray) -> np.ndarray:
        """
        convert grid coordinates into geographic ("world") coordinates

        """
        return np.array([
            grid_point[0] / (self.resolutionX - 1.0) * self.w + self.x,
            grid_point[1] / (self.resolutionY - 1.0) * self.h + self.y
        ])

    def get_grid_vertex(self, index: int) -> np.ndarray:
        """
        take linear index, return correspondent (x, y) coordinates
        """
        return np.array([
            index % self.resolutionX,
            index / self.resolutionX
        ])

    def locate_point(self, point_loc: PointLocation, point: np.ndarray) -> None:
        """
        Calculates the barycentric coordinates of a point within a triangular face.

        If face is fixed, set only barycentric coordinates within face.
        """
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
            print("det == 0!!!")
            exit(1)

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
        point_loc.barycentricCords[0] = alpha
        point_loc.barycentricCords[1] = beta
        point_loc.barycentricCords[2] = gamma

    def multiply_by_laplacian(self, first_component: np.ndarray[float], second_component: np.ndarray[float]) -> None:
        """
        multiply both vector field components by the laplacian

        """

        if (  # check dimensions
                self.resolutionX * self.resolutionY != first_component.size or
                self.resolutionX * self.resolutionY != second_component.size
        ):
            print("Error while multiplying grid by vector. Incompatible dimensions.")
            exit(1)

        number_of_vectors: int = self.resolutionX * self.resolutionY

        newFirstComponent: np.ndarray[float] = np.zeros(number_of_vectors)
        newSecondComponent: np.ndarray[float] = np.zeros(number_of_vectors)

        horizontal_cotangent_weight: float = self.delta_x / self.delta_y
        vertical_cotangent_weight: float = self.delta_y / self.delta_x

        for i in range(
                number_of_vectors):  # number_of_vectors == numberOfVertices in the grid == resolution * resolution

            row = i // self.resolutionX
            col = i % self.resolutionX

            # considering grid constraints
            can_move_left: bool = col > 0
            can_move_down: bool = row > 0
            can_move_right: bool = col < self.resolutionX - 1
            can_move_up: bool = row < self.resolutionY - 1

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
                neigh_index = i - self.resolutionX
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
                neigh_index = i + self.resolutionX
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
            newFirstComponent[i] = accum1 - degree * first_component[i]
            newSecondComponent[i] = accum2 - degree * second_component[i]

        # Update the field
        first_component[:] = newFirstComponent
        second_component[:] = newSecondComponent

    def multiply_by_laplacian2(self, first_component: np.ndarray[float], row_length2: np.ndarray[float]) -> None:
        """
        multiply only one vector field component by the laplacian

        """

        if self.resolutionX * self.resolutionY != first_component.size:
            print("Error while multiplying grid by vector. Incompatible dimensions.")
            exit(1)

        number_of_vectors = self.resolutionX * self.resolutionY

        new_first_component = np.zeros(number_of_vectors)
        row_length2[:] = np.zeros(number_of_vectors)

        horizontal_cotangent_weight = self.delta_x / self.delta_y
        vertical_cotangent_weight = self.delta_y / self.delta_x

        for i in range(number_of_vectors):
            row_length2[i] = 0.0

            row = i // self.resolutionX
            col = i % self.resolutionX

            can_move_left = col > 0
            can_move_down = row > 0
            can_move_right = col < self.resolutionX - 1
            can_move_up = row < self.resolutionY - 1

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
                row_length2[i] += coef * coef
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
                row_length2[i] += coef * coef
                degree += coef

            # DOWN
            if can_move_down:
                neigh_index = i - self.resolutionX
                coef = 0.0

                if can_move_left:
                    coef += horizontal_cotangent_weight
                if can_move_right:
                    coef += horizontal_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                row_length2[i] += coef * coef
                degree += coef

            # UP
            if can_move_up:
                neigh_index = i + self.resolutionX
                coef = 0.0

                if can_move_left:
                    coef += horizontal_cotangent_weight
                if can_move_right:
                    coef += horizontal_cotangent_weight

                coef /= 2.0

                accum1 += coef * first_component[neigh_index]
                row_length2[i] += coef * coef
                degree += coef

            # update component
            new_first_component[i] = accum1 - degree * first_component[i]
            row_length2[i] += degree * degree

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

                p = Grid.Inter()

                p.grid_point = np.interp(g1.grid_point, g2.grid_point, u)
                p.grid_point.y = next_y  # force exact integer coordinate
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
                p.grid_point.y = t - p.grid_point.y

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
    def get_u_from_points(v1: np.ndarray, v2: np.ndarray, u: np.ndarray):
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
                vert = self.clip_against_vertical_lines(inters[-1], h)

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
