from typing import Optional
import numpy as np

from PolygonalPath2D import PolygonalPath2D  # for Grid.clip_line()

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


class Grid:
    #  number of vertices along the axis
    resolutionX: int
    resolutionY: int
    #  bottom left of grid
    x: float
    y: float
    #  width and height of grid
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
    def to_grid(self, world_point: np.ndarray) -> np.ndarray:
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

        newFirstComponent: np.ndarray[float]  = np.zeros(number_of_vectors)
        newSecondComponent: np.ndarray[float] = np.zeros(number_of_vectors)

        horizontal_cotangent_weight: float = self.delta_x / self.delta_y
        vertical_cotangent_weight: float = self.delta_y / self.delta_x


        for i in range(number_of_vectors):  # number_of_vectors == numberOfVertices in the grid == resolution * resolution

            row = i // self.resolutionX
            col = i % self.resolutionX

            # considering grid constraints
            can_move_left:  bool = col > 0
            can_move_down:  bool = row > 0
            can_move_right: bool = col < self.resolutionX - 1
            can_move_up:    bool = row < self.resolutionY - 1

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

    def clip_line(self, path1: PolygonalPath2D):
        """
        tesselation - find all points where path intersects

        insert new vertices wherever the path intersects with a grid line

        divide path1 into his segments
        then clipAgainstHorizontalLines and clipAgainstVerticalLines do the actual tesselation
        """

        pass

    """
        void Grid::clipLine(PolygonalPath &path1) const  // tesselation - find all points where path intersects
    {
        /*
            insert new vertices wherever the path intersects with a grid line

            divide path1 into his segments
            then clipAgainstHorizontalLines and clipAgainstVerticalLines do the actual tesselation
        */

        // get segments
        size_t currentVertexIndex = 0;
        while(currentVertexIndex < path1.numberOfPoints() - 1){  // for each segment:
            // position of start and end
            Vector2D from = path1.getPoint(currentVertexIndex).first,
                       to = path1.getPoint(currentVertexIndex+1).first;
            // time of start and end
            float   tfrom = path1.getPoint(currentVertexIndex).second,
                      tto = path1.getPoint(currentVertexIndex+1).second;
            // direction
            Vector2D tangent = path1.getTangent(currentVertexIndex);

            // convert cordinates World->Grid
            vector<Grid::Inter> inters;
            Grid::Inter e1, e2;
            e1.grid_point = toGrid(from);
            e2.grid_point = toGrid(to);
            e1.u = 0;  // 0 = start
            e2.u = 1;  // 1 = end
            e1.kind = e2.kind = Grid::Inter::EndPoint;

            inters.push_back(e1);

            vector<Grid::Inter> horiz = clipAgainstHorizontalLines(e1, e2);
            horiz.push_back(e2);
            for (size_t i=0; i<horiz.size(); ++i) {  // for each horizontal intersection
                vector<Grid::Inter> vert = clipAgainstVerticalLines(inters.back(), horiz[i]);
                // reported barycentric coords are with respect to clipped lines;
                // make a good u here.
                for (size_t j=0; j<vert.size(); ++j) {
                    vert[j].u = get_u_from_points(e1.grid_point, e2.grid_point, vert[j].grid_point);
                }
                copy(vert.begin(), vert.end(), back_inserter(inters));
                inters.push_back(horiz[i]);
            }

            // resolve diagonal intersections which remain
            for (size_t i=0; i<inters.size()-1; ++i) {
                int x_square = min(int(inters[i  ].grid_point.x),
                                   int(inters[i+1].grid_point.x)),
                    y_square = min(int(inters[i  ].grid_point.y),
                                   int(inters[i+1].grid_point.y));
                float
                    u1 = inters[i  ].grid_point.x - x_square,
                    v1 = inters[i  ].grid_point.y - y_square,
                    u2 = inters[i+1].grid_point.x - x_square,
                    v2 = inters[i+1].grid_point.y - y_square;
                float du = u2 - u1, dv = v2 - v1;
                int s1 = sgn(u1 - v1);
                int s2 = sgn(u2 - v2);

                // if sign of x_i - y_i is different than that of
                // x_i+1 - y_i+1 then there's an intersection with the diagonal.

                if (s1 != s2) {
                    // If that's the case, solve the following linear system over point-in-square
                    // coordinates:

                    // x = y (diagonal line)
                    // (y-y2)/(y2-y1) = (x-x2)/(x2-x1)

                    //   or equivalently without divisions:
                    // (y-y2)dx = (x-x2)dy

                    // The solution is x = y = (v2 du - u2 dv) / (du - dv)

                    // if du = dv we wouldn't have arrived here, because their signs would
                    // have been different
                    float x = (v2 * du - u2 * dv) / (du - dv);
                    float y = x;
                    Grid::Inter new_inter;
                    new_inter.grid_point = Vector2D(x_square + x, y_square + y);
                    new_inter.u = get_u_from_points(e1.grid_point, e2.grid_point, new_inter.grid_point);
                    new_inter.kind = Grid::Inter::Diagonal;
                    inters.insert(inters.begin() + (i + 1), new_inter);

                    // Since we just inserted a point, and we don't want
                    // to check diagonal intersections against it, we increment the index variable.
                    ++i;
                }

                // At the end of all this, we insert points 1..end-1, which correspond to
                // new intersections. inters[0] is the origin vertex.
            }
            for (size_t i=1; i<inters.size()-1; ++i) {
                path1.addPoint(++currentVertexIndex,
                               make_pair(toWorld(inters[i].grid_point),
                                         inters[i].u * tto + (1 - inters[i].u) * tfrom),
                               tangent);
            }
            currentVertexIndex++;
        }
    }
    """