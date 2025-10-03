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

	def add_cTcx(self, result_x: np.ndarray, x: np.ndarray, k_global: float = 1):
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

	def getVertexIndex(self, x: int, y: int) -> int:
		"""
			get linear index based on (x, y) coordinate
		"""
		return y * self.resolutionX + x

	def getFaceWherePointLies(self, v: np.ndarray[float]) -> TriangularFace:
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
				self.getVertexIndex(square_x, square_y),
				self.getVertexIndex(square_x + 1, square_y),
				self.getVertexIndex(square_x + 1, square_y + 1)
			)

		else:  # top triangle
			face.indices = (  # counter clock wise order
				self.getVertexIndex(square_x, square_y),
				self.getVertexIndex(square_x + 1, square_y + 1),
				self.getVertexIndex(square_x, square_y + 1)
			)

		return face

	# resolution getters
	def getResolutionX(self) -> int:
		return self.resolutionX

	def getResolutionY(self) -> int:
		return self.resolutionY

	# coordinate converters
	def toGrid(self, world_point: np.ndarray) -> np.ndarray:
		"""
		convert geographic ("world") coordinates into grid coordinates

		"""
		return np.array([
			(world_point[0] - self.x) / self.w * (self.resolutionX - 1.0),
			(world_point[1] - self.y) / self.h * (self.resolutionY - 1.0)
		])

	def toWorld(self, grid_point: np.ndarray) -> np.ndarray:
		"""
		convert grid coordinates into geographic ("world") coordinates

		"""
		return np.array([
			grid_point[0] / (self.resolutionX - 1.0) * self.w + self.x,
			grid_point[1] / (self.resolutionY - 1.0) * self.h + self.y
		])

	def getGridVertex(self, index: int) -> np.ndarray:
		"""
		take linear index, return correspondent (x, y) coordinates
		"""
		return np.array([
			index % self.resolutionX,
			index / self.resolutionX
		])

	def locate_point(self, point_loc: PointLocation, point):
		"""
		If face is fixed, set only barycentric coordinates within face.

		:param point_loc:
		:param point:
		:return:
		"""

		vertices: tuple[np.ndarray, np.ndarray, np.ndarray] = (
			self.getGridVertex(point_loc.face.indices[0]),
			self.getGridVertex(point_loc.face.indices[1]),
			self.getGridVertex(point_loc.face.indices[2])
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

		# use cramers rule to calculate barycentric coordinates



	"""
	void Grid::locate_point(PointLocation &l, const Vector2D& point) const
	{
		Vector2D vertex1 = getGridVertex(l.face.indexV1);
		Vector2D vertex2 = getGridVertex(l.face.indexV2);
		Vector2D vertex3 = getGridVertex(l.face.indexV3);
	
		float det =
			(vertex2.X() - vertex1.X()) * (vertex3.Y() - vertex1.Y()) - 
			(vertex2.Y() - vertex1.Y()) * (vertex3.X() - vertex1.X());
	
		if(det == 0){
			cout << "det == 0!!!!" << endl;
			exit(1);
		}
	
		float beta  = ((vertex1.X() - vertex3.X()) * (point.Y() - vertex3.Y()) - (vertex1.Y() - vertex3.Y()) * (point.X() - vertex3.X()))/det;
		float gamma = ((vertex2.X() - vertex1.X()) * (point.Y() - vertex1.Y()) - (vertex2.Y() - vertex1.Y()) * (point.X() - vertex1.X()))/det;
		float alpha = 1.0 - gamma - beta;
		l.barycentric_coords[0] = alpha;
		l.barycentric_coords[1] = beta;
		l.barycentric_coords[2] = gamma;
	}
	"""
