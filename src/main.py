"""
Test bottom up
"""

import numpy as np

from Point2D import Point2D

from Grid import Grid


# POINT 2D
"""
Point2D([s, t])

space() -> s

time -> t
"""


# POLYGONAL PATH 2D
"""
PolygonalPath2D([points])
PolygonalPath2D([points], [tangents])

number_of_points()

get_point(index)

get_tangent(index)

_calculate_tangents()

add_point(index, new_point, tangent)
"""


# VECTOR FIELD 2D
"""
VectorField2D()
"""


# CLUSTER
"""
cluster(name, parent, vector_field)

clear_children()

add_child(Cluster)
"""

# TRIANGULAR FACE
"""
TriagularFace()
"""

# POINT LOCATION
"""
PointLocation(face, barycentric_cords)
"""

# SEGMENT
"""
Segment(endpoints, timestamps)
add_cx(summand, field)
add_ctx(resulting_field, v, w)
"""

# CURVE DESCRIPTION
"""
CurveDescription(path, grid)
add_ctcx(result_x, x, k_global)
"""

# GRID
"""
"""


# testing grid

"""
# ./vfkm ../data/trajectories.txt 3 2 0.5 ../result

Grid arguments
0 0 1 1 3 3

"""

# PARAMETER TAKEN FROM VFKM-1_0

grid_arguments = [0, 0, 1, 1, 3, 3]
grid_parameters = [3, 3, 0, 0, 1, 1, 0.5, 0.5]


print("testing")

g = Grid(*grid_arguments)

assert g.x == 0.0
assert g.y == 0.0
assert g.w == 1.0
assert g.h == 1.0
assert g.resolution_x == 3
assert g.resolution_y == 3
assert g.delta_x == 0.5
assert g.delta_y == 0.5
