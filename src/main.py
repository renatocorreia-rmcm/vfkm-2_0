"""
Test bottom up
"""

import numpy as np

from Point2D import Point2D

from Grid import Grid

from PolygonalPath2D import PolygonalPath2D

# testing grid

"""
# ./vfkm ../data/trajectories.txt 3 2 0.5 ../result

Grid arguments
0 0 1 1 3 3

"""

# PARAMETER TAKEN FROM VFKM-1_0

grid_arguments = [0, 0, 1, 1, 3, 3]
grid_parameters = [3, 3, 0, 0, 1, 1, 0.5, 0.5]



# CONSTRUCTOR

g = Grid(*grid_arguments)

assert [g.resolution_x, g.resolution_y, g.x, g.y, g.w, g.h, g.delta_x, g.delta_y] == grid_parameters

# CLIPLINE

raw_points = [
	[0.05, 0.6, 0],
	[0.45, 0.65, 0.25],
	[0.75, 0.55, 0.5],
	[0.95, 0.6, 1]
]

points: list[Point2D] = [Point2D((np.array(p[:2]), p[-1])) for p in raw_points]

path = PolygonalPath2D(points)

print("\npath before tesselation")
print(path)

g.clip_line(path)

print("\npath after tesselation")
print(path)
