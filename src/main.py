"""
Until this point, just testing the classes
"""
import numpy as np

from Point2D import Point2D
from PolygonalPath2D import PolygonalPath2D


# synthetic trajectory points
trajectory_points = [
	Point2D((np.array([0.0, 0.0]), 0.0)),
	Point2D((np.array([1.0, 2.0]), 1.0)),
	Point2D((np.array([3.0, 3.0]), 2.5)),
	Point2D((np.array([5.0, 1.0]), 3.0))
]

# Create a PolygonalPath instance
my_path = PolygonalPath2D(points=trajectory_points)

# Print the path and its calculated tangents
print("\n PATH \n")
print(my_path)

print("\n TANGENTS  \n")
for i, tangent in enumerate(my_path.tangents):
	print(f"Segment {i + 1}: {tangent}")
