"""

class Visualizer:
visualize curves
visuzalize vector field
visualize cluster

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from Grid import CurveDescription, Grid


def visualize_curves(curves: list[CurveDescription], grid: Grid) -> None:
	"""
	Visualizes a list of curves using matplotlib.

	:param curves: List of CurveDescription objects to visualize.
	"""
	print("\n PLOTTING CURVES \n")
	plt.figure()
	for i, curve in enumerate(curves):
		print(f"curve {curve.index}: \n{curve}")
		x = []
		y = []
		# get coordinate from face indices and barycentric coordinates
		for segment in curve.segments:
			coord: np.ndarray[float] = segment.endpoints[0].get_coordinates(grid=grid)
			print(f"segment: \n{coord}")
			x.append(coord[0])
			y.append(coord[1])

		plt.plot(x, y, label=f'Curve {curve.index}')

	plt.title('Curves Visualization')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.legend()
	plt.grid()
	plt.show()
