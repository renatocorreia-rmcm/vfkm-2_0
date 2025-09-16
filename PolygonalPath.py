import numpy as np
from typing import List, Tuple, Optional

Vector = np.ndarray  # replace all Vector2D and Vector classes in original implementation


class PolygonalPath:
    """
    Represents a trajectory as a sequence of time-stamped 2D points

    """
    
    
    def __init__(
        self,
        points: List[Tuple[Vector, float]],
        tangents: Optional[List[Vector]] = None
    ):
        """
        Initializes a PolygonalPath.

        Args:
            points: A list of tuples, where each tuple contains 
            a NumPy array (shape 2,) and a float timestamp.

            tangents: An optional list of pre-calculated tangent vectors (NumPy arrays).
            If None, they will be calculated automatically.
        """

        # POINTS
        self.points: List[Tuple[Vector, float]] = points

        # TANGENTS
        if tangents:
            self.tangents: List[Vector] = tangents
        else:
            self.tangents = self._calculate_tangents()


    def _calculate_tangents(self) -> List[Vector]:
        """
        Calculates the tangent vector for each segment of the path.
        (The velocity velocity vector, encodes speed and direction) 

        """

        calculated_tangents = []

        zero_vector = np.array([0.0, 0.0])

        if len(self.points) <= 1:  # path has a single or none point
            calculated_tangents.append(zero_vector)  # theres no velocity w/out displacement

        else:
            for i in range(1, len(self.points)):

                # raw coordinates and time of points
                p1_vec, p1_time = self.points[i-1]
                p2_vec, p2_time = self.points[i]

                # delta of coordinates and time of points
                delta_space = p2_vec - p1_vec
                delta_time = p2_time - p1_time

                if delta_time > 0:
                    tangent = delta_space / delta_time
                    calculated_tangents.append(tangent)

                else:
                    calculated_tangents.append(zero_vector)

        return calculated_tangents
    

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the path.

        """
        
        def format_vec(v: Vector) -> str:  # helper function to format a NumPy array nicely for printing
            """ Algebric notation w/ 2 decimal plate precision """
            return f"({v[0]:.2f}, {v[1]:.2f})"

        path_str = " -> ".join([f"{format_vec(p[0])}@t={p[1]:.1f}" for p in self.points])
        return f"PolygonalPath object:\n[{path_str}]"
    


# --- Example Usage ---
if __name__ == '__main__':

    # synthetic trajectory points
    trajectory_points = [
        (np.array([0.0, 0.0]), 0.0),
        (np.array([1.0, 2.0]), 1.0),
        (np.array([3.0, 3.0]), 2.5),
        (np.array([5.0, 1.0]), 3.0)
    ]

    # Create a PolygonalPath instance
    my_path = PolygonalPath(points=trajectory_points)

    # Print the path and its calculated tangents
    print("\n PATH \n")
    print(my_path)

    print("\n TANGENTS  \n")
    for i, tangent in enumerate(my_path.tangents):
        print(f"Segment {i+1}: {tangent}")