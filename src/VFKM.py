""""""

import numpy as np

from Grid import Grid, CurveDescription



class ProblemSettings:
    """
    struct carrying problem data for solvers.
    """

    grid: Grid
    curve_indices: np.ndarray[int]  # respective to cluster
    curve_descriptions: np.ndarray[CurveDescription]
    total_curve_length: float  # summed length of all curves (used for normalization)
    smoothness_weight: float

    def __init__(
            self,
            grid: Grid,
            curve_indices: np.ndarray[int],
            curve_descriptions: np.ndarray[CurveDescription],
            total_curve_length: float,
            smoothness_weight: float
    ):
        self.grid = grid
        self.curve_indices = curve_indices
        self.curve_descriptions = curve_descriptions
        self.total_curve_length = total_curve_length
        self.smoothness_weight = smoothness_weight



class VFKM:

    def __init__(self, size: int):
        """
        initialize new VF axis
        """

        ax = np.zeros(shape=size)
        ay = np.zeros(shape=size)

def compute_error_implicit(
        grid: Grid,
        vf_x_component: np.ndarray[float],
        vf_y_component: np.ndarray[float],
        total_curve_length: float,
        smoothness_weight: float,
        curve: CurveDescription
):
    pass

"""
double computeErrorImplicit
(const Grid &,
 const Vector &vfXComponent, const Vector& vfYComponent,
 float totalCurveLength,
 float smoothnessWeight,
 const CurveDescription &curve)
{
    double error = 0.0;

    Vector vx(2*curve.segments.size());
    Vector vy(2*curve.segments.size());
    
    for (size_t i = 0; i<curve.segments.size(); ++i) {
        curve.segments[i].add_cx(vx, vfXComponent);
        curve.segments[i].add_cx(vy, vfYComponent);
    }

    vx -= curve.rhsx;
    vy -= curve.rhsy;

    // LT . L = [[1/3 1/6] [1/6 1/3]]
    for (int i=0; i<vx.getDimension(); i+=2) {
        double this_error_x = (vx[i] * vx[i] + vx[i] * vx[i+1] + vx[i+1] * vx[i+1]) / 3.0;
        double this_error_y = (vy[i] * vy[i] + vy[i] * vy[i+1] + vy[i+1] * vy[i+1]) / 3.0;
        error += (this_error_x + this_error_y) * curve.length;
    }

    assert(error >= 0.0);
    return error * (1.0 - smoothnessWeight) / (totalCurveLength);
}
"""