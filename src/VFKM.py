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
) -> float:
    error: float = 0.0

    vx = np.zeros(2 * len(curve.segments), dtype=float)
    vy = np.zeros(2 * len(curve.segments), dtype=float)

    for i in range(len(curve.segments)):
        curve.segments[i].add_cx(vx, vf_x_component)
        curve.segments[i].add_cx(vy, vf_y_component)

    vx -= curve.rhsx
    vy -= curve.rhsy

    # LT . L = [[1/3 1/6] [1/6 1/3]]
    for i in range(0, len(vx), 2):
        this_error_x = (vx[i] * vx[i] + vx[i] * vx[i + 1] + vx[i + 1] * vx[i + 1]) / 3.0
        this_error_y = (vy[i] * vy[i] + vy[i] * vy[i + 1] + vy[i + 1] * vy[i + 1]) / 3.0
        error += (this_error_x + this_error_y) * curve.length

    assert error >= 0.0
    return error * (1.0 - smoothness_weight) / total_curve_length




def optimize_vector_field_with_weights(
        grid: Grid,
        initial_guess_x: np.ndarray[float], initial_guess_y: np.ndarray[float],
        curve_indices: np.ndarray[int],
        curve_descriptions: np.ndarray[CurveDescription],
        total_curve_length: float,
        smoothness_weight: float
) -> None:
    """
    optimize a single vector field (using smoothness)

    Given an initial guess for the vector field components,
    construct the RHS from curve constraints
    and solve two independent linear systems (one per component) using CG.

    The solution overwrites the provided initialGuessX/Y vectors.
    """


    number_of_vertices = grid.get_resolution_x() * grid.get_resolution_y()

    indepx = np.zeros(number_of_vertices, dtype=float)
    indepy = np.zeros(number_of_vertices, dtype=float)

    for k in range(len(curve_indices)):  # for each curve
        i = curve_indices[k]
        curve = curve_descriptions[i]

        for j in range(len(curve.segments)):  # for each segment in curve
            # Sum contributions into the RHS vectors.
            k_factor = (1.0 - smoothness_weight) * (curve.segments[j].time[1] - curve.segments[j].time[0]) / total_curve_length  # weighting factor  # Each segment's influence is weighted by the [ (1 - smoothness_weight) data-term factor ] and [ its relative curve length ].
            curve.segments[j].add_cTx(indepx, curve.rhsx, k_factor)
            curve.segments[j].add_cTx(indepy, curve.rhsy, k_factor)

    problem = ProblemSettings(grid, curve_indices, curve_descriptions, total_curve_length, smoothness_weight)


    # aux vars for calling cg_solve()
    x = initial_guess_x.copy()
    y = initial_guess_y.copy()

    # solve linear system using Conjugate Gradient (cg_solve)
    cg_solve(problem, indepx, x)
    cg_solve(problem, indepy, y)

    initial_guess_x = x.copy()
    initial_guess_y = y.copy()

def cg_solve(
        problem: ProblemSettings,
        b: np.ndarray[float],
        x: np.ndarray[float]
) -> None:
    """
    solve A*x=b without setting up whole matrix

    cg is a iterative solver
    começa com um chute pra x e vai melhorando
    """

    





"""

void optimizeVectorFieldWithWeights(  // optimize a single vector field (using smoothness)
    Grid &grid, Vector &initialGuessX, Vector &initialGuessY,
    const vector<int> &curveIndices,
    const vector<CurveDescription> &curve_descriptions,
    float totalCurveLength,
    float smoothnessWeight
)
{
    
    // optimizeVectorFieldWithWeights: given an initial guess for the vector
    // field components, construct the RHS from curve constraints and solve
    // two independent linear systems (one per component) using CG. The
    // solution overwrites the provided initialGuessX/Y vectors.



    // Compute independent (right-hand side) terms for the linear systems
    // corresponding to the X and Y components of the vector field.
    int numberOfVertices = grid.getResolutionX() * grid.getResolutionY();

    Vector indepx(numberOfVertices), indepy(numberOfVertices);
    indepx.setValues(0.0f);
    indepy.setValues(0.0f);

    // Sum contributions from each curve segment into the RHS vectors.
    // Each segment's influence is weighted by its relative curve length and 
    // the (1 - smoothnessWeight) data-term factor.
    for(size_t k = 0; k < curveIndices.size() ; ++k) {  // for each curve
        int i = curveIndices[k];
        const CurveDescription &curve = curve_descriptions[i];

        for (size_t j=0; j<curve.segments.size(); ++j) {  // for each segment in curve
            float k = (1.0 - smoothnessWeight) * (curve.segments[j].time[1] - curve.segments[j].time[0])/totalCurveLength;  // weighting factor
            curve.segments[j].add_cTx(indepx, curve.rhsx, k);
            curve.segments[j].add_cTx(indepy, curve.rhsy, k);
        }
    }

    ProblemSettings prob(
        grid, curveIndices, curve_descriptions, totalCurveLength, smoothnessWeight
    );

    Vector x(initialGuessX), y(initialGuessY);

    // solve linear system
    cg_solve(prob, indepx, x);
    cg_solve(prob, indepy, y);
    initialGuessX.setValues(x);
    initialGuessY.setValues(y);
}

"""