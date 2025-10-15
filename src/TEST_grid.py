import numpy as np

from Grid import Grid
from Point2D import Point2D
from PolygonalPath2D import PolygonalPath2D


def approx_equal(a, b, tol=1e-9):
    return np.all(np.abs(np.array(a) - np.array(b)) <= tol)


def run_lightweight_tests():
    print("Running lightweight Grid tests")

    # to_grid/to_world
    g = Grid(0.0, 0.0, 10.0, 20.0, 6, 11)
    world = np.array([2.5, 5.0])
    grid_coords = g.to_grid(world)
    back = g.to_world(grid_coords)
    print("to_grid:", grid_coords)
    print("to_world:", back)
    assert approx_equal(world, back, tol=1e-8)

    # vertex index roundtrip
    g2 = Grid(0, 0, 1, 1, 4, 3)
    ix = g2.get_vertex_index(2, 1)
    v = g2.get_grid_vertex(ix)
    print("vertex index:", ix, "->", v)
    assert (v == np.array([2, 1])).all()

    # face lookup sanity
    g3 = Grid(0, 0, 10, 10, 5, 5)
    grid_p = np.array([1.2, 1.7])
    face = g3.get_face_where_point_lies(grid_p)
    print("face indices:", face.indices)
    assert len(face.indices) == 3
    n = g3.resolution_x * g3.resolution_y
    assert all(0 <= int(idx) < n for idx in face.indices)

    # multiply_by_laplacian quick run
    g4 = Grid(0, 0, 2.0, 2.0, 3, 3)
    n = g4.resolution_x * g4.resolution_y
    a = np.arange(n, dtype=float)
    b = np.ones(n, dtype=float) * 2.0
    print("before laplacian a:", a)
    g4.multiply_by_laplacian(a, b)
    print("after laplacian a:", a)
    assert np.isfinite(a).all()

    print("Lightweight tests passed")


def run_additional_tests():
    print('\nRunning additional Grid tests')
    g = Grid(0, 0, 10, 10, 5, 5)

    # locate_point reconstruction using C++ formula
    grid_p = np.array([1.2, 1.3])
    face = g.get_face_where_point_lies(grid_p)
    v0 = g.get_grid_vertex(int(face.indices[0])).astype(float)
    v1 = g.get_grid_vertex(int(face.indices[1])).astype(float)
    v2 = g.get_grid_vertex(int(face.indices[2])).astype(float)
    det = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
    assert abs(det) > 1e-12
    beta = ((v0[0] - v2[0]) * (grid_p[1] - v2[1]) - (v0[1] - v2[1]) * (grid_p[0] - v2[0])) / det
    gamma = ((v1[0] - v0[0]) * (grid_p[1] - v0[1]) - (v1[1] - v0[1]) * (grid_p[0] - v0[0])) / det
    alpha = 1.0 - gamma - beta
    bary = np.array([alpha, beta, gamma])
    recon = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
    print("reference barycentric:", bary, "reconstructed", recon)
    assert np.allclose(recon, grid_p, atol=1e-8)

    # clip_against_horizontal_lines basic
    g_h = Grid(0,0,1,1,6,6)
    e1 = g_h.Inter(); e2 = g_h.Inter()
    e1.grid_point = np.array([0.5, 0.2]); e2.grid_point = np.array([3.5, 4.7])
    e1.u = 0.0; e2.u = 1.0; e1.kind = e2.kind = g_h.Inter.EndPoint
    res = g_h.clip_against_horizontal_lines(e1, e2)
    expected_ys = [1,2,3,4]
    assert len(res) == len(expected_ys)
    for p, ey in zip(res, expected_ys):
        assert abs(p.grid_point[1] - ey) < 1e-9

    # clip_against_vertical_lines basic
    g_v = Grid(0,0,1,1,6,6)
    e1 = g_v.Inter(); e2 = g_v.Inter()
    e1.grid_point = np.array([0.2, 0.5]); e2.grid_point = np.array([4.7, 3.5])
    e1.u = 0.0; e2.u = 1.0; e1.kind = e2.kind = g_v.Inter.EndPoint
    resv = g_v.clip_against_vertical_lines(e1, e2)
    expected_xs = [1,2,3,4]
    assert len(resv) == len(expected_xs)

    # get_u_from_points
    got = Grid.get_u_from_points(np.array([0.0,0.0]), np.array([4.0,0.0]), np.array([1.0,0.0]))
    assert abs(got - 0.25) < 1e-12

    # multiply_by_laplacian2 check
    g_l = Grid(0,0,2.0,2.0,3,3)
    n = g_l.resolution_x * g_l.resolution_y
    a = np.arange(n, dtype=float)
    def ref_lapl2(g, first_component):
        rx = g.resolution_x; ry = g.resolution_y; n = rx*ry
        horizontal = g.delta_x / g.delta_y; vertical = g.delta_y / g.delta_x
        new_first = np.zeros(n); rowlen2 = np.zeros(n)
        for i in range(n):
            rowlen2[i] = 0.0
            row = i // rx; col = i % rx
            can_left = col > 0; can_down = row > 0; can_right = col < rx - 1; can_up = row < ry - 1
            degree = 0.0; accum1 = 0.0
            if can_left:
                neigh = i - 1; coef = 0.0
                if can_up: coef += vertical
                if can_down: coef += vertical
                coef /= 2.0; accum1 += coef * first_component[neigh]; rowlen2[i] += coef*coef; degree += coef
            if can_right:
                neigh = i + 1; coef = 0.0
                if can_down: coef += vertical
                if can_up: coef += vertical
                coef /= 2.0; accum1 += coef * first_component[neigh]; rowlen2[i] += coef*coef; degree += coef
            if can_down:
                neigh = i - rx; coef = 0.0
                if can_left: coef += horizontal
                if can_right: coef += horizontal
                coef /= 2.0; accum1 += coef * first_component[neigh]; rowlen2[i] += coef*coef; degree += coef
            if can_up:
                neigh = i + rx; coef = 0.0
                if can_left: coef += horizontal
                if can_right: coef += horizontal
                coef /= 2.0; accum1 += coef * first_component[neigh]; rowlen2[i] += coef*coef; degree += coef
            new_first[i] = accum1 - degree * first_component[i]
            rowlen2[i] += degree * degree
        return new_first, rowlen2
    expected_new, expected_rowlen = ref_lapl2(g_l, a.copy())
    arr = a.copy(); rowlen = np.zeros(n)
    g_l.multiply_by_laplacian2(arr, rowlen)
    assert np.allclose(arr, expected_new)
    assert np.allclose(rowlen, expected_rowlen)

    print('Additional tests passed')


def run_clip_line_exact_test():
    print('\nRunning exact clip_line comparison')
    ref = [
        (0.05, 0.60000002, 0.0),
        (0.10714287, 0.60714287, 0.03571430),
        (0.44999999, 0.64999998, 0.25),
        (0.75, 0.55000001, 0.5),
        (0.94999999, 0.60000002, 1.0)
    ]
    points = [
        Point2D((np.array([0.05, 0.6]), 0.0)),
        Point2D((np.array([0.45, 0.65]), 0.25)),
        Point2D((np.array([0.75, 0.55]), 0.5)),
        Point2D((np.array([0.95, 0.6]), 1.0)),
    ]
    path = PolygonalPath2D(points)
    g = Grid(0,0,1,1,3,3)
    g.clip_line(path)
    got = [(p.space[0], p.space[1], p.time) for p in path.points]
    assert len(got) == len(ref)
    for (gx, gy, gt), (rx, ry, rt) in zip(got, ref):
        assert np.allclose([gx, gy, gt], [rx, ry, rt], atol=1e-6)
    print('clip_line exact test passed')


if __name__ == '__main__':
    run_lightweight_tests()
    run_additional_tests()
    run_clip_line_exact_test()
