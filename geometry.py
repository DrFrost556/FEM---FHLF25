import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv


class Material:
    BATTERY = 1


class Boundaries:
    TOP_BATTERY = 6
    COOL_CIRCLE = 8
    WARM_CIRCLE = 10
    FIXED_BOUNDARY = 12
    NONE_BOUNDARY = 14


def battery_geometry(L):
    # define parameters for the geometry
    R = 25*L  # Radius of circles
    M = 100*L  # Y-axis for middle of circles
    B = 300*L  # Length of bottom part
    T = 400*L  # Length of top part
    H = 200*L  # Height of battery
    D = 87.5*L  # Length between middle of circles

    # points
    g = cfg.geometry()

    points = [
        [0, 0],
        [0, M - R],
        [0, M],
        [R, M],
        [0, M + R],  # 0-4
        [0, H],
        [T, H],
        [B, 0],
        [D, M],
        [D - R, M],  # 5-9
        [D, M + R],
        [D + R, M],
        [D, M - R],
        [2 * D, M],
        [2 * D - R, M],  # 10-14
        [2 * D, M + R],
        [2 * D + R, M],
        [2 * D, M - R],
        [3 * D, M],
        [3 * D - R, M],  # 15-19
        [3 * D, M + R],
        [3 * D + R, M],
        [3 * D, M - R],
    ]

    for xp, yp in points:
        g.point([xp, yp])

    num = None

    # define lines / circle segments
    g.spline([0, 1], 0, marker=Boundaries.NONE_BOUNDARY, el_on_curve=num)
    g.circle([1, 2, 3], 1, marker=Boundaries.WARM_CIRCLE, el_on_curve=num)
    g.circle([3, 2, 4], 2, marker=Boundaries.WARM_CIRCLE, el_on_curve=num)
    g.spline([4, 5], 3, marker=Boundaries.NONE_BOUNDARY, el_on_curve=num)
    g.spline([5, 6], 4, marker=Boundaries.TOP_BATTERY, el_on_curve=num)
    g.spline([6, 7], 5, marker=Boundaries.FIXED_BOUNDARY, el_on_curve=num)
    g.spline([7, 0], 6, marker=Boundaries.FIXED_BOUNDARY, el_on_curve=num)

    # circle 1
    g.circle([9, 8, 10], 7, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)
    g.circle([10, 8, 11], 8, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)
    g.circle([11, 8, 12], 9, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)
    g.circle([12, 8, 9], 10, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)

    # circle 2
    g.circle([14, 13, 15], 11, marker=Boundaries.WARM_CIRCLE, el_on_curve=num)
    g.circle([15, 13, 16], 12, marker=Boundaries.WARM_CIRCLE, el_on_curve=num)
    g.circle([16, 13, 17], 13, marker=Boundaries.WARM_CIRCLE, el_on_curve=num)
    g.circle([17, 13, 14], 14, marker=Boundaries.WARM_CIRCLE, el_on_curve=num)

    # circle 3
    g.circle([19, 18, 20], 15, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)
    g.circle([20, 18, 21], 16, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)
    g.circle([21, 18, 22], 17, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)
    g.circle([22, 18, 19], 18, marker=Boundaries.COOL_CIRCLE, el_on_curve=num)

    g.surface([0, 1, 2, 3, 4, 5, 6], [[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]], marker=Material.BATTERY)
    return g


def battery_mesh(L, el_type, dof=1, size_factor=None):
    geometry = battery_geometry(L)
    mesh = cfm.GmshMesh(geometry, el_type, dof)
    mesh.return_boundary_elements = True
    if size_factor is not None:
        mesh.el_size_factor = size_factor
    return mesh.create()
