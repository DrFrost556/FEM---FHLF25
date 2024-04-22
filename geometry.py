import calfem.geometry as cfg
import calfem.mesh as cfm

def battery_geometry():
    # define parameters for the geometry
    R = 0.25  # Radius of circles
    M = 1  # Y-axis for middle of circles
    B = 3  # Length of bottom part
    L = 4  # Length of top part
    H = 2  # Height of battery
    D = 0.875  # Length between middle of circles

    #points
    g = cfg.geometry()

    points = [
        [0, 0],
        [0, M - R],
        [0, M],
        [R, M],
        [0, M + R],  # 0-4
        [0, H],
        [L, H],
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

    NUM = None

    for s in [[0, 1]]