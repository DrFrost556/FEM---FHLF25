import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np
from battery_problem import BatteryProblem

def vis_mesh(problem: BatteryProblem):
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    cfv.draw_mesh(problem.coords, problem.edof, 1, 2)
    cfv.show_and_wait()


def vis_temp(a_stat, problem: BatteryProblem):
    """Visualizes the static temperature distribution"""

    print(f"Maximum temperature {np.amax(a_stat):.2f} (°C)")

    cfv.figure(fig_size=(10, 10))
    # Mirror in the x and y plane
    draw_nodal_values_shaded(a_stat, problem.coords, problem.edof,
                             dofs_per_node=1, el_type=2, draw_elements=True)
    draw_nodal_values_shaded(a_stat, [2*problem.L, 0]+[-1, 1]*problem.coords, problem.edof, title=(f"Maximum temperature {np.amax(a_stat):.2f} °C"),
                             dofs_per_node=1, el_type=2, draw_elements=True)

    cfv.colorbar()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    # Set the colorbar label
    for c in cfv.gca().collections:
        if c.colorbar:
            c.colorbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
            cfv.show_and_wait()


def draw_nodal_values_shaded(values, coords, edof, title=None, dofs_per_node=None, el_type=None, draw_elements=False, **kwargs):
    """Draws element nodal values as shaded triangles. Element topologies
    supported are triangles, 4-node quads and 8-node quads."""

    edof_tri = cfv.topo_to_tri(edof)

    ax = plt.gca()
    ax.set_aspect('equal')

    x, y = coords.T
    v = np.asarray(values)
    plt.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud", **kwargs)

    if draw_elements:
        if dofs_per_node != None and el_type != None:
            cfv.draw_mesh(coords, edof, dofs_per_node,
                          el_type, color=(0, 1, 0, 0.1))
        else:
            cfv.info(
                "dofs_per_node and el_type must be specified to draw the mesh.")

    if title != None:
        ax.set(title=title)