import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np
from battery_problem import BatteryProblem
import seaborn as sns

def vis_mesh(problem: BatteryProblem):
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    cfv.draw_mesh(problem.coords, problem.edof, 1, 2)
    cfv.show_and_wait()


def vis_temp(a_stat, problem: BatteryProblem):
    """Visualizes the static temperature distribution"""

    print(f"Max temp {np.amax(a_stat):.2f} (°C), min temp {np.amin(a_stat):.2f} (°C)")

    cfv.figure(fig_size=(10, 10))
    # Mirror in the x and y plane
    draw_nodal_values_shaded(a_stat, problem.coords, problem.edof,
                             dofs_per_node=1, el_type=2, draw_elements=True)
    draw_nodal_values_shaded(a_stat, [2*problem.L, 0]+[-1, 1]*problem.coords, problem.edof, title=(f"Max temp {np.amax(a_stat):.2f} °C, min temp {np.amin(a_stat):.2f} °C"),
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


def vis_transient(snapshots, snapshot_time, problem: BatteryProblem, deviations, deviation_time):
    """Visualizes the transient heat distribution by showing six snapshots
    snapshot_time: timestamp for each snapshot
    """

    total_time = snapshot_time[-1]

    time_step = (total_time / 6)

    time_step_index = 0

    vis_snapshots = []
    vis_snapshot_time = []
    for snapshot, timestamp in zip(snapshots, snapshot_time):
        if timestamp >= time_step_index * time_step:
            time_step_index += 1
            vis_snapshots.append(snapshot)
            vis_snapshot_time.append(timestamp)

        if time_step_index == 6:
            break

    fig, axes = plt.subplots(3, 2)
    fig.tight_layout()

    vmax = np.max(vis_snapshots[-1])
    for snapshot, time, ax in zip(vis_snapshots, vis_snapshot_time, axes.flatten()):
        plt.sca(ax)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Mirror in x and y plane
        draw_nodal_values_shaded(snapshot, problem.coords, problem.edof,
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=problem.T_0, vmax=vmax)
        draw_nodal_values_shaded(snapshot, [2*problem.L, 0]+[-1, 1]*problem.coords, problem.edof, title=(f"t={time:.2f}s, max temp {np.amax(snapshot):.2f} °C, min temp {np.amin(snapshot):.2f}°C"),
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=problem.T_0, vmax=vmax)

    fig.subplots_adjust(right=0.8)
    plt.colorbar(ax=axes.ravel().tolist())

    for c in cfv.gca().collections:
        if c.colorbar:
            c.colorbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    cfv.show_and_wait()

    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    plt.plot(deviation_time, deviations, label="Actual", color='tomato')
    plt.title("Largest deviation from the ambient temperature over time", fontsize=18, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Temperature (°C)', fontsize=18)
    plt.show()

