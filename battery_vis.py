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


def vis_transient(snapshots, snapshot_time, problem: BatteryProblem, deviations, deviation_time, min_max):
    """Visualizes the transient heat distribution by showing six snapshots
    snapshot_time: timestamp for each snapshot
    """

    total_time = snapshot_time[-1]

    time_step = (900 / 6)

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

    vmax = np.max(vis_snapshots)
    vmin = np.min(vis_snapshots)
    for snapshot, time, ax in zip(vis_snapshots, vis_snapshot_time, axes.flatten()):
        plt.sca(ax)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Mirror in x and y plane
        draw_nodal_values_shaded(snapshot, problem.coords, problem.edof,
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=vmin, vmax=vmax)
        draw_nodal_values_shaded(snapshot, [2*problem.L, 0]+[-1, 1]*problem.coords, problem.edof, title=(f"t={time:.2f}s, max temp {np.amax(snapshot):.2f} °C, min temp {np.amin(snapshot):.2f}°C"),
                                 dofs_per_node=1, el_type=2, draw_elements=False, vmin=vmin, vmax=vmax)

    fig.subplots_adjust(right=0.8)
    plt.colorbar(ax=axes.ravel().tolist())

    for c in cfv.gca().collections:
        if c.colorbar:
            c.colorbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    cfv.show_and_wait()

    plt.figure(figsize=(15, 9))
    plt.plot(deviation_time, deviations, color='tomato')
    plt.title(f"Largest deviation: {np.amax(np.abs(deviations))} (°C) at time: "
              f"{deviation_time[list(np.abs(deviations)).index(np.amax(np.abs(deviations)))]} (s)", fontsize=18, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Temperature (°C)', fontsize=18)
    plt.show()

    plt.figure(figsize=(15, 9))
    plt.plot(deviation_time, min_max[:, 0], label="Min_temp", color='tomato')
    plt.plot(deviation_time, min_max[:, 1], label="Max_temp", color='royalblue')
    plt.title("Test")
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Temperature (°C)', fontsize=18)
    plt.legend()
    plt.show()


def vis_displacement(von_mises_node, a, problem: BatteryProblem):
    magnification = 25
    cfv.figure(fig_size=(10, 10))

    #flip_y = np.array([([1, 1]*int(a.size/2))]).T
    #flip_x = np.array([([-1, 1]*int(a.size/2))]).T

    coords_list = [problem.coords]
        #, [2*problem.L, 0]+[-1, 1]*problem.coords]
    displacement_list = [a]
    #, np.multiply(flip_y*flip_x, a), np.multiply(flip_x, a)]
    for coords, displacements in zip(coords_list, displacement_list):
        if displacements is not None:
            if displacements.shape[1] != coords.shape[1]:
                displacements = np.reshape(
                    displacements, (-1, coords.shape[1]))
                coords_disp = np.asarray(
                    coords + magnification * displacements)
        cfv.draw_mesh(coords, problem.edof, 1,
                      problem.el_type, color=(0, 0, 0, 0.1))
        draw_nodal_values_shaded(von_mises_node, coords_disp, problem.edof, title=(f"Maximum von Mises stress {np.amax(von_mises_node):.1E} [MPa]"),
                                 dofs_per_node=1, el_type=2, draw_elements=False)

        cfv.draw_mesh(coords_disp, problem.edof, 1,
                      problem.el_type, color=(0, 1, 0, 0.1))

    cfv.colorbar()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    for c in cfv.gca().collections:
        if c.colorbar:
            c.colorbar.set_label('von Mises stress (Pa)',
                                 rotation=270, labelpad=20)
    cfv.show_and_wait()
