import numpy as np

from battery_problem import BatteryProblem
from battery_vis import vis_mesh, vis_temp, vis_transient, vis_displacement

problem = BatteryProblem(0.03)

vis_mesh(problem)

a_stat, _, _ = problem.solve_static(300000000000)
vis_temp(a_stat, problem)

snapshots, snapshot_time, deviations, deviation_time, min_max = problem.solve_transient()
vis_transient(snapshots, snapshot_time, problem, deviations, deviation_time, min_max)

print(f"Largest deviation: {np.amax(deviations)} (°C) at time: {deviation_time[deviations.index(np.amax(deviations))]} (s)")

von_mises, displacement = problem.solve_von_mises()
vis_displacement(von_mises, displacement, problem)