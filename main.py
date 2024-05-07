from battery_problem import BatteryProblem
from battery_vis import vis_mesh, vis_temp, vis_transient

problem = BatteryProblem(0.03)

vis_mesh(problem)

a_stat, _, _ = problem.solve_static(600)
vis_temp(a_stat, problem)

snapshots, snapshot_time, deviations, deviation_time = problem.solve_transient()
vis_transient(snapshots, snapshot_time, problem, deviations, deviation_time)
