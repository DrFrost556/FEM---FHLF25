from battery_problem import BatteryProblem
from battery_vis import vis_mesh, vis_temp

problem = BatteryProblem(0.03)

vis_mesh(problem)

a_stat, _, _ = problem.solve_static()
vis_temp(a_stat, problem)