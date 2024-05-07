import calfem.core as cfc
import numpy as np
from operator import itemgetter
from geometry import battery_mesh, Boundaries, Material
from plantml import plantml
import math

class BatteryProblem:
    def __init__(self, size_factor=None) -> None:
        # Material parameters, found in project description
        self.materials = {
            Material.BATTERY: HomogenousMaterial(80, 5, 0.36, 60e-6, 540, 3600, "Battery")}

        self.T_inf = 20  # [C]
        self.T_0 = 20  # [C]
        self.T_in = 4  # [C]
        self.T_out = 12  # [C]

        # Convection constants
        self.alpha_c = 120
        self.alpha_n = 40

        self.thickness = [1.0]

        self.h = 100  # [Q]

        self.L = 0.001  # [M]

        self.el_type = 2

        self.coords, self.edof, self.dofs, self.bdofs, self.element_markers, self.boundary_elements = battery_mesh(
            self.L, self.el_type, 1, size_factor)

        self.ex, self.ey = cfc.coord_extract(self.edof, self.coords, self.dofs)

        # CALFEM specific shorthand
        self.ep = [2, self.thickness[0]]

    def integrate_boundary_convection(self, node_pair_list, k_sub, alpha):
        # Calculates the integral N^tN over an element edge
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            # Specific for our three pint triangular element
            k_e = np.array([[1/3, 1/6],
                            [1/6, 1/3]]) * \
                alpha * self.thickness[0] * distance

            cfc.assem(np.array([p1, p2]), k_sub, k_e)

    def solve_Q1(self, time):
        Q1 = 100 * math.exp(-144 * ((600 - time) / 3600) ** 2) * 80
        return Q1

    def solve_Q2(self, time):
        var = ((600 - time) / 3600)
        if var > 0:
            Q2 = 88.42 * 1 * 80
        else:
            Q2 = 0
        return Q2

    def boundary_conv_add(self, node_pair_list, f_b, temp, alpha):
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            f_b[np.isin(self.dofs, [p1, p2])] += temp * alpha * distance

    def solve_static(self, time):
        K = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        # Create force vector
        f = np.zeros([np.size(self.dofs), 1])
        for eldof, elx, ely, material_index in zip(self.edof, self.ex, self.ey, self.element_markers):
            if self.solve_Q1(time) == 0:
                Ke = cfc.flw2te(elx, ely, self.thickness, self.materials[material_index].D)
                cfc.assem(eldof, K, Ke)
            else:
                Ke, fe = cfc.flw2te(elx, ely, self.thickness, self.materials[material_index].D, self.solve_Q1(time))
                cfc.assem(eldof, K, Ke, f, fe)

        # Add different f_c
        f_c_top = list(map(itemgetter("node-number-list"), self.boundary_elements[Boundaries.TOP_BATTERY]))
        self.boundary_conv_add(f_c_top, f, self.T_inf, self.alpha_n)

        f_c_warm = list(map(itemgetter("node-number-list"), self.boundary_elements[Boundaries.WARM_CIRCLE]))
        self.boundary_conv_add(f_c_warm, f, self.T_out, self.alpha_c)

        f_c_cool = list(map(itemgetter("node-number-list"), self.boundary_elements[Boundaries.COOL_CIRCLE]))
        self.boundary_conv_add(f_c_cool, f, self.T_in, self.alpha_c)

        # Add different K_c
        K_c = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        self.integrate_boundary_convection(f_c_top, K_c, self.alpha_n)
        self.integrate_boundary_convection(f_c_warm, K_c, self.alpha_c)
        self.integrate_boundary_convection(f_c_cool, K_c, self.alpha_c)
        K += K_c

        a_stat = np.linalg.solve(K, f)

        return a_stat, K, f

    def solve_transient(self):
        # (C + delta_tK)a_n+1 = Ca_n + delta_t*f_n+1, a_0 = T_0
        # start at t = 0
        time = 0

        theta = 1.0

        delta_t = 100
        n = 200
        step = 0

        # a_0 from T_0
        a = np.full(self.dofs.shape, self.T_0)

        snapshot = [a]
        snapshot_time = [time]

        # Solving C matrix:

        C = np.zeros((np.size(self.dofs), np.size(self.dofs)))

        for eldof, elx, ely, material_index in zip(self.edof, self.ex, self.ey, self.element_markers):
            rho = self.materials[material_index].density
            c_v = self.materials[material_index].spec_heat
            Ce = plantml(elx, ely, (rho * c_v * self.thickness[0]))
            cfc.assem(eldof, C, Ce)

        a_stat, K, _ = self.solve_static(time)
        
        while time <= 3600:
            _, _, f = self.solve_static(time)

            K_hat = C+delta_t*theta*K
            f_hat = delta_t*f+(C-delta_t*K*(1-theta))@a
            a = np.linalg.solve(K_hat, f_hat)

            if time == n*step:
                snapshot.append(a)
                snapshot_time.append(time)
                step += 1
            time += delta_t

            if step == 6:
                break

        return snapshot, snapshot_time


class HomogenousMaterial:
    """Defines material
    k: Thermal conductivity [W/(mK)]
    E: Young's modulus [Pa]
    v: Poisson's ratio
    alpha: expansion coefficient [1/K]
    density: [kg/m^3]
    spec_heat: [J/(kg K)]
    name: optional material name
    """

    def __init__(self, k, E, v, alpha, density, spec_heat, name=None) -> None:
        self.k = k
        self.D = np.diag([k, k])
        self.E = E
        self.v = v
        self.alpha = alpha
        self.density = density
        self.spec_heat = spec_heat
        self.name = name
