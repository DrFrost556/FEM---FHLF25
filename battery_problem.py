import calfem.core as cfc
import calfem.utils as cfu
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

        self.thickness = [1.6]

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
            k_e = np.array([[1 / 3, 1 / 6],
                            [1 / 6, 1 / 3]]) * \
                  alpha * distance

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

    def boundary_load_add(self, node_pair_list, f_b, temp, alpha):
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            f_b[np.isin(self.dofs, [p1, p2])] += temp * alpha * distance * 1/2

    def solve_static(self, time):
        K = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        # Create force vector
        f = np.zeros([np.size(self.dofs), 1])
        eq = self.solve_Q2(time)

        for eldof, elx, ely, material_index in zip(self.edof, self.ex, self.ey, self.element_markers):
            if eq == 0:
                Ke = cfc.flw2te(elx, ely, self.thickness, self.materials[material_index].D)
                cfc.assem(eldof, K, Ke)
            else:
                Ke, fe = cfc.flw2te(elx, ely, self.thickness, self.materials[material_index].D, eq)
                cfc.assem(eldof, K, Ke, f, fe)

        # Add different f_c
        f_c_top = list(map(itemgetter("node-number-list"), self.boundary_elements[Boundaries.TOP_BATTERY]))
        self.boundary_load_add(f_c_top, f, self.T_inf, self.alpha_n)

        f_c_warm = list(map(itemgetter("node-number-list"), self.boundary_elements[Boundaries.WARM_CIRCLE]))
        self.boundary_load_add(f_c_warm, f, self.T_out, self.alpha_c)

        f_c_cool = list(map(itemgetter("node-number-list"), self.boundary_elements[Boundaries.COOL_CIRCLE]))
        self.boundary_load_add(f_c_cool, f, self.T_in, self.alpha_c)

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

        theta = 1

        delta_t = 120
        n = 720
        step = 0

        # a_0 from T_0
        a = np.full(self.dofs.shape, self.T_0)

        snapshot = [a]
        snapshot_time = [time]

        deviation = [(np.amax(a) - self.T_0)]
        deviation_time = [time]

        min_max = np.array([np.amin(a), np.amax(a)])

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

            K_hat = C + delta_t * theta * K
            f_hat = delta_t * f + (C - delta_t * K * (1 - theta))@a
            a = np.linalg.solve(K_hat, f_hat)

            if time == n * step:
                snapshot.append(a)
                snapshot_time.append(time)
                step += 1

            deviation.append((np.amax(a) - self.T_0))
            deviation_time.append(time)
            min_max = np.vstack([min_max, [np.amin(a), np.amax(a)]])
            print(f"Time passed = {time} second")
            time += delta_t

        return snapshot, snapshot_time, deviation, deviation_time, min_max

    def solve_von_mises(self):
        ptype = 2
        time = 3600
        a_stat, _, _ = self.solve_static(time)

        # Create new degrees of freedom for the stress problem: each node has (x, y) displacement
        num_nodes = self.coords.shape[0]
        num_dofs = num_nodes * 2

        # Expand edof for the stress problem
        expanded_edof = []
        for eldof in self.edof:
            stress_eldof = np.zeros(eldof.size * 2, dtype=int)
            stress_eldof[0::2] = eldof * 2 - 2
            stress_eldof[1::2] = eldof * 2 - 1
            expanded_edof.append(stress_eldof)
        expanded_edof = np.array(expanded_edof)

        # Construct K-matrix (twice as many dofs as for the heat problem)
        K = np.zeros((num_dofs, num_dofs))

        element_constitutive_D = []
        E = self.materials[Material.BATTERY].E
        v = self.materials[Material.BATTERY].v

        for eldof, elx, ely in zip(expanded_edof, self.ex, self.ey):
            D = cfc.hooke(ptype, E, v)/(1-2*v)
            element_constitutive_D.append(D)
            Ke = cfc.plante(elx, ely, self.ep, D)
            cfc.assem(eldof, K, Ke)

        f_0 = np.zeros((num_dofs, 1))

        for eldof, elx, ely, D in zip(expanded_edof, self.ex, self.ey, element_constitutive_D):
            alpha = self.materials[Material.BATTERY].alpha
            dt = np.abs(np.mean(a_stat[eldof[0::2] // 2]) - self.T_0)
            internal_force = cfc.plantf(elx, ely, self.ep,
                                        D[np.ix_([0, 1, 3], [0, 1, 3])] * alpha * dt * (1 + v) @ np.array([1, 1, 0]).T).reshape(6, 1)
            f_0[eldof] += internal_force

        # Get dofs fixed in xy direction (in the heat problem)
        fixed_bdofs = np.array(self.bdofs[Boundaries.FIXED_BOUNDARY]).flatten() - 1
        fixed_dofs = np.zeros(fixed_bdofs.size * 2, dtype=int)
        fixed_dofs[0::2] = fixed_bdofs * 2
        fixed_dofs[1::2] = fixed_bdofs * 2 + 1

        # Get dofs fixed in y direction (in the heat problem)
        fixed_bdofs_none = np.array(self.bdofs[Boundaries.NONE_BOUNDARY]).flatten() - 1
        fixed_dofs_none = np.zeros(fixed_bdofs_none.size, dtype=int)
        fixed_dofs_none[:] = fixed_bdofs_none * 2 + 1  # Only y-direction

        bcPrescr = np.array([], dtype=int)
        bcVal = np.array([], dtype=float)
        boundaryDofs = {1: fixed_dofs}
        boundaryDofsnone = {2: fixed_dofs_none}

        # Apply boundary conditions
        bcPrescr, bcVal = cfu.applybc(boundaryDofs, bcPrescr, bcVal, 1, value=0.0, dimension=0)
        bcPrescr, bcVal = cfu.applybc(boundaryDofsnone, bcPrescr, bcVal, 2, value=0.0, dimension=1)

        # Solve for displacements with the applied boundary conditions
        displacement, _ = cfc.solveq(K, f_0, bcPrescr, bcVal)

        ed = cfc.extract_eldisp(expanded_edof, displacement)

        von_mises_element = []
        for temp_edof, elx, ely, disp, D in zip(self.edof, self.ex, self.ey, ed, element_constitutive_D):

            # Determine element stresses and strains in the element.
            [[sigx, sigy, sigz, tauxy]], _ = cfc.plants(
                elx, ely, self.ep, D, disp)

            effective_stress = np.sqrt(
                sigx ** 2 + sigy ** 2 + sigz ** 2 - sigx * sigy - sigx * sigz - sigy * sigz + 3 * tauxy ** 2)
            von_mises_element.append(effective_stress)

        von_mises_element = np.array(von_mises_element)
        von_mises_node = [np.mean(von_mises_element[np.any(np.isin(self.edof, node), axis=1)]) for node in
                          range(1, num_nodes + 1)]

        return von_mises_node, displacement


#applybc u=0

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