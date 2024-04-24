import calfem.core as cfc
import numpy as np
from operator import itemgetter
from geometry import battery_mesh, Boundaries, Material


class BatteryProblem:
    def __init__(self, size_factor=None) -> None:
        # Material parameters, found in project description
        self.materials = {
            Material.BATTERY: IsomorphicMaterial(80, 5, 0.36, 60e-6, 540, 3600, "Battery")}

        self.T_inf = 20  # [C]
        self.T_0 = 20  # [C]
        self.T_in = 4  # [C]
        self.T_out = 12 # [C]

        # Convection constants
        self.alpha_c = 120
        self.alpha_n = 40

        self.thickness = [1.0]

        self.h = -1e2

        self.L = 0.001  # [M]

        self.el_type = 2

        self.coords, self.edof, self.dofs, self.bdofs, self.element_markers, self.boundary_elements = battery_mesh(
            self.L, self.el_type, 1, size_factor)

        self.ex, self.ey = cfc.coord_extract(self.edof, self.coords, self.dofs)

        # CALFEM specific shorthand
        self.ep = [2, self.thickness[0]]

    def integrate_boundary_convection(self, node_pair_list, k_sub):
        # Calculates the integral N^tN over an element edge
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            # Specific for our three pint triangular element
            k_e = np.array([[1/3, 1/6],
                            [1/6, 1/3]]) * \
                self.alpha_c * self.thickness[0] * distance

            cfc.assem(np.array([p1, p2]), k_sub, k_e)

    def integrate_boundary_load(self, node_pair_list, f_sub, factor):
        # Calculates the integral N^tN over an element edge
        for p1, p2 in node_pair_list:
            r1 = self.coords[(self.dofs == p1).flatten()]
            r2 = self.coords[(self.dofs == p2).flatten()]
            distance = np.linalg.norm(r1 - r2)

            f_sub[np.isin(self.dofs, [p1, p2])] += factor * distance

    def solve_static(self):
        K = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        for eldof, elx, ely, material_index in zip(self.edof, self.ex, self.ey, self.element_markers):
            Ke = cfc.flw2te(elx, ely, self.thickness, self.materials[material_index].D)
            cfc.assem(eldof, K, Ke)

        # Create force vector
        f = np.zeros([np.size(self.dofs), 1])

        # Add f_h
        f_h_nodes = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[Boundaries.TOP_BATTERY]
        ))
        self.integrate_boundary_load(
            f_h_nodes, f, -self.h*self.thickness[0] * 1/2)

        # Add f_c
        f_c_nodes = list(map(
            itemgetter("node-number-list"),
            self.element_markers[Boundaries.TOP_BATTERY]
        ))
        self.integrate_boundary_load(f_c_nodes, f, self.T_inf *
                                     self.alpha_n * self.thickness[0] * 1/2)
        f_c_nodes = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[Boundaries.WARM_CIRCLE]
        ))
        self.integrate_boundary_load(f_c_nodes, f, self.T_out *
                                     self.alpha_c * self.thickness[0] * 1/2)
        f_c_nodes = list(map(
            itemgetter("node-number-list"),
            self.boundary_elements[Boundaries.COOL_CIRCLE]
        ))
        self.integrate_boundary_load(f_c_nodes, f, self.T_in *
                                     self.alpha_c * self.thickness[0] * 1/2)

        # Add K_c
        K_c = np.zeros((np.size(self.dofs), np.size(self.dofs)))
        self.integrate_boundary_convection(f_c_nodes, K_c)
        K += K_c

        a_stat = np.linalg.solve(K, f)

        return a_stat, K, f

class IsomorphicMaterial:
    """Defines an isomorphic material
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