import numpy as np
import matplotlib.pyplot as plt
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.utils as cfu
import calfem.vis_mpl as cfv

# Define constants and problem parameters
k = 1.0  # Thermal conductivity (J/(m*s*K))

# Generate mesh and apply boundary conditions
def generate_mesh_and_bc(show_geometry=False):
    # Create geometry
    g = cfg.Geometry()

    # Define points
    g.point([0, 0], 0)   # Point 0 (center of circle)
    g.point([2, 0], 1)   # Point 1 (right end of the horizontal line)
    g.point([10, 0], 2)  # Point 2 (outer boundary right)
    g.point([10, 10], 3) # Point 3 (outer boundary top)
    g.point([0, 10], 4)  # Point 4 (outer boundary left)
    g.point([0, 2], 5)   # Point 5 (left end of the horizontal line)

    # Define splines (lines)
    g.circle([1, 0, 5], 0, marker=0)  # Circle with radius 1, marker=0 (T=1000)
    g.spline([1, 2], 1, marker=2)      # Horizontal line, marker=2 (q=0)
    g.spline([2, 3], 2, marker=1)      # Outer boundary, marker=1 (T=100)
    g.spline([3, 4], 3, marker=1)      # Outer boundary, marker=1 (T=100)
    g.spline([4, 5], 4, marker=2)      # Horizontal line, marker=2 (q=0)

    # Define surface
    g.surface([0, 1, 2, 3, 4])

    # Generate mesh
    mesh = cfm.GmshMeshGenerator(g)
    mesh.el_size_factor = 0.5
    mesh.el_type = 2
    mesh.dofs_per_node = 1
    coord, edof, dofs, bdofs, element_markers = mesh.create()

    # Display geometry
    if show_geometry:
        cfv.figure()
        cfv.draw_geometry(g)
        cfv.show()

    # Apply boundary conditions
    bc, bc_value = np.array([], dtype=int), np.array([], dtype=float)
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, 0, 1000.0, 1)  # T = 1000 in the hole (marker=0)
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, 1, 100.0, 1)   # T = 100 at the outer boundaries (marker=1)

    return coord, edof, dofs, bc, bc_value


def solve_heat_flow():
    # Generate mesh and apply boundary conditions
    coord, edof, dofs, bc, bc_value = generate_mesh_and_bc(show_geometry=True)

    # Initialize global stiffness matrix and load vector
    K = np.zeros((len(dofs), len(dofs)))
    F = np.zeros(len(dofs))

    # Assemble element stiffness matrix (using flw2te for heat flow)

    # Assemble element stiffness matrix (using flw2te for heat flow)
    mesh = cfm.GmshMeshGenerator(cfg.geometry())
    mesh.el_size_factor = 0.5
    mesh.el_type = 2
    mesh.dofs_per_node = 1
    coord, edof, dofs, bdofs, element_markers = mesh.create()

    for eltopo, elx, ely, el_marker in zip(edof, coord[edof[:, 0]-1], coord[edof[:, 1]-1], element_markers):
        ke = cfc.flw2te(elx, ely, k)
        cfc.assem(eltopo, K, ke)

    # Apply boundary conditions
    a, r = cfc.solveq(K, F, bc, bc_value)

    # Plot temperature distribution
    cfv.figure()
    cfv.draw_nodal_values(a, coord, edof)
    cfv.colorbar()
    cfv.show()

if __name__ == "__main__":
    solve_heat_flow()