import calfem.geometry as cfg
import calfem.core as cfc
import calfem.mesh as cfm
import calfem.utils as cfu
import calfem.vis_mpl as cfv
import numpy as np

# Mesh data
el_sizef, el_type, dofs_pn = 1, 2, 1
mesh_dir = ".venv/"

MARKER_TOP = 0
MARKER_SIDES = 1
MARKER_WARM = 2
MARKER_COOL = 3
MARKER_0 = 4


# initialize mesh
g = cfg.geometry()

# define parameters for the geometry
R = 0.25   # Radius of circles
M = 1   # Y-axis for middle of circles
B = 3   # Length of bottom part
L = 4   # Length of top part
H = 2   # Height of battery
D = 0.875   # Length between middle of circles

# add points
# points for surface
g.point([0, 0], 0)
g.point([0, M-R], 1)
g.point([0, M], 2)
g.point([R, M], 3)
g.point([0, M+R], 4)
g.point([0, H], 5)
g.point([L, H], 6)
g.point([B, 0], 7)

# circle 1
g.point([D, M], 8)
g.point([D-R, M], 9)
g.point([D, M+R], 10)
g.point([D+R, M], 11)
g.point([D, M-R], 12)

# circle 2
g.point([2*D, M], 13)
g.point([2*D-R, M], 14)
g.point([2*D, M+R], 15)
g.point([2*D+R, M], 16)
g.point([2*D, M-R], 17)

# circle 3
g.point([3*D, M], 18)
g.point([3*D-R, M], 19)
g.point([3*D, M+R], 20)
g.point([3*D+R, M], 21)
g.point([3*D, M-R], 22)

# define lines / circle segments
g.spline([0, 1], 0, marker=MARKER_0)
g.circle([1, 2, 3], 1, marker=MARKER_WARM)
g.circle([3, 2, 4], 2, marker=MARKER_WARM)
g.spline([4, 5], 3, marker=MARKER_0)
g.spline([5, 6], 4, marker=MARKER_TOP)
g.spline([6, 7], 5, marker=MARKER_SIDES)
g.spline([7, 0], 6, marker=MARKER_SIDES)

# circle 1
g.circle([9, 8, 10], 7, marker=MARKER_COOL)
g.circle([10, 8, 11], 8, marker=MARKER_COOL)
g.circle([11, 8, 12], 9, marker=MARKER_COOL)
g.circle([12, 8, 9], 10, marker=MARKER_COOL)

# circle 2
g.circle([14, 13, 15], 11, marker=MARKER_WARM)
g.circle([15, 13, 16], 12, marker=MARKER_WARM)
g.circle([16, 13, 17], 13, marker=MARKER_WARM)
g.circle([17, 13, 14], 14, marker=MARKER_WARM)

# circle 3
g.circle([19, 18, 20], 15, marker=MARKER_COOL)
g.circle([20, 18, 21], 16, marker=MARKER_COOL)
g.circle([21, 18, 22], 17, marker=MARKER_COOL)
g.circle([22, 18, 19], 18, marker=MARKER_COOL)

g.surface([0, 1, 2, 3, 4, 5, 6], [[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]])

cfv.draw_geometry(g)

# generate mesh
mesh = cfm.GmshMeshGenerator(g, mesh_dir=mesh_dir)
mesh.el_size_factor = el_sizef
mesh.el_type = el_type
mesh.dofs_per_node = dofs_pn
coord, edof, dofs, bdofs, element_markers = mesh.create()

cfv.figure()

# Draw the mesh.

cfv.drawMesh(
    coords=coord,
    edof=edof,
    dofs_per_node=mesh.dofsPerNode,
    el_type=mesh.elType,
    filled=True,
    title="Mesh"
)

# Boundary Conditions
bc, bc_value = np.array([], 'i'), np.array([], 'f')
bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_TOP, 293, 1)
bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_SIDES, 293, 1)
bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_WARM, 293, 1)
bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_COOL, 293, 1)

K = np.mat(np.zeros((len(dofs), len(dofs))))
F = np.mat(np.zeros((len(dofs), 1)))
ex, ey = cfc.coordxtr(edof, coord, dofs)

for eltopo, elx, ely, elMarker in zip(edof, ex, ey, element_markers):
    ep = [1.0]
    D = [[dofs_pn, 0],
         [0, dofs_pn]]
    Ke = cfc.flw2te(elx, ely, ep, D)
    cfc.assem(eltopo, K, Ke)


a, r = cfc.solveq(K, F, bc, bc_value)

cfv.figure()
cfv.draw_nodal_values(a, coord, edof)
cfv.colorbar()
cfv.show()
