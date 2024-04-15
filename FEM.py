import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import numpy as np

#Step 1: Material properties
E = 5e9  # Young's modulus [Pa]
nu = 0.36  # Poisson's ratio [-]
alpha = 60e-6  # Expansion coefficient [1/K]
rho = 540  # Density [kg/m^3]
cp = 3600  # Specific heat capacity [J/kg-K]
k = 80  # Thermal conductivity [W/m-K]


g = cfg.Geometry()
points = [[0,0],[0,2],[4,2],[3,0],[0,1],[0.25,1],[0,0.75],[0,1.25],[0.875,1],
          [0.625,1],[0.875,1.25],[1.125,1],[0.875,0.75],[1.75,1],[1.5,1],[1.75,1.25],[2,1],[1.75,0.75],
          [2.625,1],[2.375,1],[2.625,1.25],[2.875,1],[2.625,0.75]]

for xp, yp in points:
    g.point([xp,yp])

splines = [[0,6],[6,7],[7,1],[1,2],[2,3],[3,0]]

for s in splines:
    g.spline(s)

circlearcs = [[7,4,5],[5,4,6],[9,8,10],[10,8,11],[11,8,12],[12,8,9],[14,13,15],
              [15,13,16],[16,13,17],[17,13,14],[19,18,20],[20,18,21],[21,18,22],[22,18,19]]

for c in circlearcs:
    g.circle(c)

g.surface([0,1,2,3,4,5],[[8,9,10,11],[12,13,14,15],[16,17,18,19],[6,7,1]])



cfv.draw_geometry(g)

mesh = cfm.GmshMesh(g)

mesh.elType = 2          # Degrees of freedom per node.
mesh.dofsPerNode = 1     # Factor that changes element sizes.
mesh.elSizeFactor = 0.1 # Element size Factor

coords, edof, dofs, bdofs, elementmarkers = mesh.create()

cfv.figure()

# Draw the mesh.

cfv.drawMesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofsPerNode,
    el_type=mesh.elType,
    filled=True,
    title="Example 01"
        )
cfv.showAndWait()