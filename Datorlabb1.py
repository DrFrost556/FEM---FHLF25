import numpy as np
import calfem.core as cfc

L = 6.0
num_elements = 3
dx = L / num_elements

A = 10.0  # Cross-sectional area
k = 5.0  # Thermal conductivity
Q = 100.0  # Heat supply

# Edof
Edof = np.zeros((num_elements, 2), dtype=int)
for i in range(num_elements):
    Edof[i] = [i + 1, i + 2]

n_nodes = num_elements + 1  # Number of nodes (including boundary nodes)
K = np.zeros((n_nodes, n_nodes))  # stiffness matrix
F = np.zeros((n_nodes,1))  # load vector

# Assemble element stiffness matrices and load vector
for i in range(num_elements):
    ke = cfc.spring1e(k*A/dx)

    fe = (Q * dx / 2) * np.array([1, 1])

    cfc.assem(Edof[i], K, ke)
    F[Edof[i] - 1, 0] += fe

print("Global Stiffness Matrix (K):\n", K)

bc = np.array([1])  # Boundary nodes (first)
bcVal = np.array([0])  # Prescribed temperature, always 0
F[n_nodes-1] += -A*15 # Add boundary condition to load vector

a, r = cfc.solveq(K, F, bc, bcVal)

print("Temperature Distribution (a):\n", a)