import numpy as np
import calfem.core as cfc

# Define problem parameters
L = 6.0  # Length of the fin (meters)
num_elements = 3  # Number of elements
dx = L / num_elements  # Length of each element

A = 10.0  # Cross-sectional area (m^2)
k = 5.0  # Thermal conductivity (J/(°C*m*s))
Q = 100.0  # Heat supply (J/(m*s))

# Element degrees of freedom
Edof = np.zeros((num_elements, 2), dtype=int)
for i in range(num_elements):
    Edof[i] = [i + 1, i + 2]

# Initialize global stiffness matrix and load vector
n_nodes = num_elements + 1  # Number of nodes (including boundary nodes)
K = np.zeros((n_nodes, n_nodes))  # Global stiffness matrix
F = np.zeros(n_nodes)  # Global load vector

# Assemble element stiffness matrices and load vectors
for i in range(num_elements):
    # Element stiffness matrix for 1-D heat conduction (analogous to spring element in calfem)
    ke = cfc.spring1e(k*A/dx)

    # Element load vector due to heat supply Q (similar to force vector in calfem)
    fe = (Q * dx / 2) * np.array([1, 1])

    # Assemble into global stiffness matrix and load vector
    cfc.assem(Edof[i], K, ke)
    F[Edof[i] - 1] += fe

# Print global stiffness matrix for debugging
print("Global Stiffness Matrix (K):\n", K)

# Apply boundary condition at x = L (right end)
# Heat flux boundary condition: q(x = L) = 15 (J/m^2)
bc = np.array([n_nodes])  # Boundary nodes (last)
bcVal = np.array([-A * 15.0])  # Prescribed heat flux

# Solve for temperatures at interior nodes
K_interior = K[1:, 1:]  # Remove first row and column from K
F[n_nodes-1] += bcVal
F_interior = F[1:]  # Remove first element from F

a_interior = np.linalg.solve(K_interior, F_interior)

# Insert known temperature at the first node (x = 0)
a = np.insert(a_interior, 0, 0.0)

# Print calculated temperatures and heat flux
print("Temperature Distribution (a):\n", a)