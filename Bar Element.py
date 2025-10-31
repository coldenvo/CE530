import numpy as np
import matplotlib.pyplot as plt

#Problem 3.1, Page 77

# ===================== USER INPUT =====================
# Node coordinates (in inches)
coordinates = np.array([0.0, 30, 60, 90])  # 4 nodes → 3 elements

# Connectivity (each row = [start_node, end_node])
connectivity = np.array([
    [0, 1],
    [1, 2],
    [2, 3]
])

# Element properties
E_elem = np.array([30e6, 15e6, 15e6])  # Young's modulus (psi)
A_elem = np.array([1.0, 1.0, 2.0])     # Cross-sectional area (in²)

# Boundary conditions (1 = fixed, 0 = free)
supports = np.array([1, 0, 0, 1])  # Node 1 and Node 4 fixed

# Applied nodal loads (lb)
forces = np.array([0.0, 3000.0, 0.0, 0.0])  # 3000 lb at node 2

# ======================================================
num_nodes = len(coordinates)
num_elements = len(connectivity)

# Initialize global stiffness matrix and force vector
K_global = np.zeros((num_nodes, num_nodes))
F_global = np.zeros(num_nodes)

# Assemble global stiffness matrix
for e in range(num_elements):
    n1, n2 = connectivity[e]
    x1, x2 = coordinates[n1], coordinates[n2]
    L = abs(x2 - x1)
    E = E_elem[e]
    A = A_elem[e]
    
    # Local stiffness matrix
    k = (E * A / L) * np.array([[1, -1], [-1, 1]])
    
    # Assemble into global stiffness
    K_global[n1, n1] += k[0, 0]
    K_global[n1, n2] += k[0, 1]
    K_global[n2, n1] += k[1, 0]
    K_global[n2, n2] += k[1, 1]

# Apply loads
F_global[:] = forces

# Apply boundary conditions
K_mod = K_global.copy()
F_mod = F_global.copy()
for i in range(num_nodes):
    if supports[i] == 1:
        K_mod[i, :] = 0
        K_mod[:, i] = 0
        K_mod[i, i] = 1
        F_mod[i] = 0

# Solve for displacements
U = np.linalg.solve(K_mod, F_mod)

# Compute reaction forces using original (unmodified) K
reaction_forces = K_global @ U - F_global

# Compute element forces and stresses
forces_elem = np.zeros(num_elements)
stress_elem = np.zeros(num_elements)
for e in range(num_elements):
    n1, n2 = connectivity[e]
    x1, x2 = coordinates[n1], coordinates[n2]
    L = abs(x2 - x1)
    E = E_elem[e]
    A = A_elem[e]
    
    strain = (U[n2] - U[n1]) / L
    stress = E * strain
    stress_elem[e] = stress
    forces_elem[e] = stress * A

# ===================== OUTPUT =====================
np.set_printoptions(precision=3, suppress=True)

def print_matrix(matrix, title):
    print(f"\n{'='*10} {title} {'='*10}")
    header = "       " + "".join([f"{j+1:^12}" for j in range(matrix.shape[1])])
    print(header)
    print("   " + "-" * (len(header)-3))
    for i, row in enumerate(matrix):
        row_str = " ".join([f"{val:12.3e}" for val in row])
        print(f"{i+1:^3}| {row_str}")

print_matrix(K_global, "GLOBAL STIFFNESS MATRIX (psi·in)")

print("\n========== NODAL DISPLACEMENTS (inches) ==========")
for i, u in enumerate(U):
    print(f"Node {i+1}: {u:.6e}")

print("\n========== REACTION FORCES (lb) ==========")
for i, r in enumerate(reaction_forces):
    if supports[i] == 1:
        print(f"Node {i+1}: {r:10.3f}")

print("\n========== ELEMENT RESULTS ==========")
print(f"{'Element':<10}{'Force (lb)':>15}{'Stress (psi)':>20}")
print("-" * 45)
for i in range(num_elements):
    print(f"{i+1:<10}{forces_elem[i]:>15.3f}{stress_elem[i]:>20.3e}")

# ===================== PLOTTING =====================
import matplotlib.pyplot as plt

magnification = 5000  # deformation scale factor for visibility
deformed_coords = coordinates + U * magnification

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
plt.subplots_adjust(wspace=0.3)

# --- Undeformed shape ---
axes[0].plot(coordinates, np.zeros_like(coordinates), 'ko-', label='Undeformed')
for i, x in enumerate(coordinates):
    axes[0].text(x, 0.002, f'N{i+1}', ha='center', fontsize=9)
axes[0].set_title("Undeformed Shape")
axes[0].set_xlabel("Position along truss (in)")
axes[0].set_ylabel("Truss axis (schematic)")
axes[0].grid(True)
axes[0].legend()

# --- Deformed shape ---
axes[1].plot(deformed_coords, np.zeros_like(deformed_coords), 'bo-', label=f'Deformed (x{magnification:g})')
for i, x in enumerate(deformed_coords):
    axes[1].text(x, 0.002, f'N{i+1}', ha='center', fontsize=9)
axes[1].set_title(f"Deformed Shape (Magnified ×{magnification:g})")
axes[1].set_xlabel("Position along truss (in)")
axes[1].grid(True)
axes[1].legend()

# Keep same x-scale for visual comparison
xmin = min(coordinates[0], deformed_coords[0])
xmax = max(coordinates[-1], deformed_coords[-1])
axes[0].set_xlim(xmin, xmax)
axes[1].set_xlim(xmin, xmax)

plt.tight_layout()
plt.show()
