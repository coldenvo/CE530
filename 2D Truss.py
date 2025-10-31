import numpy as np
import matplotlib.pyplot as plt

# Input node coordinates
coordinates = np.array([
    [0, 0],
    [4, 0],
    [4, 3]
])

# Connectivity: pairs of node indices for each element
connectivity = np.array([
    [0, 1],
    [0, 2]
])

# Element properties
E = np.array([5, 5])  # Young's modulus per element
A = np.array([25, 20])  # Area per element

# Supports: 1 = fixed, 0 = free
supports = np.array([
    [0, 0],
    [1, 1],
    [1, 1]
])

# Applied loads: [Fx, Fy] at each node
applied_loads = np.array([
    [0, -0.15],
    [0, 0],
    [0, 0]
])

num_nodes = len(coordinates)
num_dofs = 2 * num_nodes
num_elements = len(connectivity)

# DOF mapping
dofs = np.zeros((num_elements, 4), dtype=int)
NodeDof = np.zeros((num_nodes, 2), dtype=int)
for i, (n1, n2) in enumerate(connectivity):
    dofs[i, :] = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
    NodeDof[n1, :] = [2*n1, 2*n1+1]
    NodeDof[n2, :] = [2*n2, 2*n2+1]

# Initialize global stiffness matrix and force vector
KG = np.zeros((num_dofs, num_dofs))
F_global = np.zeros(num_dofs)

# Assemble global stiffness matrix
for i, (n1, n2) in enumerate(connectivity):
    x1, y1 = coordinates[n1]
    x2, y2 = coordinates[n2]
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    k_local = (E[i] * A[i] / L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    # Assemble into global matrix
    KG[np.ix_(dofs[i, :], dofs[i, :])] += k_local
    

# Apply boundary conditions and loads
kt_global = KG.copy()
for node in range(num_nodes):
    for i in range(2):
        dof = NodeDof[node, i]
        if supports[node, i] == 1:
            kt_global[dof, :] = 0
            kt_global[:, dof] = 0
            kt_global[dof, dof] = 1
            F_global[dof] = 0
        else:
            F_global[dof] = applied_loads[node, i]

# Solve for displacements
displacements = np.linalg.solve(kt_global, F_global)

# Compute reactions
reactions = np.dot(KG, displacements)

# Output
print("Node Displacements (m):")
for i in range(num_nodes):
    print(f"Node {i}: u = {displacements[2*i]:.6f}, v = {displacements[2*i+1]:.6f}")

print("\nReaction Forces (N):")
for i in range(num_nodes):
    if supports[i, 0] == 1:
        print(f"Node {i}, Fx = {reactions[2*i]:.6f}")
    if supports[i, 1] == 1:
        print(f"Node {i}, Fy = {reactions[2*i+1]:.6f}")

#----PLOT----#

# Scaling factor for visualization
scale = 5  # Adjust to make deformations visible

# Compute deformed coordinates
deformed_coords = coordinates + scale * displacements.reshape((num_nodes, 2))

plt.figure(figsize=(6,6))

# Plot undeformed truss
for i, elem in enumerate(connectivity):
    n1, n2 = elem
    x = [coordinates[n1,0], coordinates[n2,0]]
    y = [coordinates[n1,1], coordinates[n2,1]]
    label = 'Undeformed' if i == 0 else None  # Only first element gets label
    plt.plot(x, y, 'k--', label=label)

# Plot deformed truss
for i, elem in enumerate(connectivity):
    n1, n2 = elem
    x_def = [deformed_coords[n1,0], deformed_coords[n2,0]]
    y_def = [deformed_coords[n1,1], deformed_coords[n2,1]]
    label = 'Deformed' if i == 0 else None  # Only first element gets label
    plt.plot(x_def, y_def, 'r-', linewidth=2, label=label)

# Plot nodes
plt.scatter(coordinates[:,0], coordinates[:,1], color='k')
plt.scatter(deformed_coords[:,0], deformed_coords[:,1], color='r')

plt.legend()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('2D Truss: Undeformed vs Deformed Shape')
plt.axis('equal')
plt.grid(True)
plt.show()
