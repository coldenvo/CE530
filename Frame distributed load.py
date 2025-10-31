import numpy as np
import matplotlib.pyplot as plt

# -------------------- Problem setup (units: lb, in) --------------------
# Nodes placed so element lengths = 40 ft = 480 in
coordinates = np.array([
    [0.0,   0.0],   # node 0 (fixed)
    [30, 30],   # node 1 (middle)
    [70, 30]    # node 2 (fixed)
])

connectivity = np.array([
    [0, 1],
    [1, 2]
])

# Material / section (from problem)
E = 30e6      # psi
A = 100.0     # in^2
I = 1000.0    # in^4

# supports: node0 fixed, node1 free, node2 fixed
supports = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=int)

# Distributed load on second element
# Given: 1000 lb/ft downward over the element.
w_lb_per_ft = 1000.0       # lb/ft (given)
w = w_lb_per_ft / 12.0     # convert to lb/in
elem_index_with_w = 1      # element (1-> nodes [1,2]) receives the distributed load

# initialize applied nodal loads [Fx, Fy, M] per node
Nnodes = len(coordinates)
applied_loads = np.zeros((Nnodes, 3), dtype=float)


# -------------------- FE helper functions --------------------
def length_cos_sin(x1, y1, x2, y2):
    L = np.hypot(x2 - x1, y2 - y1)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    return L, c, s

def ke_local_frame(E, A, I, L):
    k = np.zeros((6,6))
    EA = E * A
    EI = E * I
    # axial
    k[0,0] =  EA/L;  k[0,3] = -EA/L
    k[3,0] = -EA/L;  k[3,3] =  EA/L
    # bending (standard Euler-Bernoulli 2D frame element)
    k[1,1] =  12*EI/L**3;  k[1,2] =  6*EI/L**2;  k[1,4] = -12*EI/L**3;  k[1,5] =  6*EI/L**2
    k[2,1] =   6*EI/L**2;  k[2,2] =  4*EI/L;     k[2,4] =  -6*EI/L**2;  k[2,5] =  2*EI/L
    k[4,1] = -12*EI/L**3;  k[4,2] = -6*EI/L**2;  k[4,4] =  12*EI/L**3;  k[4,5] = -6*EI/L**2
    k[5,1] =   6*EI/L**2;  k[5,2] =  2*EI/L;     k[5,4] =  -6*EI/L**2;  k[5,5] =  4*EI/L
    return k

def T_global_to_local(c, s):
    R = np.array([[ c,  s, 0],
                  [-s,  c, 0],
                  [ 0,  0, 1]])
    T = np.zeros((6,6))
    T[:3,:3] = R
    T[3:,3:] = R
    return T

# -------------------- Build equivalent nodal loads for uniform w on chosen element --------------------
if 0 <= elem_index_with_w < len(connectivity):
    n1, n2 = connectivity[elem_index_with_w]
    x1, y1 = coordinates[n1]
    x2, y2 = coordinates[n2]
    L, c, s = length_cos_sin(x1, y1, x2, y2)

    # CONSISTENT nodal load vector in the element local coordinates for uniform transverse load q (positive *up*).
    # Many references give: fe_local = q * L/2 * [0, 1, L/6, 0, 1, -L/6]  where "q" is load per length in local y-direction.
    # w positive downward (lb/ft). Local +y is upward, so q_local = -w (lb/in).
    q_local = -w
    fe_local = q_local * L / 2.0 * np.array([0.0, 1.0, L/6.0, 0.0, 1.0, -L/6.0])

    # Transform to global coordinates: fe_global = T^T * fe_local (because fe_local defined in local DOFs)
    T = T_global_to_local(c, s)
    fe_global = T.T @ fe_local

    # Add to nodal loads array. fe_global ordering = [u1, v1, th1, u2, v2, th2] in GLOBAL components now.
    dof_list = [ (n1,0), (n1,1), (n1,2), (n2,0), (n2,1), (n2,2) ]
    for i_local, (node, comp) in enumerate(dof_list):
        applied_loads[node, comp] += fe_global[i_local]

    # print the equivalent nodal quantities (notice the moment components)
    print("\nEquivalent nodal vector (global) from uniform w on element", elem_index_with_w)
    for i,(Fx,Fy,M) in enumerate(applied_loads):
        # note: Fy sign: negative = downward in this coordinate system
        print(f" Node {i}: Fx = {Fx:,.3f} lb, Fy = {Fy:,.3f} lb, M = {M:,.3f} lb·in")
else:
    raise IndexError("elem_index_with_w out of range")

# -------------------- Assembly --------------------
Ne = len(connectivity)
ndofs = 3 * Nnodes
KG = np.zeros((ndofs, ndofs))
F = np.zeros(ndofs)

# element DOF map
elem_dofs = []
for e,(n1,n2) in enumerate(connectivity):
    dofs = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
    elem_dofs.append(dofs)

# assemble element stiffness matrices (global)
for e, (n1, n2) in enumerate(connectivity):
    x1, y1 = coordinates[n1]
    x2, y2 = coordinates[n2]
    L, c, s = length_cos_sin(x1, y1, x2, y2)
    Kl = ke_local_frame(E, A, I, L)
    T = T_global_to_local(c, s)
    Kg = T.T @ Kl @ T
    dofs = elem_dofs[e]
    KG[np.ix_(dofs, dofs)] += Kg

# assemble nodal loads into global F
for node in range(Nnodes):
    F[3*node:3*node+3] += applied_loads[node]

# -------------------- Apply bcs and solve --------------------
KT = KG.copy()
FT = F.copy()
node_dof = np.arange(ndofs).reshape(Nnodes, 3)

for n in range(Nnodes):
    for k in range(3):
        if supports[n, k] == 1:
            d = node_dof[n, k]
            KT[d, :] = 0.0
            KT[:, d] = 0.0
            KT[d, d] = 1.0
            FT[d] = 0.0

U = np.linalg.solve(KT, FT)
Reac = KG @ U - F

# -------------------- Print results --------------------
np.set_printoptions(precision=6, suppress=True)
print("\nNodal displacements [ux (in), uy (in), theta (rad)]:")
for n in range(Nnodes):
    ux, uy, th = U[3*n:3*n+3]
    print(f" Node {n}: ux={ux:.6e}, uy={uy:.6e}, theta={th:.6e}")

print("\nReaction forces/moments [Fx (lb), Fy (lb), M (lb·in)]:")
for n in range(Nnodes):
    Fx, Fy, M = Reac[3*n:3*n+3]
    print(f" Node {n}: Fx={Fx:,.3f}, Fy={Fy:,.3f}, M={M:,.3f}")

# -------------------- Simple plot (undeformed vs deformed, translations only) --------------------
def plot_nodes_only(coords, connectivity, U, scale=None):
    ux = U[0::3]; uy = U[1::3]
    if len(connectivity):
        Lc = max(np.hypot(*(coords[j] - coords[i])) for i,j in connectivity)
    else:
        Lc = 1.0
    Um = np.max(np.sqrt(ux**2 + uy**2)) if U.size else 0.0
    if scale is None:
        scale = 0.15 * Lc / (Um if Um>0 else 1.0)
    def_xy = coords + np.column_stack((ux, uy)) * scale
    fig, ax = plt.subplots(figsize=(8,4))
    for i,j in connectivity:
        ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'k--', lw=1.5)
    for i,j in connectivity:
        ax.plot([def_xy[i,0], def_xy[j,0]], [def_xy[i,1], def_xy[j,1]], 'r-', lw=2)
    ax.scatter(coords[:,0], coords[:,1], color='gray', label='undeformed')
    ax.scatter(def_xy[:,0], def_xy[:,1], facecolors='none', edgecolors='r', label='deformed')
    ax.set_aspect('equal'); ax.grid(True); ax.legend(); ax.set_xlabel('in'); ax.set_ylabel('in')
    plt.show()

plot_nodes_only(coordinates, connectivity, U)
