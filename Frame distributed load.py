import numpy as np
import matplotlib.pyplot as plt

#Example 5.2 from textbook

#geometry
coords_ft = np.array([
    [0.0, 0.0],    # node 0 (fixed)
    [30.0, 30.0],  # node 1 (knee)
    [70.0, 30.0]   # node 2 (fixed)
])
coords_in = coords_ft * 12.0  # convert to in

connectivity = np.array([
    [0, 1],  # element 0 (diagonal bar)
    [1, 2]   # element 1 (horizontal bar)
])

#material params from textbook
E = 30e6        # psi
A = 100.0       # in^2
I = 1000.0      # in^4

#define supports

supports = np.array([
    [1, 1, 1],  # node 0 fixed
    [0, 0, 0],  # node 1 free
    [1, 1, 1]   # node 2 fixed
], dtype=int)

# Define distributed load on second element of 1 kip/ft downward
w_kip_per_ft = 1.0
w_lb_per_in = (w_kip_per_ft * 1000.0) / 12.0  # lb/in (positive downward)

#length definition. note that textbook miscalculates the length of elem 1 I think
def length_cos_sin(x1, y1, x2, y2):
    L = np.hypot(x2 - x1, y2 - y1)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    return (L, c, s)


def ke_local_frame(E, A, I, L):
    k = np.zeros((6,6)) #6 by 6 matrix since 2 nodes and 3 DOF's
    EA, EI = E*A, E*I
    k[0,0] = EA/L; k[0,3] = -EA/L #axial DOF's
    k[3,0] = -EA/L; k[3,3] = EA/L #axial DOF's
    #beam theory bending
    k[1,1] = 12*EI/L**3; k[1,2] = 6*EI/L**2; k[1,4] = -12*EI/L**3; k[1,5] = 6*EI/L**2
    k[2,1] = 6*EI/L**2; k[2,2] = 4*EI/L; k[2,4] = -6*EI/L**2; k[2,5] = 2*EI/L
    k[4,1] = -12*EI/L**3; k[4,2] = -6*EI/L**2; k[4,4] = 12*EI/L**3; k[4,5] = -6*EI/L**2
    k[5,1] = 6*EI/L**2; k[5,2] = 2*EI/L; k[5,4] = -6*EI/L**2; k[5,5] = 4*EI/L
    return k

#convert to global coordinates
def T_global_to_local(c, s):
    R = np.array([[ c,  s, 0],
                  [-s,  c, 0],
                  [ 0,  0, 1]])
    T = np.zeros((6,6))
    T[:3,:3] = R
    T[3:,3:] = R
    return T


#Convert distributed load in local coordinates
def fe_local_consistent(q_local, L):

    # Horizontal forces = 0 (UDL only in vertical direction)
    Fx1 = 0.0
    Fx2 = 0.0

    # Vertical forces at nodes
    Fy1 = q_local * L / 2.0
    Fy2 = q_local * L / 2.0

    # Moments at nodes due to distributed load
    M1 = q_local * L**2 / 12.0  # positive at node 1
    M2 = -q_local * L**2 / 12.0  # negative at node 2

    # Assemble vector
    fe_local = np.array([Fx1, Fy1, M1, Fx2, Fy2, M2])
    return fe_local

#Global assembly
Nnodes = len(coords_in)
ndofs = 3 * Nnodes
KG = np.zeros((ndofs, ndofs)) #global stiffness matrix
F = np.zeros(ndofs) #global force vector - stores all external loads applied at every DOF

element_data = []  # store element info

for e, (n1, n2) in enumerate(connectivity):
    # Node coordinates
    x1, y1 = coords_in[n1]
    x2, y2 = coords_in[n2]

    # Element length and orientation
    L, c, s = length_cos_sin(x1, y1, x2, y2)
    print(f"Element {e} length: {L/12:.4f} ft")  # debug print in ft

    # Local stiffness matrix
    Kl = ke_local_frame(E, A, I, L)

    # Transformation to global
    T = T_global_to_local(c, s)
    Kg = T.T @ Kl @ T

    # Global DOFs for this element
    dofs = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]

    # Assemble stiffness into global matrix
    KG[np.ix_(dofs, dofs)] += Kg

    # Distributed loads
    if e == 1:  # only element 1 has UDL
        q_local = -w_lb_per_in
        fe_local = fe_local_consistent(q_local, L)
    else:
        fe_local = np.zeros(6)

    # Assemble element loads into global force vector
    if np.any(fe_local):
        fe_global = (T.T @ fe_local).flatten()  # convert to 1D
        F[dofs] += fe_global

    # Store element info for later (internal forces, output)
    element_data.append((Kl, T, dofs, L, fe_local))

#Apply BC's + Solve
KT, FT = KG.copy(), F.copy()
node_dof = np.arange(ndofs).reshape(Nnodes,3) #creates an array mapping node to its 3 DOF's (ux, uy, rot)
#apply BC's
for n in range(Nnodes):
    for k in range(3):
        if supports[n,k]:
            d = node_dof[n,k]
            KT[d,:] = 0; KT[:,d] = 0; KT[d,d] = 1
            FT[d] = 0
#solve displacements
U = np.linalg.solve(KT, FT)
#solve reactions
Reac = KG @ U - F

# -------------------- Element Internals --------------------
#convert
def to_kip(lb): return lb / 1000.0
def to_kip_in(lb_in): return lb_in / 1000.0

print("\nNODAL DISPLACEMENTS (in):")
for n in range(Nnodes):
    ux, uy, th = U[3*n:3*n+3]
    print(f" Node {n}: ux={ux:.6e}, uy={uy:.6e}, theta={th:.6e}")

print("\nREACTIONS (kip, kip·in):")
for n in range(Nnodes):
    Fx, Fy, M = Reac[3*n:3*n+3]
    print(f" Node {n}: Fx={to_kip(Fx):8.4f}, Fy={to_kip(Fy):8.4f}, M={to_kip_in(M):8.4f}")

print("\nELEMENT INTERNAL FORCES (local coords, kip / kip·in):")
for e,(Kl,T,dofs,L,fe_local) in enumerate(element_data):
    u_local = T @ U[dofs]
    f_local = Kl @ u_local - fe_local
    Fx1,Fy1,M1,Fx2,Fy2,M2 = f_local
    print(f" Element {e}: L={L/12:.2f} ft")
    print(f"  End1: Fx={to_kip(Fx1):.3f}, Fy={to_kip(Fy1):.3f}, M={to_kip_in(M1):.3f}")
    print(f"  End2: Fx={to_kip(Fx2):.3f}, Fy={to_kip(Fy2):.3f}, M={to_kip_in(M2):.3f}")

# -------------------- Plot undeformed vs deformed --------------------
scale = 1000  # exaggeration factor for clarity
coords_def = coords_in.copy()
for n in range(Nnodes):
    ux, uy = U[3*n], U[3*n+1]
    coords_def[n] += np.array([ux, uy]) * scale

plt.figure(figsize=(8,6))
for e,(n1,n2) in enumerate(connectivity):
    # undeformed
    plt.plot([coords_in[n1,0]/12, coords_in[n2,0]/12],
             [coords_in[n1,1]/12, coords_in[n2,1]/12],
             'k--', lw=1.5, label='undeformed' if e==0 else "")
    # deformed
    plt.plot([coords_def[n1,0]/12, coords_def[n2,0]/12],
             [coords_def[n1,1]/12, coords_def[n2,1]/12],
             'r-', lw=2.5, label='deformed' if e==0 else "")

plt.xlabel("X (ft)")
plt.ylabel("Y (ft)")
plt.title("Frame Deformation (exaggerated ×{:.0f})".format(scale))
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

