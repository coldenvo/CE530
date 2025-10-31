import numpy as np
import matplotlib.pyplot as plt

# -------------------- User/problem data --------------------
coords_ft = np.array([
    [0.0, 0.0],    # node 0 (fixed)
    [30.0, 30.0],  # node 1 (knee)
    [70.0, 30.0]   # node 2 (fixed)
])
coords_in = coords_ft * 12.0  # convert ft → in

connectivity = np.array([
    [0, 1],  # element 0 (diagonal)
    [1, 2]   # element 1 (horizontal)
])

E = 30e6        # psi
A = 100.0       # in^2
I = 1000.0      # in^4

supports = np.array([
    [1, 1, 1],  # node 0 fixed
    [0, 0, 0],  # node 1 free
    [1, 1, 1]   # node 2 fixed
], dtype=int)

# Distributed load on element 1 (horizontal beam): 1 kip/ft downward
w_kip_per_ft = 1.0
w_lb_per_in = (w_kip_per_ft * 1000.0) / 12.0  # lb/in (positive downward)

# -------------------- FE helpers --------------------
def length_cos_sin(x1, y1, x2, y2):
    L = np.hypot(x2 - x1, y2 - y1)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    return L, c, s

def ke_local_frame(E, A, I, L):
    k = np.zeros((6,6))
    EA, EI = E*A, E*I
    k[0,0] = EA/L; k[0,3] = -EA/L
    k[3,0] = -EA/L; k[3,3] = EA/L
    k[1,1] = 12*EI/L**3; k[1,2] = 6*EI/L**2; k[1,4] = -12*EI/L**3; k[1,5] = 6*EI/L**2
    k[2,1] = 6*EI/L**2; k[2,2] = 4*EI/L; k[2,4] = -6*EI/L**2; k[2,5] = 2*EI/L
    k[4,1] = -12*EI/L**3; k[4,2] = -6*EI/L**2; k[4,4] = 12*EI/L**3; k[4,5] = -6*EI/L**2
    k[5,1] = 6*EI/L**2; k[5,2] = 2*EI/L; k[5,4] = -6*EI/L**2; k[5,5] = 4*EI/L
    return k

def T_global_to_local(c, s):
    R = np.array([[ c,  s, 0],
                  [-s,  c, 0],
                  [ 0,  0, 1]])
    T = np.zeros((6,6))
    T[:3,:3] = R
    T[3:,3:] = R
    return T

def fe_local_consistent(q_local, L):
    return q_local * L / 2.0 * np.array([0.0, 1.0, L/6.0, 0.0, 1.0, -L/6.0])

# -------------------- Assembly --------------------
Nnodes = len(coords_in)
ndofs = 3 * Nnodes
KG = np.zeros((ndofs, ndofs))
F = np.zeros(ndofs)

element_data = []
for e,(n1,n2) in enumerate(connectivity):
    x1,y1 = coords_in[n1]; x2,y2 = coords_in[n2]
    L, c, s = length_cos_sin(x1,y1,x2,y2)
    Kl = ke_local_frame(E,A,I,L)
    T = T_global_to_local(c,s)
    Kg = T.T @ Kl @ T
    dofs = [3*n1,3*n1+1,3*n1+2,3*n2,3*n2+1,3*n2+2]
    KG[np.ix_(dofs,dofs)] += Kg

    if e == 1:  # horizontal element load
        q_local = -w_lb_per_in
        fe_local = fe_local_consistent(q_local, L)
    else:
        fe_local = np.zeros(6)
    element_data.append((Kl, T, dofs, L, fe_local))

    if np.any(fe_local):
        F[np.ix_(dofs)] += T.T @ fe_local

# -------------------- Apply BCs & Solve --------------------
KT, FT = KG.copy(), F.copy()
node_dof = np.arange(ndofs).reshape(Nnodes,3)
for n in range(Nnodes):
    for k in range(3):
        if supports[n,k]:
            d = node_dof[n,k]
            KT[d,:] = 0; KT[:,d] = 0; KT[d,d] = 1
            FT[d] = 0

U = np.linalg.solve(KT, FT)
Reac = KG @ U - F

# -------------------- Element Internals --------------------
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

