import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    # reduced state: [v_z]
    x_ids: np.ndarray = np.array([8])
    # reduced input: [P_avg]
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        nx, nu, N = self.nx, self.nu, self.N

        Q = np.diag([10.0])     # delta_v_z
        R = np.diag([1.0])     # delta_P_avg

        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K_lqr
        A_cl = self.A + self.B @ K

        Pavg_min = 40.0
        Pavg_max = 80.0

        Pavg_s = float(self.xs[0]) if self.xs.shape[0] == 1 else float(self.us[0])
        Pavg_s = float(self.us[0])

        dP_lo = Pavg_min - Pavg_s
        dP_hi = Pavg_max - Pavg_s

        F_u = np.array([[1.0], [-1.0]])
        f_u = np.array([dP_hi, -dP_lo], dtype=float)
        Uset = Polyhedron.from_Hrep(F_u, f_u)

        def max_invariant_set(Acl: np.ndarray, X: Polyhedron, max_iter: int = 50) -> Polyhedron:
            O = X
            for _ in range(max_iter):
                Oprev = O
                Fh, fh = O.A, O.b
                O = Polyhedron.from_Hrep(
                    np.vstack((Fh, Fh @ Acl)),
                    np.hstack((fh, fh)),
                )
                O.minHrep(True)
                if O == Oprev:
                    break
            return O

        KU = Polyhedron.from_Hrep(Uset.A @ K, Uset.b)

        O_inf = max_invariant_set(A_cl, KU)

        A_h = O_inf.A                 
        b_h = O_inf.b + O_inf.A @ self.xs  

        lb = -np.inf
        ub = np.inf

        for i in range(len(b_h)):
            a = A_h[i, 0]
            if a > 0:
                ub = min(ub, b_h[i] / a)
            elif a < 0:
                lb = max(lb, b_h[i] / a)

        x_s = float(self.xs[0])  

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.hlines(1, lb, ub, colors="blue", linewidth=8, label="Xf")
        ax.scatter(x_s, 1, c="black", marker="x", s=60, label="steady state")
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.set_xlabel("v_z")
        ax.set_title("1D Terminal invariant set")
        ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig("Xf_1D_terminal_set.png", dpi=300)
        plt.show()




        X = cp.Variable((nx, N + 1), name="X")   
        U = cp.Variable((nu, N), name="U")      
        x0_p   = cp.Parameter(nx, name="x0")
        xref_p = cp.Parameter(nx, name="xref")

        constraints = [X[:, 0] == x0_p]
        cost = 0

        for k in range(N):
            constraints += [X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k]]

            constraints += [Pavg_s + U[0, k] >= Pavg_min]
            constraints += [Pavg_s + U[0, k] <= Pavg_max]

            dx = X[:, k] - xref_p
            cost += cp.quad_form(dx, Q) + cp.quad_form(U[:, k], R)

        cost += cp.quad_form(X[:, N] - xref_p, Qf)
        constraints += [O_inf.A @ X[:, N] <= O_inf.b.reshape(-1)]

        self._X, self._U = X, U
        self._x0_p, self._xref_p = x0_p, xref_p
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        dx0 = x0 - self.xs

        if x_target is None:
            dx_ref = np.zeros(self.nx)
        else:
            dx_ref = x_target - self.xs

        self._x0_p.value = dx0
        self._xref_p.value = dx_ref

        self.ocp.solve(solver=cp.PIQP, warm_start=True, verbose=False)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            u0 = self.us.copy()
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        du0 = self._U.value[:, 0]
        u0 = self.us + du0

        X_pred = self._X.value
        U_pred = self._U.value
        x_traj = X_pred + self.xs.reshape(-1, 1)
        u_traj = U_pred + self.us.reshape(-1, 1)

        return u0, x_traj, u_traj
    