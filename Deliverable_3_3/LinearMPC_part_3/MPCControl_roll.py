import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron

import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    # state: [delta_z, gamma]
    x_ids: np.ndarray = np.array([2, 5])
    # input: [P_diff]
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        nx, nu, N = self.nx, self.nu, self.N

        Q = np.diag([50.0, 10.0])   
        R = np.diag([1.0])         

        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K_lqr
        A_cl = self.A + self.B @ K
        gamma_max = np.deg2rad(10.0)  
        Pdiff_max = 20.0             

        Pdiff_s = float(self.us[0])    

        dP_lo = -Pdiff_max - Pdiff_s
        dP_hi =  Pdiff_max - Pdiff_s

      
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
        O_inf = max_invariant_set(A_cl,KU)

        A_inf = O_inf.A
        b_inf = O_inf.b + O_inf.A @ self.xs

        Rf_abs = Polyhedron.from_Hrep(A_inf, b_inf)

        proj1 = [0, 1]   
        Rf_2d_1 = Rf_abs.projection(proj1)

        z_s, gamma_s = self.xs  

        state_labels = {
            0: r"$\omega z$",
            1: r"$\gamma$",
        }

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        Rf_2d_1.plot(ax=ax, color="blue")
        ax.set_xlabel(state_labels[proj1[0]])
        ax.set_ylabel(state_labels[proj1[1]])

        ax.scatter(z_s, gamma_s, c="black", marker="x", s=50)

        ax.set_title(r"$R_f$ in $(\omega z,\gamma)$")

        fig.tight_layout()
        fig.savefig("Rf_projections_roll.png", dpi=300)
        plt.show()


        X = cp.Variable((nx, N + 1), name="X")   
        U = cp.Variable((nu, N), name="U")       
        x0_p   = cp.Parameter(nx, name="x0")
        xref_p = cp.Parameter(nx, name="xref")

        constraints = [X[:, 0] == x0_p]
        cost = 0

        for k in range(N):
            constraints += [X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k]]

            constraints += [cp.abs(Pdiff_s + U[0, k]) <= Pdiff_max]

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
