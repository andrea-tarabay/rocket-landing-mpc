import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    # Reduced state: [w_y, beta, v_x]
    x_ids: np.ndarray = np.array([1, 4, 6])
    # Reduced input: [delta_2]
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        nx, nu, N = self.nx, self.nu, self.N

        Q = np.diag([2000.0, 10.0, 10.0])
        R = np.diag([1.0])
        S = np.diag([500.0])
        S_lin=1000.0
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)   
        K = -K_lqr                                  
        A_cl = self.A + self.B @ K

        beta_max = np.deg2rad(10.0)     
        delta2_max = np.deg2rad(15.0)    

        beta_s = float(self.xs[1])      
        d2_s = float(self.us[0])      

        beta_lo = -beta_max - beta_s
        beta_hi =  beta_max - beta_s

        d2_lo = -delta2_max - d2_s
        d2_hi =  delta2_max - d2_s

        F_x = np.array([
            [0.0,  1.0, 0.0],  
            [0.0, -1.0, 0.0],   
        ])
        f_x = np.array([beta_hi, -beta_lo], dtype=float)
        Xset = Polyhedron.from_Hrep(F_x, f_x)

        F_u = np.array([[1.0], [-1.0]])
        f_u = np.array([d2_hi, -d2_lo], dtype=float)
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

        O_inf = max_invariant_set(A_cl, Xset.intersect(KU))


        X = cp.Variable((nx, N + 1), name="X")    
        U = cp.Variable((nu, N), name="U")        
        x0_p = cp.Parameter(nx, name="x0")        
        xref_p = cp.Parameter(nx, name="xref")   
        eps=cp.Variable((1,1),nonneg=True, name="eps")
        constraints = [X[:, 0] == x0_p]
        cost = 0

        for k in range(N):
            constraints += [X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k]]
            constraints += [cp.abs(beta_s + X[1, k]) - eps <= beta_max]
            constraints += [cp.abs(d2_s + U[0, k]) <= delta2_max]

            dx = X[:, k] - xref_p
            cost += cp.quad_form(dx, Q) + cp.quad_form(U[:, k], R) +cp.quad_form(eps,S) + eps * S_lin

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
