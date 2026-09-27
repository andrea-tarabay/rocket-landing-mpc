import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron 
from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        Q = np.diag([200.0, 10.0, 10.0, 10.0])
        R = np.diag([0.05])
        S = np.diag([500.0])
        S_lin=1000.0
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)   
        K = -K_lqr                                  
        A_cl = self.A + self.B @ K

        beta_max = np.deg2rad(10.0)    
        delta2_max = np.deg2rad(15.0)    

        beta_s = float(self.xs[1])      
        delta2_s = float(self.us[0])      

        F_x = np.array([
            [0.0,  1.0, 0.0, 0.0],   
            [0.0, -1.0, 0.0, 0.0],  
        ])
        f_x = np.array([beta_max - beta_s, beta_max + beta_s], dtype=float)
        Xset = Polyhedron.from_Hrep(F_x, f_x)

        F_u = np.array([[1.0], [-1.0]])
        f_u = np.array([delta2_max - delta2_s, delta2_max + delta2_s], dtype=float)
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


        X = cp.Variable((self.nx, self.N + 1), name="X")    
        U = cp.Variable((self.nu, self.N), name="U")  
        eps=cp.Variable((1,1),nonneg=True, name="eps")        
        x0_p = cp.Parameter(self.nx, name="x0")        
        xref_p = cp.Parameter(self.nx, name="xref")    
        constraints=[]
        constraints.append(X[:, 0] == x0_p)
        constraints.append(X[:,1:] == self.A @ X[:, :-1] + self.B @ U) #dynamics
        #constraints.append(Xset.A@ X[:,:-1] <= Xset.b.reshape(-1,1)) #state constraint
        constraints.append(cp.abs(beta_s+X[1,:-1]) <= beta_max+eps) #state constraint softened
        constraints.append(Uset.A@ U <= Uset.b.reshape(-1,1)) #input constraint
        #constraints.append(O_inf.A@X[:,-1]<=O_inf.b.reshape(-1,1)) #terminal set for last state
        cost = 0

        for k in range(self.N):
            dx = X[:, k] - xref_p
            cost += cp.quad_form(dx, Q) + cp.quad_form(U[:, k], R) +cp.quad_form(eps,S) + eps * S_lin

        cost += cp.quad_form(X[:, self.N] - xref_p, Qf)

        self._X, self._U = X, U
        self._x0_p, self._xref_p = x0_p, xref_p
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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