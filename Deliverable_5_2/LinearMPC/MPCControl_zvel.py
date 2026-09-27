import cvxpy as cp
import numpy as np
from control import dlqr
from scipy.signal import place_poles
from mpt4py import Polyhedron  

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    # reduced state: [v_z]
    x_ids: np.ndarray = np.array([8])
    # reduced input: [P_avg]
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        nx, nu, N = self.nx, self.nu, self.N

        Q = np.diag([1.0])      
        R = np.diag([0.1])      

        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)

        Pavg_min = 40.0
        Pavg_max = 80.0

        Pavg_s = float(self.us[0])   


        self.nd = 1
        Bd = self.B.copy()              
        self.Bd = Bd

        ny = nx                        
        C = np.eye(ny)
        Cd = np.zeros((ny, self.nd))


        A_hat = np.vstack((
            np.hstack((self.A, Bd)),
            np.hstack((np.zeros((self.nd, nx)), np.eye(self.nd)))
        ))
        B_hat = np.vstack((self.B, np.zeros((self.nd, nu))))
        C_hat = np.hstack((C, Cd))

        # poles = np.array([0.6, 0.7])     
        # poles = np.array([0.9, 0.95])    
        poles = np.array([0.8, 0.85])     
 

        L = place_poles(A_hat.T, C_hat.T, poles).gain_matrix.T

        self.A_hat = A_hat
        self.B_hat = B_hat
        self.C_hat = C_hat
        self.L = L

        self._x_hat = np.zeros(nx)
        self._d_hat = np.zeros(self.nd)
        self._u_prev = np.zeros(nu)

        self.d_hat_hist = []
        self.x_hat_hist = []
        self.x_err_hist = []


        X = cp.Variable((nx, N + 1), name="X")   
        U = cp.Variable((nu, N), name="U")      

        x0_hat_p = cp.Parameter(nx, name="x0_hat")  
        xref_p   = cp.Parameter(nx, name="x_ref")   
        d_hat_p  = cp.Parameter(self.nd, name="d_hat")  

        constraints = [X[:, 0] == x0_hat_p]
        cost = 0

        for k in range(N):
            constraints += [
                X[:, k + 1] == (
                    self.A @ X[:, k]
                    + self.B @ U[:, k]
                    + self.Bd @ d_hat_p  
                )
            ]


            constraints += [Pavg_s + U[0, k] >= Pavg_min]
            constraints += [Pavg_s + U[0, k] <= Pavg_max]

            dx = X[:, k] - xref_p
            cost += cp.quad_form(dx, Q) + cp.quad_form(U[:, k], R)


        cost += cp.quad_form(X[:, N] - xref_p, Qf)

        self._X = X
        self._U = U
        self._x0_hat_p = x0_hat_p
        self._xref_p = xref_p
        self._d_hat_p = d_hat_p

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):

        dx_meas = x0 - self.xs

        yk = dx_meas

        z_hat = np.concatenate((self._x_hat, self._d_hat))              # [x_tilda; d_tilda]
        z_hat_next = (
            self.A_hat @ z_hat
            + self.B_hat @ self._u_prev
            + self.L @ (yk - self.C_hat @ z_hat)
        )

        self._x_hat = z_hat_next[:self.nx]
        self._d_hat = z_hat_next[self.nx:]

        self.d_hat_hist.append(self._d_hat.copy())
        self.x_hat_hist.append(self._x_hat.copy())
        self.x_err_hist.append((x0 - self.xs) - self._x_hat)

        
        if x_target is None:
            dx_ref = np.zeros(self.nx)
        else:
            dx_ref = x_target - self.xs

        self._x0_hat_p.value = self._x_hat
        self._xref_p.value = dx_ref
        self._d_hat_p.value = self._d_hat

        self.ocp.solve(solver=cp.PIQP, warm_start=True, verbose=False)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            u0 = self.us.copy()
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        du0 = self._U.value[:, 0]
        self._u_prev = du0.copy()     

        u0 = self.us + du0


        X_pred = self._X.value
        U_pred = self._U.value
        x_traj = X_pred + self.xs.reshape(-1, 1)
        u_traj = U_pred + self.us.reshape(-1, 1)

        return u0, x_traj, u_traj