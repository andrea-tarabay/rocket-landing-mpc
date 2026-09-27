import numpy as np
import casadi as ca
from typing import Tuple

# For terminal cost (same style as your tube MPC z code)
from control import dlqr
from scipy.linalg import solve_discrete_lyapunov


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """


    def __init__(self, rocket, H: float, xs: np.ndarray, us: np.ndarray):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]
        self.rocket = rocket
        self.Ts = float(rocket.Ts)         
        self.H = float(H)                  
        self.N = int(np.round(self.H / self.Ts))  
        self.N = max(self.N, 1)
        self.nx = 12
        self.nu = 4
        self.xs = np.asarray(xs).reshape(self.nx,)
        self.us = np.asarray(us).reshape(self.nu,)

        self.F = self._build_rk4()
        self._setup_controller()

    def _build_rk4(self) -> ca.Function:

        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        dt = self.Ts
        k1 = self.f(x, u)
        k2 = self.f(x + dt/2 * k1, u)
        k3 = self.f(x + dt/2 * k2, u)
        k4 = self.f(x + dt * k3, u)
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return ca.Function("F_rk4", [x, u], [x_next])

    def _compute_terminal_P(self, Q: np.ndarray, R: np.ndarray) -> np.ndarray:

        sys = self.rocket.linearize_sys(self.xs, self.us)
        Ad, Bd, _ = sys._discretize(self.Ts)
        K_lqr, _, _ = dlqr(Ad, Bd, Q, R)
        K = -K_lqr 
        Acl = Ad + Bd @ K
        Qbar = Q + K.T @ R @ K
        P = solve_discrete_lyapunov(Acl.T, Qbar)
        
        return P

    def _setup_controller(self) -> None:
        opti = ca.Opti()

        X = opti.variable(self.nx, self.N + 1)  
        U = opti.variable(self.nu, self.N)      
        X0 = opti.parameter(self.nx)

        opti.subject_to(X[:, 0] == X0)

        for k in range(self.N):
            opti.subject_to(X[:, k+1] == self.F(X[:, k], U[:, k]))

        opti.subject_to(X[11, :] >= 0.0) # z >= 0
        
        beta_max = np.deg2rad(80.0) # |beta| <= 80 deg
        opti.subject_to(X[4, :] <=  beta_max)
        opti.subject_to(X[4, :] >= -beta_max)

        delta_max = 0.26
        Pdiff_max = 20.0
        Pavg_min = 40.0
        Pavg_max = 80.0
        opti.subject_to(U[0, :] <=  delta_max)
        opti.subject_to(U[0, :] >= -delta_max)

        opti.subject_to(U[1, :] <=  delta_max)
        opti.subject_to(U[1, :] >= -delta_max)

        opti.subject_to(U[2, :] <=  Pavg_max)
        opti.subject_to(U[2, :] >=  Pavg_min)

        opti.subject_to(U[3, :] <=  Pdiff_max)
        opti.subject_to(U[3, :] >= -Pdiff_max)

        Q = np.diag([
            500.0, 500.0, 500.0,
            200.0, 200.0, 200.0,
            50.0, 50.0, 55.0,
            100.0, 200.0, 90.0
        ])

        R = np.diag([
            3000.0, 3000.0,   
            0.2,   20      
        ])

        P = self._compute_terminal_P(Q, R)

        cost = 0
        for k in range(self.N):
            x_err = X[:, k] - self.xs
            u_err = U[:, k] - self.us
            cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_err.T, R, u_err])

        xN_err = X[:, self.N] - self.xs
        cost += ca.mtimes([xN_err.T, P, xN_err])

        opti.minimize(cost)

        opti.solver(
            "ipopt",
            {"expand": True},
            {
                "print_level": 0,
                "max_iter": 20000,
                "tol": 1e-3,
                "acceptable_tol": 1e-2
            }
        )

        self.ocp = {"opti": opti, "X": X, "U": U, "X0": X0}

    
    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        opti = self.ocp["opti"]
        X = self.ocp["X"]
        U = self.ocp["U"]
        X0 = self.ocp["X0"]

        x0 = np.asarray(x0).reshape(self.nx,)
        opti.set_value(X0, x0)
        sol = opti.solve()

        x_ol = sol.value(X)     
        u_ol = sol.value(U)     
        u0 = u_ol[:, 0].copy() 
        t_ol = t0 + self.Ts * np.arange(self.N + 1)

        return u0, x_ol, u_ol, t_ol
