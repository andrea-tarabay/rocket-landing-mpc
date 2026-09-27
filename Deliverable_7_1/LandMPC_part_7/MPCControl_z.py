import numpy as np
import cvxpy as cp
from scipy.signal import place_poles
from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron 
from control import dlqr
from scipy.linalg import solve_discrete_lyapunov
import matplotlib.pyplot as plt
class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        Q=np.diag([100.0,100.0])
        R=np.diag([0.5])
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K_lqr
        #K=-place_poles(self.A,self.B,[0.45,0.5]).gain_matrix
        A_cl = self.A + self.B @ K
        # Qbar = Q + K.T @ R @ K
        # Qf = solve_discrete_lyapunov(A_cl.T, Qbar)
        self._K=K   
        print("Eigenvalues of closed loop state matrix=",np.linalg.eigvals(A_cl))
        Pavg_min = 40.0
        Pavg_max = 80.0
        Pavg_s = float(self.us)
        dP_lo = Pavg_min - Pavg_s
        dP_hi = Pavg_max - Pavg_s
        F_u = np.array([[1.0], [-1.0]])
        f_u = np.array([dP_hi, -dP_lo], dtype=float)
        Uset = Polyhedron.from_Hrep(F_u, f_u)

        #Add constraint to z
        z_equib=self.xs[1]
        #dz+zs>=0 -> -dz<=zs
        F_x = np.array([[0.0,-1.0]])
        f_x = np.array([z_equib], dtype=float)
        Xset = Polyhedron.from_Hrep(F_x, f_x)

        #Finding min robus invariant set E
        Wconst=np.array([[1.0], [-1.0]])
        wrhs=np.array([5,15], dtype=float)
        W=Polyhedron.from_Hrep(Wconst, wrhs)
        W=self.B@W
        def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 30) -> Polyhedron:
            nx = A_cl.shape[0]
            Omega = W
            itr = 1
            A_cl_ith_power = np.eye(nx)
            while itr < max_iter:
                A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
                #print(itr,np.linalg.matrix_norm(A_cl_ith_power,ord=2))
                if np.linalg.matrix_norm(A_cl_ith_power, ord=2) < 1e-2:
                    print('Minimal robust invariant set computation converged after {0} iterations.'.format(itr))
                    break
                Omega_next = Omega + A_cl_ith_power @ W
                Omega_next.minHrep()               
                Omega = Omega_next
                itr += 1
            return Omega_next
        E = min_robust_invariant_set(A_cl, W)
        E.minVrep()
        #Restrict virtual input V to U-KE
        KE=E.affine_map(K)
        KE.minVrep()
        U_tightened=Uset-KE
        U_tightened_global=Polyhedron.from_Hrep(U_tightened.A,U_tightened.b+U_tightened.A@self.us)
        print("Vertices of the input constraint U=",Uset.V+self.us)
        print("Vertices of the tightened input constraint U_tilde=",U_tightened_global.V)
        X_tightened=Xset-E
        # Max invariant set for z since z now has a constraint of z>=0
        def max_invariant_set(Acl: np.ndarray, X: Polyhedron, max_iter: int = 50) -> Polyhedron:
            O = X
            for itr in range(max_iter):
                Oprev = O
                Fh, fh = O.A, O.b
                O = Polyhedron.from_Hrep(
                    np.vstack((Fh, Fh @ Acl)),
                    np.hstack((fh, fh)),
                )
                O.minHrep(True)
                if O == Oprev:
                    break
            if itr==max_iter-1:
                print('Maximum invariant set not converged.\n'.format(itr))
            return O
        KU = Polyhedron.from_Hrep(Uset.A @ K, Uset.b)
        O_inf = max_invariant_set(A_cl, Xset.intersect(KU))
        #O_inf=O_inf.minHrep()
        fig1, ax1 = plt.subplots(1, 1)
        #xs not added to E since error formulation remains the same in global and delta formulation : x-x_bar=(x-x_s)-(x_bar-xs)
        E.plot(ax1, color='r', opacity=0.5, label=r'$\mathcal{E}$')
        plt.legend()
        plt.xlabel('v_z')
        plt.ylabel('z')
        plt.title('Minimal robust invariant set E')
        plt.savefig('deliverable_6_1_E.png')
        fig2, ax2 = plt.subplots(1, 1)
        #Add equilibirum state to plot terminal set in global coordinates, not in delta formulation
        O_inf_global=Polyhedron.from_Hrep(O_inf.A,O_inf.b+O_inf.A@self.xs)
        O_inf_global.plot(ax2, color='g', opacity=0.5, label=r'$\mathcal{X}_f$')
        #KU.plot(ax2,color='r',opacity=0.5, label=r'$\mathcal{X}$' )
        plt.legend()
        plt.xlabel('v_z')
        plt.ylabel('z')
        plt.title('Terminal set for nominal trajectory for tube MPC (z)')
        plt.savefig('deliverable_6_1_X_tilde.png')
        #print("terminal set","A=",O_inf.A,"b=",O_inf.b)

        x_bar=cp.Variable((self.nx,self.N+1), name="x_bar")
        v=cp.Variable((self.nu,self.N), name="v")
        x0_var=cp.Parameter(self.nx,name="x0_var")
        constraints = []
        constraints.append(E.A @ (x0_var - x_bar[:,0]) <= E.b)
        constraints.append(x_bar[:,1:] == self.A @ x_bar[:,:-1] + self.B @ v)
        constraints.append(X_tightened.A @ x_bar[:,:-1] <= X_tightened.b)
        constraints.append(U_tightened.A @ v <= U_tightened.b.reshape(-1,1))
        constraints.append(O_inf.A @ x_bar[:,-1] <= O_inf.b)
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(x_bar[:,i], Q)
            cost += cp.quad_form(v[:,i], R)
        cost += cp.quad_form(x_bar[:,-1], Qf)
        self._x0_var = x0_var
        self._v=v
        self._x_bar=x_bar
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx0 = x0 - self.xs

        self._x0_var.value = dx0

        self.ocp.solve(solver=cp.PIQP, warm_start=True, verbose=False)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            u0 = self.us.copy()
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        v0 = self._v.value[:, 0]
        x_bar_0=self._x_bar.value[:,0]
        u0 = self.us + self._K@(dx0-x_bar_0)+v0

        X_pred = self._x_bar.value
        U_pred = self._v.value
        x_traj = X_pred + self.xs.reshape(-1, 1)
        u_traj = U_pred + self.us.reshape(-1, 1)

        return u0, x_traj, u_traj
