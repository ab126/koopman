import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ----------------------------
# Dynamics (nonlinear)
# ----------------------------
def sample_F(t, y):
    # Example: simple stabilizing feedback 
    k_theta = 20
    k_theta_dot = 5
    return -k_theta * y[2] - k_theta_dot * y[3]

def dynamics(t, y, F, M, m, g, l):
    x_pos, x_dot, theta, theta_dot = y # In normalized units

    u = F(t, y) / (m * g)
    return dynamics_open_loop(y, u, M, m)

def dynamics_open_loop(y, u, M, m):
    """Continuous-time normalized dynamics with direct normalized input u."""
    x_pos, x_dot, theta, theta_dot = y

    # Mass matrix components
    D11 = M/m + 1
    D12 = 0.5 * np.cos(theta)
    D21 = D12
    D22 = (1/3) 

    # RHS
    RHS1 = u + 0.5 * np.sin(theta) *  theta_dot**2
    RHS2 = 0.5 * np.sin(theta)

    # Solve linear system for accelerations
    D = np.array([[D11, D12],
                  [D21, D22]])
    RHS = np.array([RHS1, RHS2])

    dd = np.linalg.solve(D, RHS)
    x_ddot = dd[0]
    theta_ddot = dd[1]

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)

def get_si_values(y, m, g, l):
    t_0 = np.sqrt(l / g)
    y_1, y_2, y_3, y_4 = y
    return [y_1 * l, y_2 * l / t_0, y_3, y_4 / t_0]

# ----------------------------
# Simulation
# ----------------------------
def simulate(F, M, m, g, l, y0=None, t_span=(0, 10), num_points=500):
    """Simulates the system dynamics given a control policy F(t, y)."""
    t_eval = np.linspace(*t_span, num_points)

    if y0 is None:
        # Initial condition: [x, xdot, theta, thetadot]
        y0 = [0.0, 0.0, 0.03, 0.0]  # In normalized units m=g=l=1

    dyn_caller = lambda t, y: dynamics(t, y, F=F, M=M, m=m, g=g, l=l)
    sol = solve_ivp(dyn_caller, t_span, y0, t_eval=t_eval)

    # Extract
    t = sol.t
    x = sol.y[0]
    x_dot = sol.y[1]
    theta = sol.y[2]
    theta_dot = sol.y[3]

    # Compute input over time
    u = np.array([F(t[i], sol.y[:, i]) / (m * g) for i in range(len(t))])

    return t, x, x_dot, theta, theta_dot, u

def rk4_step(y, u, dt, M, m):
    """One RK4 step of the normalized nonlinear dynamics."""
    y = np.asarray(y, dtype=float).reshape(4)
    u = float(u)

    k1 = dynamics_open_loop(y, u, M, m)
    k2 = dynamics_open_loop(y + 0.5 * dt * k1, u, M, m)
    k3 = dynamics_open_loop(y + 0.5 * dt * k2, u, M, m)
    k4 = dynamics_open_loop(y + dt * k3, u, M, m)

    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rollout_nonlinear_dynamics(y0, u_seq, dt, M, m):
    """Rolls out the nonlinear system over a piecewise-constant input sequence."""
    y = np.asarray(y0, dtype=float).reshape(4)
    X = np.zeros((4, len(u_seq) + 1))
    X[:, 0] = y

    for k, u in enumerate(u_seq):
        y = rk4_step(y, u, dt, M, m)
        X[:, k + 1] = y

    return X


def simulate_lin_sys(A, B, x0, u_caller, t_span=(0, 10), num_points=500):
    """Simulates the continuous-time linear system dx/dt = A x + B u."""
    t_eval = np.linspace(*t_span, num_points)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    B = np.asarray(B, dtype=float)

    def lin_dynamics(t, x):
        u = u_caller(t, x)
        u = np.asarray(u, dtype=float).reshape(-1)

        ax = np.asarray(A @ x, dtype=float).reshape(-1)
        bu = np.asarray(B @ u, dtype=float).reshape(-1)

        return ax + bu

    sol = solve_ivp(lin_dynamics, t_span, x0, t_eval=t_eval)

    return sol.t, sol.y

def simulate_discrete_lin_sys(A, B, x0, u_caller, num_steps):
    """Simulates the discrete-time linear system x_{k+1} = A x_k + B u_k."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    X = np.zeros((x0.size, num_steps + 1))
    X[:, 0] = x0

    for k in range(num_steps):
        u = np.asarray(u_caller(k, X[:, k]), dtype=float).reshape(-1)
        X[:, k + 1] = np.asarray(A @ X[:, k], dtype=float).reshape(-1) + np.asarray(B @ u, dtype=float).reshape(-1)

    return np.arange(num_steps + 1), X

def inv_lift(z):
    # Inverse of the lift function (only returns original state variables)
    return z[:4]

# ----------------------------
# Linearization
# ----------------------------
def lift(y):
    y1, y2, y3, y4 = y
    return np.array([
        y1,
        y2,
        y3,
        y4,
        np.sin(y3),
        np.cos(y3),
        #y4**2, # Gives rank deficient controllability matrix
        y2 * y4
    ])

def build_dataset(X, u):
    Z = np.array([lift(X[:, i]) for i in range(X.shape[1])]).T

    Zk = Z[:, :-1]
    Zkp1 = Z[:, 1:]
    Uk = u[:-1].reshape(1, -1)

    return Zk, Zkp1, Uk

def koopman_identification(Zk, Zkp1, Uk):
    """Fits the discrete-time lifted model z_{k+1} = A z_k + B u_k."""
    # stack data
    W = np.vstack((Zk, Uk))

    # least squares
    K = Zkp1 @ np.linalg.pinv(W)

    n = Zk.shape[0]

    A = K[:, :n]
    B = K[:, n:]

    return A, B

def finite_difference(Z, t):
    """Estimates dZ/dt column-wise using the sample times t."""
    t = np.asarray(t, dtype=float).reshape(-1)
    if Z.shape[1] != t.size:
        raise ValueError("Z and t must contain the same number of samples.")

    dZdt = np.gradient(Z, t, axis=1, edge_order=2)
    return dZdt

def koopman_identification_ct(Z, u, t):
    """Fits the continuous-time lifted model dz/dt = A z + B u."""
    dZdt = finite_difference(Z, t)
    W = np.vstack((Z, np.asarray(u, dtype=float).reshape(1, -1)))
    K = dZdt @ np.linalg.pinv(W)

    n = Z.shape[0]
    A = K[:, :n]
    B = K[:, n:]

    return A, B

# ----------------------------
# Identifiability
# ----------------------------
def controllability_matrix(A, B):
    n = A.shape[0]
    R_C = B
    for i in range(1, n):
        R_C = np.hstack((R_C, np.linalg.matrix_power(A, i) @ B))
    return R_C

def is_controllable(A, B):
    R_C = controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(R_C)
    return rank, (rank == A.shape[0])

def hankel_matrix(u, L):
    N = len(u)
    H = np.array([u[i:i+L] for i in range(N - L + 1)]).T
    return H

def is_persistently_exciting(u, L):
    H = hankel_matrix(u, L)
    rank = np.linalg.matrix_rank(H)
    return rank, (rank == L)

def identify_sys(x, x_dot, theta, theta_dot, u, t=None, model_type="continuous"):
    """Identifies lifted system matrices from state/input trajectories.

    model_type="continuous" fits dz/dt = A z + B u.
    model_type="discrete" fits z_{k+1} = A z_k + B u_k.
    """
    X = np.column_stack((x, x_dot, theta, theta_dot)).T
    Z = np.array([lift(X[:, i]) for i in range(X.shape[1])]).T

    if model_type == "continuous":
        if t is None:
            raise ValueError("Time vector t is required for continuous-time identification.")
        A, B = koopman_identification_ct(Z, u, t)
    elif model_type == "discrete":
        Zk, Zkp1, Uk = build_dataset(X, u)
        A, B = koopman_identification(Zk, Zkp1, Uk)
    else:
        raise ValueError("model_type must be either 'continuous' or 'discrete'.")

    # Controllability
    rank_C, ctrl_flag = is_controllable(A, B)
    print("Controllability rank:", rank_C, "Full:", ctrl_flag)

    # PE check
    L = 20
    rank_PE, pe_flag = is_persistently_exciting(u, L)
    print("PE rank:", rank_PE, "Full:", pe_flag)

    return A, B

# ----------------------------
# Control
# ----------------------------
def lqr(A, B, Q, R):
    """Computes the infinite-horizon LQR gain matrix K for discrete-time system."""
    from scipy.linalg import solve_discrete_are

    # Solve the discrete-time algebraic Riccati equation
    P = solve_discrete_are(A, B, Q, R)

    # Compute the LQR gain
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    return K

def lqr_ct(A, B, Q, R):
    """Computes the infinite-horizon LQR gain matrix K for continuous-time system."""
    from scipy.linalg import solve_continuous_are

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    return K

def linearize_upright_dynamics(M, m):
    """Returns the continuous-time 4-state linearization about [x, xdot, theta, thetadot] = 0."""
    alpha = M / m + 1.0
    det = alpha / 3.0 - 0.25

    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -0.25 / det, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5 * alpha / det, 0.0],
    ])
    B = np.array([
        [0.0],
        [1.0 / (3.0 * det)],
        [0.0],
        [-0.5 / det],
    ])

    return A, B

def lqr_4state_F_caller(M, m, g, Q=None, R=None, umax=10, y_ref=None, u_ref=0.0):
    """Builds a continuous-time LQR controller on the original 4-state linearization."""
    A, B = linearize_upright_dynamics(M, m)

    if Q is None:
        Q = np.diag([1.0, 1.0, 50.0, 10.0])
    if R is None:
        R = np.array([[0.1]])
    if y_ref is None:
        y_ref = np.zeros(4)

    K = lqr_ct(A, B, Q, R)
    y_ref = np.asarray(y_ref, dtype=float).reshape(4)

    def F(_, y):
        y = np.asarray(y, dtype=float).reshape(4)
        y_err = y - y_ref
        u = u_ref - (K @ y_err).item()
        return np.clip(u, -umax, umax) * (m * g)

    return F, K, A, B

def solve_nonlinear_mpc(y0, M, m, g, Q=None, R=None, Qf=None, horizon=25, dt=0.05,
                        umax=10.0, y_ref=None, u_ref=0.0, u_guess=None,
                        theta_wrap=True, rate_penalty=None):
    """Solves a direct-shooting nonlinear MPC problem on the 4-state system."""
    if Q is None:
        Q = np.diag([1.0, 1.0, 80.0, 12.0])
    if R is None:
        R = np.array([[0.1]])
    if Qf is None:
        Qf = 5.0 * Q
    if y_ref is None:
        y_ref = np.zeros(4)

    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    Qf = np.asarray(Qf, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float).reshape(4)
    u_ref = float(u_ref)
    u_bound = float(umax) / float(m * g)

    if u_guess is None:
        u_guess = np.zeros(horizon)
    else:
        u_guess = np.asarray(u_guess, dtype=float).reshape(horizon)

    def state_error(y):
        err = np.asarray(y, dtype=float).reshape(4) - y_ref
        if theta_wrap:
            err[2] = np.arctan2(np.sin(err[2]), np.cos(err[2]))
        return err

    def objective(u_seq):
        X = rollout_nonlinear_dynamics(y0, u_seq, dt, M, m)
        cost = 0.0

        for k in range(horizon):
            err = state_error(X[:, k])
            du = u_seq[k] - u_ref
            cost += err @ Q @ err + R[0, 0] * du * du

            if rate_penalty is not None and k > 0:
                delta_u = u_seq[k] - u_seq[k - 1]
                cost += rate_penalty * delta_u * delta_u

        terminal_err = state_error(X[:, -1])
        cost += terminal_err @ Qf @ terminal_err
        return float(cost)

    bounds = [(-u_bound, u_bound)] * horizon
    result = minimize(objective, u_guess, method="SLSQP", bounds=bounds)

    if not result.success:
        u_opt = u_guess
    else:
        u_opt = result.x

    X_opt = rollout_nonlinear_dynamics(y0, u_opt, dt, M, m)
    return u_opt, X_opt, result

def nonlinear_mpc_F_caller(M, m, g, l, Q=None, R=None, Qf=None, horizon=25, dt=0.05,
                           umax=10.0, y_ref=None, u_ref=0.0, theta_wrap=True,
                           rate_penalty=0.1):
    """Builds a receding-horizon nonlinear MPC controller with sample-and-hold updates."""
    if y_ref is None:
        y_ref = np.zeros(4)

    controller_state = {
        "next_update_t": None,
        "u_seq": np.zeros(horizon),
        "current_u": 0.0,
    }

    def F(t, y):
        y = np.asarray(y, dtype=float).reshape(4)

        if controller_state["next_update_t"] is None or t >= controller_state["next_update_t"] - 1e-12:
            u_guess = controller_state["u_seq"]
            if u_guess.size != horizon:
                u_guess = np.zeros(horizon)

            u_opt, _, _ = solve_nonlinear_mpc(
                y0=y,
                M=M,
                m=m,
                g=g,
                Q=Q,
                R=R,
                Qf=Qf,
                horizon=horizon,
                dt=dt,
                umax=umax,
                y_ref=y_ref,
                u_ref=u_ref / (m * g),
                u_guess=u_guess,
                theta_wrap=theta_wrap,
                rate_penalty=rate_penalty,
            )

            controller_state["current_u"] = float(u_opt[0]) * (m * g)
            controller_state["u_seq"] = np.concatenate((u_opt[1:], u_opt[-1:]))
            controller_state["next_update_t"] = t + dt

        return controller_state["current_u"]

    return F

def lqr_F_caller(A, B, m, g, umax=10, y_ref=None, u_ref=0.0, Q_phys=None, R=None, model_type="continuous"):
    C = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # x
        [0, 1, 0, 0, 0, 0, 0],   # xdot
        [0, 0, 1, 0, 0, 0, 0],   # theta
        [0, 0, 0, 1, 0, 0, 0],   # thetadot
    ])
    if y_ref is None:
        y_ref = np.zeros(4)
    if Q_phys is None:
        Q_phys = np.diag([0, 10, 50, 10])
    if R is None:
        R = np.array([[1.0]])

    Q_z = C.T @ Q_phys @ C

    if model_type == "continuous":
        K = lqr_ct(A, B, Q_z, R)
    elif model_type == "discrete":
        K = lqr(A, B, Q_z, R)
    else:
        raise ValueError("model_type must be either 'continuous' or 'discrete'.")

    z_ref = lift(y_ref)

    def F(_, y):
        z_err = lift(y) - z_ref
        u = u_ref - (K @ z_err).item()
        return np.clip(u, -umax, umax) * (m * g)  # Scale back to physical units

    return F

# TODO add DeePC control design as well

