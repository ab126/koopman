import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import casadi as ca, do_mpc

from src.utils import gauss_process

# ----------------------------
# Dynamics (nonlinear)
# ----------------------------
def sample_F(t, y):
    # Example: simple stabilizing feedback 
    k_theta = 20
    k_theta_dot = 5
    return -k_theta * y[2] - k_theta_dot * y[3]

def wrap_u_caller_as_F_caller(u_caller, m, g):
    """Wraps a normalized-input controller so callers can provide/receive force."""
    def F(t, y):
        return u_caller(t, y) * (m * g)
    return F

def wrap_F_caller_as_u_caller(F, m, g):
    """Wraps a force-based controller so the normalized dynamics can use it."""
    def u_caller(t, y):
        return F(t, y) / (m * g)
    return u_caller

def _physical_state_scale(m, g, l):
    t0 = np.sqrt(l / g)
    return t0, np.array([l, l / t0, 1.0, 1.0 / t0], dtype=float), m * g

def wrap_physical_F_caller_as_u_caller(F, m, g, l):
    """Wraps a physical force callback F(t_phys, y_phys) for normalized dynamics."""
    t0, state_scale, mg = _physical_state_scale(m, g, l)

    def u_caller(t, y):
        y = np.asarray(y, dtype=float).reshape(4)
        y_phys = y * state_scale
        t_phys = float(t) * t0
        return F(t_phys, y_phys) / mg

    return u_caller

def wrap_u_caller_as_physical_F_caller(u_caller, m, g, l):
    """Wraps a normalized-input controller as a physical force callback."""
    t0, state_scale, mg = _physical_state_scale(m, g, l)

    def F(t, y):
        y = np.asarray(y, dtype=float).reshape(4)
        y_norm = y / state_scale
        t_norm = float(t) / t0
        return u_caller(t_norm, y_norm) * mg

    return F

def dynamics(t, y, F, M, m, g, l):
    """Compatibility wrapper around the dimensionless dynamics core."""
    t0, state_scale, mg = _physical_state_scale(m, g, l)
    y = np.asarray(y, dtype=float).reshape(4)
    y_norm = y / state_scale
    ydot_norm = dynamics_u(t / t0, y_norm, wrap_physical_F_caller_as_u_caller(F, m, g, l), M / m)
    return ydot_norm * (state_scale / t0)

def dynamics_u(t, y, u_caller, M):
    """Continuous-time normalized dynamics driven by a normalized input caller."""
    y = np.asarray(y, dtype=float).reshape(4)
    u = float(u_caller(t, y))
    return dynamics_open_loop(y, u, M)

def dynamics_open_loop(y, u, M):
    """Continuous-time normalized dynamics with direct normalized input u."""
    x_pos, x_dot, theta, theta_dot = y

    # Mass matrix components
    D11 = M + 1
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

# ----------------------------
# Simulation
# ----------------------------
def _sample_piecewise_constant_control(call_log, t_eval):
    """Samples the last applied control value at each requested output time."""
    if not call_log:
        return np.zeros_like(t_eval, dtype=float)

    samples = sorted((float(t), float(u)) for t, u in call_log)
    times = np.array([item[0] for item in samples], dtype=float)
    values = np.array([item[1] for item in samples], dtype=float)

    indices = np.searchsorted(times, np.asarray(t_eval, dtype=float), side="right") - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]

def simulate_u(u_caller, M, y0=None, t_span=(0, 10), num_points=500):
    """Simulates the normalized dynamics given a normalized control policy u(t, y)."""
    t_eval = np.linspace(*t_span, num_points)

    if y0 is None:
        # Initial condition: [x, xdot, theta, thetadot]
        y0 = [0.0, 0.0, 0.03, 0.0]  # In normalized units m=g=l=1

    control_calls = []

    def logged_u_caller(t, y):
        u = float(u_caller(t, y))
        control_calls.append((float(t), u))
        return u

    dyn_caller = lambda t, y: dynamics_u(t, y, u_caller=logged_u_caller, M=M)
    sol = solve_ivp(dyn_caller, t_span, y0, t_eval=t_eval)

    # Extract
    t = sol.t
    x = sol.y[0]
    x_dot = sol.y[1]
    theta = sol.y[2]
    theta_dot = sol.y[3]

    u = _sample_piecewise_constant_control(control_calls, t)

    return t, x, x_dot, theta, theta_dot, u

def simulate(F, M, m, g, l, y0=None, t_span=(0, 10), num_points=500):
    """Simulates the system dynamics given a force policy F(t, y)."""

    t0, state_scale, mg = _physical_state_scale(m, g, l)
    t_span = (t_span[0] / t0, t_span[1] / t0)  # Normalize time span
    u_caller = wrap_physical_F_caller_as_u_caller(F, m, g, l)
    if y0 is not None:
        y0 = np.asarray(y0, dtype=float).reshape(4) / state_scale  # Normalize initial state
    t, x, x_dot, theta, theta_dot, u = simulate_u(u_caller, M/m, y0=y0, t_span=t_span, num_points=num_points)
    return t*t0, x*state_scale[0], x_dot*state_scale[1], theta*state_scale[2], theta_dot*state_scale[3], u*mg

def rk4_step(y, u, dt, M):
    """One RK4 step of the normalized nonlinear dynamics."""
    y = np.asarray(y, dtype=float).reshape(4)
    u = float(u)

    k1 = dynamics_open_loop(y, u, M)
    k2 = dynamics_open_loop(y + 0.5 * dt * k1, u, M)
    k3 = dynamics_open_loop(y + 0.5 * dt * k2, u, M)
    k4 = dynamics_open_loop(y + dt * k3, u, M)

    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rollout_nonlinear_dynamics(y0, u_seq, dt, M):
    """Rolls out the nonlinear system over a piecewise-constant input sequence."""
    y = np.asarray(y0, dtype=float).reshape(4)
    X = np.zeros((4, len(u_seq) + 1))
    X[:, 0] = y

    for k, u in enumerate(u_seq):
        y = rk4_step(y, u, dt, M)
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

def inv_lift(z):
    # Inverse of the lift function (only returns original state variables)
    return z[:4]

def build_dataset(X, u):
    Z = np.array([lift(X[:, i]) for i in range(X.shape[1])]).T

    Zk = Z[:, :-1]
    Zkp1 = Z[:, 1:]
    u = np.asarray(u, dtype=float).reshape(-1)
    if u.size == Z.shape[1]:
        Uk = u[:-1].reshape(1, -1)
    elif u.size == Z.shape[1] - 1:
        Uk = u.reshape(1, -1)
    else:
        raise ValueError("u must have either the same number of samples as X or one fewer sample.")

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

def identify_sys_u(x, x_dot, theta, theta_dot, u, t=None, model_type="continuous"):
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

def identify_sys(x, x_dot, theta, theta_dot, F, M, m, g, l, t=None, model_type="continuous"):
    """Identifies lifted system matrices from state/force trajectories."""
    t0 = np.sqrt(l / g)
    u = F / (m * g)  # Assuming normalized input
    return identify_sys_u(x/l, x_dot/(l/t0), theta, theta_dot/(1/t0), u, t=t/t0, model_type=model_type)

def gen_small_theta_data(M, m, g, l, sigma=0.5, theta_max=0.15, t_span=(0, 10), num_points=500, n_repeats=10):
    """Generates state/input trajectories for small theta."""

    t_lin = np.linspace(*t_span, num_points)
    t_all = []
    X_all = []
    F_all = []

    def first_greater(arr, value):
        idx = np.where(arr > value)[0]
        return idx[0] if len(idx) > 0 else -1

    for _ in range(n_repeats):
        all_F = gauss_process(t_lin, sigma=sigma*m*g)

        def gauss_F(t_val, y):
            def closest_index(arr, val):
                return min(range(len(arr)), key=lambda i: abs(arr[i] - val))    
            return all_F[closest_index(t_lin, t_val)]
        
        x0 = [0.0, 0.0, np.random.uniform(-theta_max/2, theta_max/2), 0.0]

        t, x, x_dot, theta, theta_dot, F = simulate(gauss_F, M, m, g, l, y0=x0, t_span=t_span, num_points=num_points)
        ind = first_greater(np.abs(theta), theta_max)
        if ind > 0:
            t = t[:ind]
            x = x[:ind]
            x_dot = x_dot[:ind]
            theta = theta[:ind]
            theta_dot = theta_dot[:ind]
            F = F[:ind]
        
        t_all.append(t)
        X_all.append(np.column_stack((x, x_dot, theta, theta_dot)).T)
        F_all.append(F)
    
    return t_all, X_all, F_all

def identify_sys_multiple_trajectories_u(t_all, X_all, u_all, model_type="continuous", lift=lift):
    """Identifies lifted system matrices from multiple trajectories."""
    Z_blocks = []
    dZdt_blocks = []
    Zk_blocks = []
    Zkp1_blocks = []
    Uk_blocks = []

    for X, u, t in zip(X_all, u_all, t_all):
        Z = np.array([lift(X[:, i]) for i in range(X.shape[1])]).T

        if model_type == "continuous":
            dZdt_blocks.append(finite_difference(Z, t))
            Z_blocks.append(Z)
            Uk_blocks.append(np.asarray(u, dtype=float).reshape(1, -1))
        elif model_type == "discrete":
            u = np.asarray(u, dtype=float).reshape(-1)
            if u.size == Z.shape[1]:
                Uk = u[:-1].reshape(1, -1)
            elif u.size == Z.shape[1] - 1:
                Uk = u.reshape(1, -1)
            else:
                raise ValueError("u must have either the same number of samples as X or one fewer sample.")
            Zk_blocks.append(Z[:, :-1])
            Zkp1_blocks.append(Z[:, 1:])
            Uk_blocks.append(Uk)
        else:
            raise ValueError("model_type must be either 'continuous' or 'discrete'.")

    if model_type == "continuous":
        Z = np.hstack(Z_blocks)
        dZdt = np.hstack(dZdt_blocks)
        U = np.hstack(Uk_blocks)
        K = dZdt @ np.linalg.pinv(np.vstack((Z, U)))
        n = Z.shape[0]
        A = K[:, :n]
        B = K[:, n:]
    else:
        Zk = np.hstack(Zk_blocks)
        Zkp1 = np.hstack(Zkp1_blocks)
        U = np.hstack(Uk_blocks)
        A, B = koopman_identification(Zk, Zkp1, U)

    # Controllability
    rank_C, ctrl_flag = is_controllable(A, B)
    print("Controllability rank:", rank_C, "Full:", ctrl_flag)

    # PE check
    L = 20
    rank_PE, pe_flag = is_persistently_exciting(np.hstack(Uk_blocks).ravel(), L)
    print("PE rank:", rank_PE, "Full:", pe_flag)

    return A, B

def identify_sys_multiple_trajectories(t_all, X_all, F_all, M, m, g, l, model_type="continuous", lift=lift):
    """Identifies lifted system matrices from multiple trajectories."""
    t0 = np.sqrt(l / g)
    u_all = [F / (m * g) for F in F_all]  # Assuming normalized input
    t_all = [t / t0 for t in t_all]  # Normalize time
    X_all = [X / np.array([[l], [l/t0], [1], [1/t0]]) for X in X_all]  # Normalize states
    return identify_sys_multiple_trajectories_u(t_all, X_all, u_all, model_type=model_type, lift=lift)

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

def linearize_upright_dynamics(M):
    """Returns the continuous-time 4-state linearization about [x, xdot, theta, thetadot] = 0."""
    alpha = M + 1.0
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

def lqr_4state_u_caller(M, Q=None, R=None, umax=10, y_ref=None, u_ref=0.0):
    """Builds a normalized-input LQR controller on the original 4-state linearization."""
    A, B = linearize_upright_dynamics(M)

    if Q is None:
        Q = np.diag([1.0, 1.0, 50.0, 10.0])
    if R is None:
        R = np.array([[0.1]])
    if y_ref is None:
        y_ref = np.zeros(4)

    K = lqr_ct(A, B, Q, R)
    y_ref = np.asarray(y_ref, dtype=float).reshape(4)

    def u_caller(_, y):
        y = np.asarray(y, dtype=float).reshape(4)
        y_err = y - y_ref
        u = u_ref - (K @ y_err).item()
        return np.clip(u, -umax, umax)

    return u_caller, K, A, B

def lqr_4state_F_caller(M, m, g, l=None, Q=None, R=None, umax=10, y_ref=None, u_ref=0.0):
    """Builds a force-based wrapper around the normalized 4-state LQR controller."""
    u_caller, K, A, B = lqr_4state_u_caller(
        M=M/m,
        Q=Q,
        R=R,
        umax=umax,
        y_ref=y_ref,
        u_ref=u_ref,
    )
    F_caller = wrap_u_caller_as_F_caller(u_caller, m, g) if l is None else wrap_u_caller_as_physical_F_caller(u_caller, m, g, l)
    return F_caller, K, A, B

def solve_nonlinear_mpc(y0, M, Q=None, R=None, Qf=None, horizon=25, dt=0.05,
                        umax=10.0, y_ref=None, u_ref=0.0, u_guess=None,
                        theta_wrap=True, rate_penalty=None):
    """Solves a direct-shooting nonlinear MPC problem in normalized coordinates."""
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
    u_bound = float(umax)

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
        X = rollout_nonlinear_dynamics(y0, u_seq, dt, M)
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

    X_opt = rollout_nonlinear_dynamics(y0, u_opt, dt, M)
    return u_opt, X_opt, result

def nonlinear_mpc_u_caller(M, Q=None, R=None, Qf=None, horizon=25, dt=0.05,
                           umax=10.0, y_ref=None, u_ref=0.0, theta_wrap=True,
                           rate_penalty=0.1):
    """Builds a receding-horizon MPC controller that operates entirely on normalized input u."""
    if y_ref is None:
        y_ref = np.zeros(4)

    controller_state = {
        "next_update_t": None,
        "u_seq": np.zeros(horizon),
        "current_u": 0.0,
    }

    def u_caller(t, y):
        y = np.asarray(y, dtype=float).reshape(4)

        if controller_state["next_update_t"] is None or t >= controller_state["next_update_t"] - 1e-12:
            u_guess = controller_state["u_seq"]
            if u_guess.size != horizon:
                u_guess = np.zeros(horizon)

            u_opt, _, _ = solve_nonlinear_mpc(
                y0=y,
                M=M,
                Q=Q,
                R=R,
                Qf=Qf,
                horizon=horizon,
                dt=dt,
                umax=umax,
                y_ref=y_ref,
                u_ref=u_ref,
                u_guess=u_guess,
                theta_wrap=theta_wrap,
                rate_penalty=rate_penalty,
            )

            controller_state["current_u"] = float(u_opt[0])
            controller_state["u_seq"] = np.concatenate((u_opt[1:], u_opt[-1:]))
            controller_state["next_update_t"] = t + dt

        return controller_state["current_u"]

    return u_caller

def nonlinear_mpc_F_caller(M, m, g, l, Q=None, R=None, Qf=None, horizon=25, dt=0.05,
                           umax=10.0, y_ref=None, u_ref=0.0, theta_wrap=True,
                           rate_penalty=0.1):
    """Builds a force-based wrapper around the normalized nonlinear MPC controller."""
    
    t0, state_scale, mg = _physical_state_scale(m, g, l)
    u_caller = nonlinear_mpc_u_caller(
        M=M/m,
        Q=Q,
        R=R,
        Qf=Qf,
        horizon=horizon,
        dt=dt/t0,
        umax=umax,
        y_ref=y_ref / state_scale,
        u_ref=u_ref,
        theta_wrap=theta_wrap,
        rate_penalty=rate_penalty,
    )
    return wrap_u_caller_as_physical_F_caller(u_caller, m, g, l)

def module_mpc_F_caller(M, m, g, l, Q=None, R=None, model_type='continuous', horizon=20, dt=0.05, umax=10, y_ref=None, u_ref=0.0, rate_penalty=0.1):
    """Builds a force-based wrapper around the normalized module do-mpc library controller."""
    t0, state_scale, mg = _physical_state_scale(m, g, l)
    u_caller = module_mpc_u_caller(
        M=M/m,
        Q=Q,
        R=R,
        horizon=horizon,
        dt=dt/t0,
        umax=umax,
        y_ref=y_ref / state_scale if y_ref is not None else None,
        u_ref=u_ref,
        rate_penalty=rate_penalty,
    )
    return wrap_u_caller_as_physical_F_caller(u_caller, m, g, l)

def module_mpc_u_caller(M, Q=None, R=None, model_type='continuous', horizon=20, dt=0.05, umax=10, y_ref=None, u_ref=0.0, rate_penalty=0.1):
    """ Builds a receding-horizon MPC controller from do-mpc library that operates entirely on normalized input u. """

    # Define Model
    model = do_mpc.model.Model(model_type)

    # States
    x = model.set_variable(var_type='_x', var_name='x')
    x_dot = model.set_variable(var_type='_x', var_name='x_dot')
    theta = model.set_variable(var_type='_x', var_name='theta')
    theta_dot = model.set_variable(var_type='_x', var_name='theta_dot')

    # Input
    u = model.set_variable(var_type='_u', var_name='u')

    # Example nonlinear dynamics TODO
    delta = 1/3 * (M + 1) - 0.25 * np.cos(theta)**2
    model.set_rhs('x', x_dot)
    model.set_rhs('x_dot', (1/3*u + 0.6*ca.sin(theta)*theta_dot**2 - 0.25*ca.sin(theta)*ca.cos(theta)) / delta)
    model.set_rhs('theta', theta_dot)
    model.set_rhs('theta_dot', (-0.5*u*ca.cos(theta) - 0.25*ca.sin(theta)*ca.cos(theta)*theta_dot**2 + 0.5*(M+1)*ca.sin(theta)) / delta)

    model.setup()

    # Create MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': horizon,
        't_step': dt,
        'state_discretization': 'collocation',
        'store_full_solution': True,
        'nlpsol_opts': {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes'   # suppress IPOPT banner
        }
    }

    mpc.set_param(**setup_mpc)

    # Cost Function
    x_ref = y_ref if y_ref is not None else np.zeros(4)

    x = model.x['x']
    x_dot = model.x['x_dot']
    theta = model.x['theta']
    theta_dot = model.x['theta_dot']
    u = model.u['u']

    x_vars = [x, x_dot, theta, theta_dot]
    u_vars = [u]

    lterm, mterm = quadratic_tracking_cost(x_vars, u_vars, Q, R, x_ref=y_ref, u_ref=u_ref)
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=rate_penalty)  # penalize input changes

    # Constraints
    mpc.bounds['lower','_u','u'] = -umax
    mpc.bounds['upper','_u','u'] = umax

    mpc.setup()

    # Initialize
    mpc.set_initial_guess()

    return lambda t, y: mpc.make_step(y).flatten()[0]

def quadratic_cost_from_QR(x_vars, u_vars, Q, R):
    """
    Build CasADi expressions for MPC cost from Q and R matrices.

    Parameters
    ----------
    x_vars : list of CasADi variables (states)
    u_vars : list of CasADi variables (inputs)
    Q : numpy array (nx x nx)
    R : numpy array (nu x nu)

    Returns
    -------
    lterm : CasADi expression (stage cost)
    mterm : CasADi expression (terminal cost)
    """

    # Stack variables into vectors
    x = ca.vertcat(*x_vars)
    u = ca.vertcat(*u_vars)

    # Convert Q, R to CasADi
    Q_ca = ca.DM(Q)
    R_ca = ca.DM(R)

    # Quadratic forms
    x_cost = ca.mtimes([x.T, Q_ca, x])
    u_cost = ca.mtimes([u.T, R_ca, u])

    lterm = x_cost + u_cost
    mterm = x_cost  # standard choice

    return lterm, mterm

def quadratic_tracking_cost(x_vars, u_vars, Q, R, x_ref=None, u_ref=None):
    """
    Build CasADi expressions for MPC cost from Q and R matrices.

    Parameters
    ----------
    x_vars : list of CasADi variables (states)
    u_vars : list of CasADi variables (inputs)
    Q : numpy array (nx x nx)
    R : numpy array (nu x nu)
    x_ref : numpy array (nx,) or None (reference state)
    u_ref : numpy array (nu,) or None (reference input)

    Returns
    -------
    lterm : CasADi expression (stage cost)
    mterm : CasADi expression (terminal cost)
    """

    x = ca.vertcat(*x_vars)
    u = ca.vertcat(*u_vars)

    Q_ca = ca.DM(Q)
    R_ca = ca.DM(R)

    if x_ref is not None:
        x_ref = ca.DM(x_ref)
        x_err = x - x_ref
    else:
        x_err = x

    if u_ref is not None:
        u_ref = ca.DM(u_ref)
        u_err = u - u_ref
    else:
        u_err = u

    x_cost = ca.mtimes([x_err.T, Q_ca, x_err])
    u_cost = ca.mtimes([u_err.T, R_ca, u_err])

    lterm = x_cost + u_cost
    mterm = x_cost

    return lterm, mterm

def lqr_u_caller(A, B, umax=10, y_ref=None, u_ref=0.0, Q_phys=None, R=None, model_type="continuous"):
    """Builds a normalized-input LQR controller on the lifted system."""
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

    Q_z = C.T @ Q_phys @ C # TODO: Check this, C doesnt look right

    if model_type == "continuous":
        K = lqr_ct(A, B, Q_z, R)
    elif model_type == "discrete":
        K = lqr(A, B, Q_z, R)
    else:
        raise ValueError("model_type must be either 'continuous' or 'discrete'.")

    z_ref = lift(y_ref)

    def u_caller(_, y):
        z_err = lift(y) - z_ref
        u = u_ref - (K @ z_err).item()
        return np.clip(u, -umax, umax)

    return u_caller

def lqr_F_caller(A, B, m, g, l=None, umax=10, y_ref=None, u_ref=0.0, Q_phys=None, R=None, model_type="continuous"):
    """Builds a force-based wrapper around the normalized lifted-system LQR controller."""
    u_caller = lqr_u_caller(
        A=A,
        B=B,
        umax=umax,
        y_ref=y_ref,
        u_ref=u_ref,
        Q_phys=Q_phys,
        R=R,
        model_type=model_type,
    )
    return wrap_u_caller_as_F_caller(u_caller, m, g) if l is None else wrap_u_caller_as_physical_F_caller(u_caller, m, g, l)



# TODO add DeePC control design as well

