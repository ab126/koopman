import numpy as np
from tqdm import tqdm
from typing import Sequence

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_are

from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint


from src.utils import gauss_process

# ----------------------------
# Dynamics (nonlinear)
# ----------------------------
def sample_F(t, y):
    # Example: simple stabilizing feedback 
    k_theta = 20
    k_theta_dot = 5
    return -k_theta * y[2] - k_theta_dot * y[3]

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

def simulate_u(u_caller, M, y0=None, t_span=(0, 10), num_points=500, verbose=False):
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
    if verbose:
        print("Running ODE integration...")
    sol = solve_ivp(dyn_caller, t_span, y0, t_eval=t_eval)

    # Extract
    t = sol.t
    x = sol.y[0]
    x_dot = sol.y[1]
    theta = sol.y[2]
    theta_dot = sol.y[3]

    u = _sample_piecewise_constant_control(control_calls, t)

    return t, x, x_dot, theta, theta_dot, u

def simulate_u_rk4(u_caller, M, y0=None, t_span=(0, 10), num_points=500, verbose=False):
    """
    Simulates the normalized system using fixed-step RK4.
    
    Parameters
    ----------
    u_caller : callable
        Normalized input caller u(t, y)
    M : system parameter
    y0 : initial state [x, xdot, theta, thetadot]
    t_span : (t0, tf)
    dt : timestep

    Returns
    -------
    t, x, x_dot, theta, theta_dot, u
    """

    if y0 is None:
        y0 = np.array([0.0, 0.0, 0.03, 0.0], dtype=float)
    else:
        y0 = np.asarray(y0, dtype=float).reshape(4)

    t0, tf = t_span
    dt = (float(tf) - float(t0)) / num_points  # Adjust dt to fit exactly into t_span

    t_vals = np.zeros(num_points + 1)
    X = np.zeros((4, num_points + 1))
    U = np.zeros(num_points)

    X[:, 0] = y0
    t_vals[0] = t0

    y = y0.copy()
    t = t0

    loop_iter = tqdm(range(num_points), desc="RK4 Integration") if verbose else range(num_points)
    for k in loop_iter:
        # control evaluated ONCE per step
        u = float(u_caller(t, y))
        U[k] = u

        k1 = dynamics_open_loop(y, u, M)
        k2 = dynamics_open_loop(y + 0.5 * dt * k1, u, M)
        k3 = dynamics_open_loop(y + 0.5 * dt * k2, u, M)
        k4 = dynamics_open_loop(y + dt * k3, u, M)

        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt

        X[:, k+1] = y
        t_vals[k+1] = t

    return t_vals, X[0], X[1], X[2], X[3], U

def simulate(F, M, m, g, l, y0=None, t_span=(0, 10), num_points=500, method='ivp', verbose=False):
    """Simulates the system dynamics given a force policy F(t, y)."""

    t0, state_scale, mg = _physical_state_scale(m, g, l)
    t_span = (t_span[0] / t0, t_span[1] / t0)  # Normalize time span
    u_caller = wrap_physical_F_caller_as_u_caller(F, m, g, l)
    if y0 is not None:
        y0 = np.asarray(y0, dtype=float).reshape(4) / state_scale  # Normalize initial state
    if method == 'ivp':
        t, x, x_dot, theta, theta_dot, u = simulate_u(u_caller, M/m, y0=y0, t_span=t_span, num_points=num_points, verbose=verbose)
    elif method == 'rk4':
        t, x, x_dot, theta, theta_dot, u = simulate_u_rk4(u_caller, M/m, y0=y0, t_span=t_span, num_points=num_points, verbose=verbose)
    else:
        raise ValueError("method must be either 'ivp' or 'rk4'.")
    
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

def dummy_lift(y):
    return y

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

def build_dataset(X, u, lift=dummy_lift):
    """ Builds lifted datasets for discrete-time Koopman identification from state/input trajectories. """
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

def identify_sys_u(x, x_dot, theta, theta_dot, u, t=None, model_type="continuous", lift=dummy_lift):
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
        Zk, Zkp1, Uk = build_dataset(X, u, lift=lift)
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

def identify_sys(x, x_dot, theta, theta_dot, F, m=1., g=1., l=1., t=None, model_type="continuous", lift=dummy_lift):
    """Identifies lifted system matrices from state/force trajectories."""
    t0, state_scale, mg = _physical_state_scale(m, g, l)
    x = x / state_scale[0]
    x_dot = x_dot / state_scale[1]
    theta = theta / state_scale[2]
    theta_dot = theta_dot / state_scale[3]
    u = F / (mg)  # Assuming normalized input
    return identify_sys_u(x, x_dot, theta, theta_dot, u, t=t/t0, model_type=model_type, lift=lift)

def gen_max_theta_data(
    M, m, g, l, sigma=0.5, theta_max=0.15, t_span=(0, 10),
    num_points=500, n_repeats=10, verbose=True, method='rk4', umax=2.0,
    control_dt=None, seed=None,
):
    """Generate fixed-step physical trajectories with bounded normalized inputs.

    ``umax`` is the bound on normalized input ``u = F / (m g)``. The
    corresponding physical force bound is ``umax * m * g``.
    """

    t0, state_scale, mg = _physical_state_scale(m, g, l)
    if method != "rk4":
        raise ValueError("gen_max_theta_data supports RK4 only")
    duration = float(t_span[1] - t_span[0])
    if control_dt is None:
        control_dt = duration / num_points
    num_points = round(duration / control_dt)
    t_lin = t_span[0] + np.arange(num_points) * control_dt
    rng = np.random.default_rng(seed)
    t_all = []
    X_all = []
    F_all = []

    def first_greater(arr, value):
        idx = np.where(arr > value)[0]
        return idx[0] if len(idx) > 0 else -1

    loop_iter = tqdm(range(n_repeats), desc="Trajectory Generation") if verbose else range(n_repeats)
    for _ in loop_iter:
        u_samples = np.clip(rng.normal(scale=sigma, size=num_points), -umax, umax)
        all_F = u_samples * mg

        def gauss_F(t_val, y):
            index = min(max(int((t_val - t_span[0]) / control_dt), 0), num_points - 1)
            return all_F[index]
        
        x0_normalized = np.array([
            rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25),
            rng.uniform(-0.6 * theta_max, 0.6 * theta_max), rng.uniform(-0.50, 0.50),
        ])
        x0 = x0_normalized * state_scale

        t, x, x_dot, theta, theta_dot, F = simulate(gauss_F, M, m, g, l, y0=x0, t_span=t_span, num_points=num_points, method=method, verbose=False)
        ind = first_greater(np.abs(theta), theta_max)
        if ind > 0:
            t = t[:ind + 1]
            x = x[:ind + 1]
            x_dot = x_dot[:ind + 1]
            theta = theta[:ind + 1]
            theta_dot = theta_dot[:ind + 1]
            F = F[:ind]
        
        t_all.append(t)
        X_all.append(np.column_stack((x, x_dot, theta, theta_dot)).T)
        F_all.append(F)
    
    return t_all, X_all, F_all

def build_inv_pend_training_matrix(
    X_all: Sequence[np.ndarray],
    F_all: Sequence[np.ndarray],
    *,
    H: int,
    m: float,
    g: float,
    l: float,
    device: "torch.device",
) -> "torch.Tensor":
    import numpy as np
    import torch

    from src.inverted_pendulum import _physical_state_scale

    _, state_scale, mg = _physical_state_scale(m, g, l)
    rows = []

    for X, F in zip(X_all, F_all):
        Xn = X / state_scale.reshape(4, 1)
        un = F / mg
        num_steps = np.asarray(F).size
        n_windows = num_steps - H + 1

        for k in range(max(0, n_windows)):
            x_seq = Xn[:, k : k + H + 1].T
            u_seq = un[k : k + H].reshape(H, 1)
            rows.append(np.concatenate([x_seq.reshape(-1), u_seq.reshape(-1)]))

    if not rows:
        raise RuntimeError(
            "no training windows were generated; try reducing H or increasing "
            "num-points/t-span"
        )

    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)

def identify_sys_multiple_trajectories_u(t_all, X_all, u_all, model_type="continuous", lift=dummy_lift):
    """Identifies lifted system matrices from multiple trajectories."""
    Z_blocks = []
    dZdt_blocks = []
    Zk_blocks = []
    Zkp1_blocks = []
    Uk_blocks = []

    for X, u, t in zip(X_all, u_all, t_all):
        Z = np.array([lift(X[:, i]) for i in range(X.shape[1])]).T

        if model_type == "continuous":
            u = np.asarray(u, dtype=float).reshape(-1)
            if u.size == Z.shape[1] - 1:
                Z = Z[:, :-1]
                t = np.asarray(t)[:-1]
            elif u.size != Z.shape[1]:
                raise ValueError("u must have either the same number of samples as X or one fewer sample.")
            try:
                dZdt_blocks.append(finite_difference(Z, t))
            except ValueError as e:
                continue  # Skip this trajectory if it doesn't have the right number of samples
            Z_blocks.append(Z)
            Uk_blocks.append(u.reshape(1, -1))
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
    rank_PE, pe_flag = is_persistently_exciting(np.hstack(Uk_blocks).ravel(), L) # TODO: Check L
    print("PE rank:", rank_PE, "Full:", pe_flag)

    return A, B

def identify_sys_multiple_trajectories(t_all, X_all, F_all, m=1., g=1., l=1., model_type="continuous", lift=dummy_lift):
    """Identifies lifted system matrices from multiple trajectories."""
    t0, state_scale, mg = _physical_state_scale(m, g, l)
    u_all = [F / mg for F in F_all]  # Assuming normalized input
    t_all = [t / t0 for t in t_all]  # Normalize time
    X_all = [X / state_scale.reshape(4, 1) for X in X_all]  # Normalize states
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
    

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    return K

def lqr_u_caller(A, B, umax=10, y_ref=None, u_ref=0.0, Q=None, R=None, model_type="continuous", lift=dummy_lift):
    """Builds a normalized-input LQR controller """
    if y_ref is None:
        y_ref = np.zeros(4)
    if Q is None:
        Q = np.diag([0, 10, 50, 10])
    if R is None:
        R = np.array([[1.0]])

    if model_type == "continuous":
        K = lqr_ct(A, B, Q, R)
    elif model_type == "discrete":
        K = lqr(A, B, Q, R)
    else:
        raise ValueError("model_type must be either 'continuous' or 'discrete'.")

    z_ref = lift(y_ref)

    def u_caller(_, y):
        z_err = lift(y) - z_ref
        u = u_ref - (K @ z_err).item()
        return np.clip(u, -umax, umax)

    return u_caller

def lqr_F_caller(A, B, m=1., g=1., l=1., umax=10, y_ref=None, u_ref=0.0, Q=None, R=None, model_type="continuous", lift=dummy_lift):
    """Builds a force-based wrapper around the normalized lifted-system LQR controller."""
    t0, state_scale, mg = _physical_state_scale(m, g, l)
    u_caller = lqr_u_caller(
        A=A,
        B=B,
        umax=umax,
        y_ref=y_ref,
        u_ref=u_ref,
        Q=Q,
        R=R,
        model_type=model_type,
        lift=lift
    )
    return wrap_u_caller_as_physical_F_caller(u_caller, m, g, l) 

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



# TODO: input should start right away

