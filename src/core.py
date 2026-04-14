import numpy as np
from scipy.integrate import solve_ivp


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

    return [x_dot, x_ddot, theta_dot, theta_ddot]

def get_si_values(y, m, g, l):
    t_0 = np.sqrt(l / g)
    y_1, y_2, y_3, y_4 = y
    return [y_1 * l, y_2 * l / t_0, y_3, y_4 / t_0]

# ----------------------------
# Simulation
# ----------------------------
def simulate(F, M, m, g, l):
    """Simulates the system dynamics given a control policy F(t, y)."""
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 500)

    # Initial condition: [x, xdot, theta, thetadot]
    y0 = [0.0, 0.0, 0.03, 0.0] # In normalized units m=g=l=1

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
    # stack data
    W = np.vstack((Zk, Uk))

    # least squares
    K = Zkp1 @ np.linalg.pinv(W)

    n = Zk.shape[0]

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


def identify_sys(x, x_dot, theta, theta_dot, u):
    """ Given state variables, identifies the system matrices A and B using Koopman operator theory. """
    X = np.column_stack((x, x_dot, theta, theta_dot)).T

    # Dataset
    Zk, Zkp1, Uk = build_dataset(X, u)

    # Koopman model
    A, B = koopman_identification(Zk, Zkp1, Uk)

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

def opt_F_caller(A, B, m, g, umax = 10):
    C = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # x
        [0, 1, 0, 0, 0, 0, 0],   # xdot
        [0, 0, 1, 0, 0, 0, 0],   # theta
        [0, 0, 0, 1, 0, 0, 0],   # thetadot
    ])
    Q_phys = np.diag([0, 10, 50, 10])
    Q_z = C.T @ Q_phys @ C
    R = np.array([[1.]])

    K = lqr(A, B, Q_z, R)
    return lambda _, y: np.clip((-K @ lift(y)).item(), -umax, umax) * (m * g) # Scale back to physical units

# TODO add MPC control design as well

