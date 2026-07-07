import numpy as np
import casadi as ca
import cvxpy as cp
from cvxpy import Expression, Constraint
from typing import List

import do_mpc
from pydeepc import DeePC
from pydeepc.utils import Data

from src.inverted_pendulum import _physical_state_scale, dummy_lift, wrap_u_caller_as_physical_F_caller, quadratic_tracking_cost


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

# Deepc
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


def deepc_caller(t_all, X_all, u_all, x_tar, dt=0.02, Q=None, R=None, horizon=25, memory=50, umax=1000.0, lambda_g=1e-3, lambda_y=1e-3, lift=dummy_lift,
                 y_max=1000.0, f_horizon=1, verbose=False):
    """Builds a data-driven MPC controller from trajectories"""
    Z_blocks = []
    
    for X, u, t in zip(X_all, u_all, t_all):
        Zt = np.array([lift(X[:, i]) for i in range(X.shape[1])]).T
        Z_blocks.append(Zt)

    Zt = np.hstack(Z_blocks).T
    Ut = np.hstack(u_all).T.reshape(-1, 1)
    if verbose:
        print("rank Zt:", np.linalg.matrix_rank(Zt))

    n_y = Zt.shape[1]
    n_u = Ut.shape[1]

    # -----------------------------
    # Default weights
    # -----------------------------
    if Q is None:
        Q = np.eye(n_y)
    if R is None:
        R = np.eye(n_u)

    Q = np.array(Q)
    R = np.array(R)
    y_tar = lift(x_tar)


    # DeePC Definitions
    def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
        cost = 0

        for k in range(horizon):
            y_k = y[k, :]
            u_k = u[k, :]

            cost += cp.quad_form(y_k - y_tar, Q)
            cost += cp.quad_form(u_k, R)

        return cost
    
    def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
        horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
        # Define a list of input/output constraints
        return [u >= -umax * np.ones((horizon, n_u)), u <= umax * np.ones((horizon, n_u)), cp.norm_inf(y) <= y_max]

    data = Data(Ut, Zt)
    deepc = DeePC(data, memory, horizon)

    # Build the deepc problem
    deepc.build_problem(
        build_loss = loss_callback,
        build_constraints = constraints_callback,
        lambda_g = lambda_g,
        lambda_y = lambda_y)
    
    # Controller
    # store past window
    controller_state = {
        "next_update_t": None,
        "current_u": 0.0,
        "u_seq": None,  # Full horizon sequence
        "seq_index": 0,  # Current index in the sequence
    }
    u_hist = [np.zeros(n_u) for _ in range(memory)]  # start with zero input history (can warm-start with real data if desired)
    y_hist = [np.zeros(n_y) for _ in range(memory)]  # start with zero output history (can be warm-started with real data if desired)

    def u_caller(t, y_current):
        nonlocal u_hist, y_hist
        # print(u_hist)

        z = lift(y_current)

        # update history
        y_hist.append(z)
        if len(y_hist) > memory:
            y_hist.pop(0)

        if controller_state["next_update_t"] is None or t >= controller_state["next_update_t"]:
        
            if len(y_hist) < memory:
                return 0.0

            y_ini = np.array(y_hist)
            u_ini = np.array(u_hist)
            if verbose:
                print(f"||y_ini||: {np.linalg.norm(y_ini)}, ||u_ini||: {np.linalg.norm(u_ini)}")

            data_ini = Data(u_ini, y_ini)

            # Solve
            try:
                u_opt = deepc.solve(data_ini, solver=cp.OSQP, 
                    warm_start=True, 
                    verbose=False)[0]
            except Exception as e:
                if verbose:
                    print(f"DeePC OSQP solver failed with error: {e}\nSwitching to default solver")
                u_opt = deepc.solve(data_ini)[0]
            
            # Store the full horizon sequence
            controller_state["u_seq"] = u_opt[:f_horizon, :] if f_horizon < horizon else u_opt
            controller_state["seq_index"] = 0
            controller_state["next_update_t"] = t + f_horizon * dt
            
            u_next = u_opt[0, :]
            
            # update histories
            u_hist.append(u_next)
            if len(u_hist) > memory:
                u_hist.pop(0)

            controller_state["current_u"] = float(u_next.flatten()[0])
        else:
            # Use next input from the locked-in sequence
            controller_state["seq_index"] += 1
            if controller_state["u_seq"] is not None and controller_state["seq_index"] < len(controller_state["u_seq"]):
                u_next = controller_state["u_seq"][controller_state["seq_index"], :]
                
                # update histories with the locked-in input
                u_hist.append(u_next)
                if len(u_hist) > memory:
                    u_hist.pop(0)
                
                controller_state["current_u"] = float(u_next.flatten()[0])
        
        return controller_state["current_u"]
    
    return u_caller
