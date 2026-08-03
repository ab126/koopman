import numpy as np
import torch

from scripts.run_inverted_pendulum_mc import (
    build_arg_parser,
    simulate_discrete_inverted_pendulum,
)
from src.inverted_pendulum import (
    dynamics,
    identify_sys_multiple_trajectories_u,
    rk4_step,
    simulate_u_rk4,
    wrap_u_caller_as_physical_F_caller,
)
from src.manifold_control import build_trajectory_training_matrix


def test_dynamics_matches_linearization_in_physical_units():
    M = 1.0
    m = 0.2
    g = 9.81
    l = 0.5

    y = np.array([0.1, -0.2, 1e-6, 0.3], dtype=float)

    dy = dynamics(0.0, y, lambda t, state: 0.0, M, m, g, l)

    t0 = np.sqrt(l / g)
    alpha = M / m + 1.0
    det = alpha / 3.0 - 0.25
    a22 = -0.25 / det
    a42 = 0.5 * alpha / det

    expected = np.array([
        y[1],
        a22 * y[2] / t0**2,
        y[3],
        a42 * y[2] / t0**2,
    ])

    assert np.allclose(dy, expected, atol=1e-5, rtol=1e-5)


def test_discrete_identification_respects_custom_lift():
    A_true = np.array([
        [1.0, 0.1, 0.0, 0.0],
        [0.0, 0.9, 0.2, 0.0],
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 0.0, 0.0, 0.95],
    ])
    B_true = np.array([[0.0], [0.3], [0.0], [0.1]])

    x0 = np.array([0.2, -0.1, 0.05, 0.0])
    u = np.linspace(-0.3, 0.3, 20)

    X = np.zeros((4, u.size + 1))
    X[:, 0] = x0
    for k in range(u.size):
        X[:, k + 1] = A_true @ X[:, k] + (B_true[:, 0] * u[k])

    t = np.arange(u.size + 1, dtype=float)
    A_hat, B_hat = identify_sys_multiple_trajectories_u(
        [t],
        [X],
        [u],
        model_type="discrete",
        lift=lambda y: y,
    )

    assert A_hat.shape == A_true.shape
    assert B_hat.shape == B_true.shape
    assert np.allclose(A_hat, A_true, atol=1e-10, rtol=1e-10)
    assert np.allclose(B_hat, B_true, atol=1e-10, rtol=1e-10)


def test_fixed_step_simulators_hold_one_bounded_input_per_step():
    calls = 0

    def controller(_time, _state):
        nonlocal calls
        calls += 1
        return 3.0

    t, x, x_dot, theta, theta_dot, u = simulate_u_rk4(
        controller, 2.0, num_points=8, t_span=(0.0, 0.16)
    )
    assert rk4_step(np.zeros(4), 0.0, 0.02, 2.0).shape == (4,)
    assert t.shape == x.shape == x_dot.shape == theta.shape == theta_dot.shape == (9,)
    assert u.shape == (8,)
    assert calls == 8

    calls = 0
    t, X, U = simulate_discrete_inverted_pendulum(
        controller, 2.0, np.zeros(4), 0.02, 8, umax=2.0
    )
    assert t.shape == (9,)
    assert X.shape == (4, 9)
    assert U.shape == (1, 8)
    assert calls == 8
    assert np.all(np.abs(U) <= 2.0)


def test_training_width_defaults_and_normalized_force_conversion():
    horizon = 25
    X = np.zeros((4, horizon + 2))
    U = np.zeros((1, horizon + 1))
    W = build_trajectory_training_matrix(
        [X], [U], horizon=horizon, device=torch.device("cpu"), dtype=torch.float32
    )
    assert W.shape[1] == (horizon + 1) * 4 + horizon
    args = build_arg_parser().parse_args([])
    assert args.H == 25
    assert args.alpha_dim == 29

    umax, m, g, l = 2.0, 3.0, 9.81, 0.5
    physical = wrap_u_caller_as_physical_F_caller(
        lambda _time, _state: umax, m, g, l
    )
    assert np.isclose(abs(physical(0.0, np.zeros(4))), umax * m * g)
