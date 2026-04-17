import numpy as np

from src.core import dynamics, identify_sys_multiple_trajectories_u


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
