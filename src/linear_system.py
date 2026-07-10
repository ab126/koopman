"""Utilities for controllable linear-system experiments.

This module provides a simple discrete-time linear-system sandbox

    x_{k+1} = A x_k + B u_k,

for testing trajectory-manifold learning and control before moving to nonlinear
systems such as the inverted pendulum.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


def controllability_matrix(A: Array, B: Array) -> Array:
    """
    Build the discrete-time controllability matrix.

    Parameters
    ----------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    Returns
    -------
    np.ndarray, shape (x_dim, x_dim * u_dim)
        Controllability matrix ``[B, AB, ..., A^{n-1}B]``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    x_dim = A.shape[0]
    blocks = [B]

    for k in range(1, x_dim):
        blocks.append(np.linalg.matrix_power(A, k) @ B)

    return np.hstack(blocks)


def is_controllable(A: Array, B: Array, tol: float = 1e-10) -> tuple[int, bool]:
    """
    Check controllability of a discrete-time linear system.

    Parameters
    ----------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    tol : float, optional
        Numerical tolerance used in matrix-rank computation.

    Returns
    -------
    rank : int
        Rank of the controllability matrix.

    controllable : bool
        Whether the system is controllable.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must have shape (x_dim, x_dim).")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B must have shape (x_dim, u_dim).")

    C = controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(C, tol=tol)
    return rank, rank == A.shape[0]


def sample_controllable_linear_system(
    x_dim: int,
    u_dim: int,
    *,
    spectral_radius: float = 0.95,
    max_tries: int = 1000,
    seed: Optional[int] = None,
) -> tuple[Array, Array]:
    """
    Sample a random controllable discrete-time linear system.

    Parameters
    ----------
    x_dim : int
        State dimension.

    u_dim : int
        Input dimension.

    spectral_radius : float, optional
        Desired spectral radius of ``A`` after rescaling. Values below one
        produce a stable open-loop system.

    max_tries : int, optional
        Maximum number of random systems to try.

    seed : int, optional
        Random seed.

    Returns
    -------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    Raises
    ------
    RuntimeError
        If no controllable system is found within ``max_tries``.
    """
    if x_dim <= 0:
        raise ValueError("x_dim must be positive.")
    if u_dim <= 0:
        raise ValueError("u_dim must be positive.")
    if spectral_radius <= 0:
        raise ValueError("spectral_radius must be positive.")

    rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        A = rng.standard_normal((x_dim, x_dim))
        eig_radius = np.max(np.abs(np.linalg.eigvals(A)))

        if eig_radius > 0:
            A = A / eig_radius * spectral_radius

        B = rng.standard_normal((x_dim, u_dim))

        _, controllable = is_controllable(A, B)
        if controllable:
            return A, B

    raise RuntimeError(
        f"could not sample a controllable system after {max_tries} attempts"
    )


def simulate_linear_system(
    A: Array,
    B: Array,
    x0: Array,
    U: Array,
) -> Array:
    """
    Simulate a discrete-time linear system.

    Parameters
    ----------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    x0 : np.ndarray, shape (x_dim,)
        Initial state.

    U : np.ndarray, shape (u_dim, num_steps)
        Input sequence.

    Returns
    -------
    X : np.ndarray, shape (x_dim, num_steps + 1)
        State trajectory, including the initial state.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must have shape (x_dim, x_dim).")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B must have shape (x_dim, u_dim).")
    if x0.shape[0] != A.shape[0]:
        raise ValueError("x0 must have shape (x_dim,).")
    if U.ndim != 2 or U.shape[0] != B.shape[1]:
        raise ValueError("U must have shape (u_dim, num_steps).")

    x_dim = A.shape[0]
    num_steps = U.shape[1]

    X = np.zeros((x_dim, num_steps + 1), dtype=float)
    X[:, 0] = x0

    for k in range(num_steps):
        X[:, k + 1] = A @ X[:, k] + B @ U[:, k]

    return X


def default_x0_sampler(
    rng: np.random.Generator,
    x_dim: int,
    scale: float = 1.0,
) -> Array:
    """
    Sample a Gaussian initial condition.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    x_dim : int
        State dimension.

    scale : float, optional
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray, shape (x_dim,)
        Initial state sample.
    """
    return scale * rng.standard_normal(x_dim)


def default_u_sampler(
    rng: np.random.Generator,
    u_dim: int,
    num_steps: int,
    scale: float = 1.0,
) -> Array:
    """
    Sample a Gaussian input sequence.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    u_dim : int
        Input dimension.

    num_steps : int
        Number of control steps.

    scale : float, optional
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray, shape (u_dim, num_steps)
        Input sequence.
    """
    return scale * rng.standard_normal((u_dim, num_steps))


def generate_linear_trajectory_data(
    A: Array,
    B: Array,
    *,
    num_steps: int,
    n_repeats: int,
    x0_sampler: Optional[Callable[[np.random.Generator, int], Array]] = None,
    u_sampler: Optional[Callable[[np.random.Generator, int, int], Array]] = None,
    process_noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[list[Array], list[Array], list[Array]]:
    """
    Generate state-input trajectories from a discrete-time linear system.

    Parameters
    ----------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    num_steps : int
        Number of discrete time steps per trajectory.

    n_repeats : int
        Number of trajectories to generate.

    x0_sampler : callable, optional
        Function called as ``x0_sampler(rng, x_dim)``. If ``None``, standard
        Gaussian initial states are used.

    u_sampler : callable, optional
        Function called as ``u_sampler(rng, u_dim, num_steps)``. If ``None``,
        standard Gaussian inputs are used.

    process_noise_std : float, optional
        Standard deviation of additive process noise. If zero, dynamics are
        deterministic.

    seed : int, optional
        Random seed.

    Returns
    -------
    t_all : list of np.ndarray
        Time indices. Each entry has shape ``(num_steps + 1,)``.

    X_all : list of np.ndarray
        State trajectories. Each entry has shape ``(x_dim, num_steps + 1)``.

    U_all : list of np.ndarray
        Input trajectories. Each entry has shape ``(u_dim, num_steps)``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    if process_noise_std < 0:
        raise ValueError("process_noise_std must be nonnegative.")

    x_dim = A.shape[0]
    u_dim = B.shape[1]

    rng = np.random.default_rng(seed)
    x0_sampler = x0_sampler or default_x0_sampler
    u_sampler = u_sampler or default_u_sampler

    t_all: list[Array] = []
    X_all: list[Array] = []
    U_all: list[Array] = []

    for _ in range(n_repeats):
        x0 = np.asarray(x0_sampler(rng, x_dim), dtype=float).reshape(x_dim)
        U = np.asarray(u_sampler(rng, u_dim, num_steps), dtype=float)

        if U.shape != (u_dim, num_steps):
            raise ValueError(
                f"u_sampler must return shape {(u_dim, num_steps)}, "
                f"got {U.shape}."
            )

        X = simulate_linear_system(A, B, x0, U)

        if process_noise_std > 0:
            noise = process_noise_std * rng.standard_normal(X.shape)
            noise[:, 0] = 0.0
            X = X + noise

        t_all.append(np.arange(num_steps + 1, dtype=float))
        X_all.append(X)
        U_all.append(U)

    return t_all, X_all, U_all


def simulate_discrete_closed_loop(
    A: Array,
    B: Array,
    u_caller: Callable[[int, Array], Array],
    x0: Array,
    *,
    num_steps: int,
) -> tuple[Array, Array]:
    """
    Simulate a closed-loop discrete-time linear system.

    Parameters
    ----------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    u_caller : callable
        Controller called as ``u_caller(k, x_k)``.

    x0 : np.ndarray, shape (x_dim,)
        Initial state.

    num_steps : int
        Number of closed-loop steps.

    Returns
    -------
    X : np.ndarray, shape (x_dim, num_steps + 1)
        Closed-loop state trajectory.

    U : np.ndarray, shape (u_dim, num_steps)
        Closed-loop input trajectory.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    x_dim = A.shape[0]
    u_dim = B.shape[1]

    X = np.zeros((x_dim, num_steps + 1), dtype=float)
    U = np.zeros((u_dim, num_steps), dtype=float)

    X[:, 0] = x0

    for k in range(num_steps):
        u_k = np.asarray(u_caller(k, X[:, k]), dtype=float).reshape(u_dim)
        U[:, k] = u_k
        X[:, k + 1] = A @ X[:, k] + B @ u_k

    return X, U


def finite_horizon_lqr_gain(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    *,
    Qf: Optional[Array] = None,
    horizon: int = 50,
) -> list[Array]:
    """
    Compute finite-horizon discrete-time LQR gains. The optimal input is of form ``u_k = -K_k @ x_k``.

    Parameters
    ----------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    Q : np.ndarray, shape (x_dim, x_dim)
        State cost matrix.

    R : np.ndarray, shape (u_dim, u_dim)
        Input cost matrix.

    Qf : np.ndarray, optional
        Terminal cost matrix. If ``None``, ``Q`` is used.

    horizon : int, optional
        Number of control intervals.

    Returns
    -------
    list of np.ndarray
        Feedback gains ``K_k`` such that ``u_k = -K_k x_k``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    Qf = Q if Qf is None else np.asarray(Qf, dtype=float)

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    P = Qf.copy()
    gains: list[Array] = []

    for _ in range(horizon):
        G = R + B.T @ P @ B
        K = np.linalg.solve(G, B.T @ P @ A)
        gains.append(K)
        P = Q + A.T @ P @ (A - B @ K)

    gains.reverse()
    return gains


def finite_horizon_lqr_u_caller(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    *,
    Qf: Optional[Array] = None,
    horizon: int = 50,
    x_ref: Optional[Array] = None,
    u_ref: Optional[Array] = None,
    umax: Optional[float] = None,
) -> Callable[[int, Array], Array]:
    """
    Build a finite-horizon discrete-time LQR controller.

    Parameters
    ----------
    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    Q : np.ndarray, shape (x_dim, x_dim)
        State cost matrix.

    R : np.ndarray, shape (u_dim, u_dim)
        Input cost matrix.

    Qf : np.ndarray, optional
        Terminal cost matrix.

    horizon : int, optional
        Number of precomputed LQR gains.

    x_ref : np.ndarray, optional
        Reference state. If ``None``, zero is used.

    u_ref : np.ndarray, optional
        Reference input. If ``None``, zero is used.

    umax : float, optional
        Symmetric input saturation bound.

    Returns
    -------
    callable
        Controller called as ``u_caller(k, x_k)``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    x_dim = A.shape[0]
    u_dim = B.shape[1]

    x_ref = np.zeros(x_dim) if x_ref is None else np.asarray(x_ref, dtype=float)
    u_ref = np.zeros(u_dim) if u_ref is None else np.asarray(u_ref, dtype=float)

    gains = finite_horizon_lqr_gain(
        A,
        B,
        Q,
        R,
        Qf=Qf,
        horizon=horizon,
    )

    def u_caller(k: int, x: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(x_dim)
        K = gains[min(k, len(gains) - 1)]
        u = u_ref - K @ (x - x_ref)

        if umax is not None:
            u = np.clip(u, -umax, umax)

        return u.reshape(u_dim)

    return u_caller

