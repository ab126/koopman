"""Trajectory-manifold learning and control with PyTorch.

This module implements the optimization problem

    min_{u, x, theta, alpha} sum_tau l_QR(x_tau, u_tau)
        + lambda_theta ||w - phi_theta(alpha)||_2^2
        + lambda_C Omega(M_theta),

where ``phi_theta`` is a neural parametrization of the behavior manifold in
trajectory space and ``w = [vec(x), vec(u)]``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
import numpy as np
from torch import nn, seed
from pathlib import Path

from .inverted_pendulum import wrap_u_caller_as_physical_F_caller


Array = np.ndarray
TensorLike = torch.Tensor


class BehaviorDecoder(nn.Module):
    """Neural decoder for a trajectory-space behavior manifold.

    Parameters
    ----------
    alpha_dim : int
        Dimension of the latent coordinate ``alpha``.
    w_dim : int
        Dimension of the flattened trajectory vector ``w``.
    hidden_dims : sequence of int, optional
        Widths of hidden layers in the MLP.
    activation : callable, optional
        Torch module class used between linear layers.

    Notes
    -----
    The forward map is ``phi_theta(alpha)``. If ``alpha`` has shape
    ``(alpha_dim,)``, the output has shape ``(w_dim,)``. Batched input with
    shape ``(batch, alpha_dim)`` returns ``(batch, w_dim)``.
    """

    def __init__(
        self,
        alpha_dim: int,
        w_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if alpha_dim <= 0:
            raise ValueError("alpha_dim must be positive.")
        if w_dim <= 0:
            raise ValueError("w_dim must be positive.")

        dims = [alpha_dim, *hidden_dims, w_dim]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.alpha_dim = int(alpha_dim)
        self.w_dim = int(w_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, alpha: TensorLike) -> TensorLike:
        """Evaluate ``phi_theta(alpha)``.

        Parameters
        ----------
        alpha : torch.Tensor
            Latent coordinate with trailing dimension ``alpha_dim``.

        Returns
        -------
        torch.Tensor
            Decoded flattened behavior vector with trailing dimension
            ``w_dim``.
        """

        if alpha.shape[-1] != self.alpha_dim:
            raise ValueError(
                f"Expected alpha trailing dimension {self.alpha_dim}, "
                f"got {alpha.shape[-1]}."
            )
        return self.net(alpha)


class BehaviorEncoder(nn.Module):
    """Neural encoder for a trajectory-space behavior manifold.

    Parameters
    ----------
    w_dim : int
        Dimension of the flattened trajectory vector ``w``.
    alpha_dim : int
        Dimension of the unconstrained latent coordinate ``alpha``.
    hidden_dims : sequence of int, optional
        Decoder hidden widths. The encoder uses these widths in reverse order.
    activation : callable, optional
        Torch module class used between linear layers.

    Notes
    -----
    The forward map has dimensions ``w_dim -> reversed(hidden_dims) ->
    alpha_dim``. The output layer is linear. Batched inputs are supported.
    """

    def __init__(
        self,
        w_dim: int,
        alpha_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if w_dim <= 0:
            raise ValueError("w_dim must be positive.")
        if alpha_dim <= 0:
            raise ValueError("alpha_dim must be positive.")

        dims = [w_dim, *reversed(hidden_dims), alpha_dim]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.w_dim = int(w_dim)
        self.alpha_dim = int(alpha_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, w: TensorLike) -> TensorLike:
        """Encode flattened behavior vectors.

        Parameters
        ----------
        w : torch.Tensor
            Behavior vector with trailing dimension ``w_dim``.

        Returns
        -------
        torch.Tensor
            Latent coordinates with trailing dimension ``alpha_dim``.
        """

        if w.shape[-1] != self.w_dim:
            raise ValueError(
                f"Expected w trailing dimension {self.w_dim}, got {w.shape[-1]}."
            )
        return self.net(w)


@dataclass
class BehaviorAutoencoder:
    """Paired encoder and decoder for trajectory behavior vectors."""

    encoder: BehaviorEncoder
    decoder: BehaviorDecoder


@dataclass
class BehaviorManifoldSolution:
    """Container returned by :class:`BehaviorManifoldControlSolver`.

    Attributes
    ----------
    x : torch.Tensor
        Optimized state sequence with shape ``(horizon + 1, x_dim)``.
    u : torch.Tensor
        Optimized input sequence with shape ``(horizon, u_dim)``.
    alpha : torch.Tensor
        Optimized latent coordinate.
    w_hat : torch.Tensor
        Decoder prediction ``phi_theta(alpha)`` at the final iterate.
    loss : float
        Final scalar objective value.
    loss_dict : dict
        Final component losses.
    history : list of dict
        Sampled scalar loss diagnostics, or an empty list when history storage
        is disabled.
    iterations : int
        Number of optimizer steps completed. This can be less than the
        configured maximum when early stopping is enabled.
    """

    x: TensorLike
    u: TensorLike
    alpha: TensorLike
    w_hat: TensorLike
    loss: float
    loss_dict: Dict[str, float]
    history: List[Dict[str, float]]
    iterations: int = 0


class ManifoldLoss:
    """Composite QR, manifold-fit, and curvature objective.

    Parameters
    ----------
    decoder : torch.nn.Module
        Neural map ``phi_theta(alpha)``.
    Q : torch.Tensor
        State tracking matrix with shape ``(x_dim, x_dim)``.
    R : torch.Tensor
        Input tracking matrix with shape ``(u_dim, u_dim)``.
    lambda_theta : float, optional
        Weight on ``||w - phi_theta(alpha)||_2^2``.
    lambda_curvature : float, optional
        Weight on the smoothness penalty ``Omega(M_theta)``.
    x_ref : torch.Tensor, optional
        Reference state, either shape ``(x_dim,)`` or ``x_seq.shape``.
    u_ref : torch.Tensor, optional
        Reference input, either shape ``(u_dim,)`` or ``u_seq.shape``.
    curvature_mode : {"exact", "local", "none"}, optional
        Curvature estimator. ``"exact"`` computes Hessians directly, while
        ``"local"`` uses a second-order finite perturbation residual.
    local_eps : float, optional
        Perturbation scale used by ``curvature_mode="local"``.
    """

    def __init__(
        self,
        decoder: nn.Module,
        Q: TensorLike,
        R: TensorLike,
        lambda_theta: float = 1.0,
        lambda_curvature: float = 0.0,
        x_ref: Optional[TensorLike] = None,
        u_ref: Optional[TensorLike] = None,
        curvature_mode: str = "exact",
        local_eps: float = 1e-2,
    ) -> None:
        self.decoder = decoder
        self.Q = Q
        self.R = R
        self.lambda_theta = float(lambda_theta)
        self.lambda_curvature = float(lambda_curvature)
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.curvature_mode = curvature_mode
        self.local_eps = float(local_eps)

        if curvature_mode not in {"exact", "local", "none"}:
            raise ValueError("curvature_mode must be 'exact', 'local', or 'none'.")

    def __call__(
        self,
        x_seq: TensorLike,
        u_seq: TensorLike,
        alpha: TensorLike,
        w_target: Optional[TensorLike] = None,
        alpha_samples: Optional[TensorLike] = None,
    ) -> Tuple[TensorLike, Dict[str, TensorLike]]:
        """Evaluate the full objective.

        Parameters
        ----------
        x_seq : torch.Tensor
            State trajectory with shape ``(H + 1, x_dim)``.
        u_seq : torch.Tensor
            Input trajectory with shape ``(H, u_dim)``.
        alpha : torch.Tensor
            Latent coordinate for the current trajectory.
        w_target : torch.Tensor, optional
            Behavior vector to fit. If omitted, ``build_w(x_seq, u_seq)`` is
            used so the optimized trajectory is encouraged to lie on the
            learned manifold.
        alpha_samples : torch.Tensor, optional
            Points used to estimate the curvature penalty. If omitted, the
            current ``alpha`` is used.

        Returns
        -------
        loss_total : torch.Tensor
            Scalar objective.
        loss_dict : dict
            Tensor-valued components ``qr``, ``fit``, ``curvature``, and
            ``total``.
        """

        qr_loss = quadratic_tracking_loss(
            x_seq=x_seq,
            u_seq=u_seq,
            Q=self.Q.to(device=x_seq.device, dtype=x_seq.dtype),
            R=self.R.to(device=u_seq.device, dtype=u_seq.dtype),
            x_ref=_to_optional_device(self.x_ref, x_seq),
            u_ref=_to_optional_device(self.u_ref, u_seq),
        )

        w = build_w(x_seq, u_seq) if w_target is None else w_target.reshape(-1)
        w_hat = self.decoder(alpha).reshape(-1)
        if w.shape != w_hat.shape:
            raise ValueError(
                f"w and phi_theta(alpha) must have the same shape; got "
                f"{tuple(w.shape)} and {tuple(w_hat.shape)}."
            )
        fit_loss = torch.sum((w - w_hat) ** 2)

        curvature_loss = x_seq.new_tensor(0.0)
        if self.lambda_curvature != 0.0 and self.curvature_mode != "none":
            samples = alpha.reshape(1, -1) if alpha_samples is None else alpha_samples
            if self.curvature_mode == "exact":
                curvature_loss = curvature_penalty_exact(self.decoder, samples)
            else:
                curvature_loss = curvature_penalty_local(
                    self.decoder, samples, eps=self.local_eps
                )

        loss_total = (
            qr_loss
            + self.lambda_theta * fit_loss
            + self.lambda_curvature * curvature_loss
        )
        return loss_total, {
            "qr": qr_loss,
            "fit": fit_loss,
            "curvature": curvature_loss,
            "total": loss_total,
        }


class VariableManager:
    """Select trainable tensors according to freeze flags.

    Parameters
    ----------
    decoder : torch.nn.Module
        Decoder whose parameters represent ``theta``.
    x_seq : torch.Tensor
        State sequence parameter candidate.
    u_seq : torch.Tensor
        Input sequence parameter candidate.
    alpha : torch.Tensor
        Latent coordinate parameter candidate.
    freeze : mapping, optional
        Boolean flags for ``"theta"``, ``"x"``, ``"u"``, and ``"alpha"``.
        Missing keys default to ``False``.
    """

    default_freeze = {"theta": False, "x": False, "u": False, "alpha": False}

    def __init__(
        self,
        decoder: nn.Module,
        x_seq: TensorLike,
        u_seq: TensorLike,
        alpha: TensorLike,
        freeze: Optional[Mapping[str, bool]] = None,
    ) -> None:
        self.decoder = decoder
        self.x_seq = x_seq
        self.u_seq = u_seq
        self.alpha = alpha
        self.freeze = {**self.default_freeze, **(freeze or {})}

    def get_trainable_params(self) -> List[TensorLike]:
        """Return optimizer parameters consistent with ``freeze``.

        Returns
        -------
        params : list of torch.Tensor
            Tensors and decoder parameters that should receive optimizer
            updates.
        """

        params: List[TensorLike] = []
        for name, tensor in (
            ("x", self.x_seq),
            ("u", self.u_seq),
            ("alpha", self.alpha),
        ):
            tensor.requires_grad_(not self.freeze[name])
            if not self.freeze[name]:
                params.append(tensor)

        theta_trainable = not self.freeze["theta"]
        for param in self.decoder.parameters():
            param.requires_grad_(theta_trainable)
        if theta_trainable:
            params.extend(self.decoder.parameters())

        return params

# TODO: Add a seperate optimized online solver. It should first find the alpha (low dim optimization problem) then should not create a new instance of the solver each step but use the previous state and alpha efficiently
class BehaviorManifoldControlSolver:
    """Gradient solver for behavior-manifold learning/control.

    Parameters
    ----------
    decoder : torch.nn.Module
        Neural decoder ``phi_theta``.
    x_dim : int
        State dimension.
    u_dim : int
        Input dimension.
    horizon : int
        Number of control intervals. ``x`` has ``horizon + 1`` rows and ``u``
        has ``horizon`` rows.
    Q : torch.Tensor, optional
        State tracking matrix. Defaults to identity.
    R : torch.Tensor, optional
        Input tracking matrix. Defaults to identity.
    lambda_theta : float, optional
        Weight on manifold-fit term.
    lambda_curvature : float, optional
        Weight on smoothness term.
    x_ref : torch.Tensor, optional
        State reference used in the QR cost.
    u_ref : torch.Tensor, optional
        Input reference used in the QR cost.
    lr : float, optional
        Optimizer learning rate.
    max_iter : int, optional
        Number of gradient steps.
    optimizer_cls : type, optional
        Torch optimizer class.
    curvature_mode : {"exact", "local", "none"}, optional
        Curvature penalty mode.
    u_bounds : tuple of float, optional
        Elementwise lower and upper bounds for ``u`` applied after each step.
    x_bounds : tuple of float, optional
        Elementwise lower and upper bounds for ``x`` applied after each step.
    min_iter : int, optional
        Minimum optimizer steps before early stopping is allowed.
    patience : int, optional
        Stop after this many consecutive iterations without a relative loss
        improvement larger than ``relative_loss_tol``. ``None`` disables
        early stopping, preserving the previous fixed-iteration behavior.
    relative_loss_tol : float, optional
        Minimum relative decrease in loss that resets early-stopping patience.
    store_history : bool, optional
        Store scalar per-iteration diagnostics. Disabled by default to avoid
        device synchronization on every optimizer step.
    store_every : int, optional
        Store every Nth iteration when ``store_history`` is enabled.
    """

    def __init__(
        self,
        decoder: nn.Module,
        x_dim: int,
        u_dim: int,
        horizon: int,
        Q: Optional[TensorLike] = None,
        R: Optional[TensorLike] = None,
        lambda_theta: float = 1.0,
        lambda_curvature: float = 1.0,
        x_ref: Optional[TensorLike] = None,
        u_ref: Optional[TensorLike] = None,
        lr: float = 1e-3,
        max_iter: int = 1000,
        optimizer_cls: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
        curvature_mode: str = "exact",
        local_eps: float = 1e-2,
        u_bounds: Optional[Tuple[float, float]] = None,
        x_bounds: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        min_iter: int = 0,
        patience: Optional[int] = None,
        relative_loss_tol: float = 0.0,
        store_history: bool = False,
        store_every: int = 1,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if x_dim <= 0 or u_dim <= 0:
            raise ValueError("x_dim and u_dim must be positive.")
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if min_iter < 0:
            raise ValueError("min_iter must be non-negative.")
        if patience is not None and patience <= 0:
            raise ValueError("patience must be positive or None.")
        if relative_loss_tol < 0:
            raise ValueError("relative_loss_tol must be non-negative.")
        if store_every <= 0:
            raise ValueError("store_every must be positive.")

        self.decoder = decoder
        self.x_dim = int(x_dim)
        self.u_dim = int(u_dim)
        self.horizon = int(horizon)
        self.device = device or next(decoder.parameters()).device
        self.dtype = dtype
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.optimizer_cls = optimizer_cls
        self.u_bounds = u_bounds
        self.x_bounds = x_bounds
        self.min_iter = int(min_iter)
        self.patience = None if patience is None else int(patience)
        self.relative_loss_tol = float(relative_loss_tol)
        self.store_history = bool(store_history)
        self.store_every = int(store_every)

        self.Q = self._as_tensor(Q, (self.x_dim, self.x_dim), torch.eye(self.x_dim))
        self.R = self._as_tensor(R, (self.u_dim, self.u_dim), torch.eye(self.u_dim))
        self.x_ref = self._optional_tensor(x_ref)
        self.u_ref = self._optional_tensor(u_ref)
        self.loss_fn = ManifoldLoss(
            decoder=self.decoder,
            Q=self.Q,
            R=self.R,
            lambda_theta=lambda_theta,
            lambda_curvature=lambda_curvature,
            x_ref=self.x_ref,
            u_ref=self.u_ref,
            curvature_mode=curvature_mode,
            local_eps=local_eps,
        )
        self.last_result: Optional[BehaviorManifoldSolution] = None

    def solve(
        self,
        x_init: Optional[TensorLike] = None,
        u_init: Optional[TensorLike] = None,
        alpha_init: Optional[TensorLike] = None,
        w_target: Optional[TensorLike] = None,
        alpha_samples: Optional[TensorLike] = None,
        freeze: Optional[Mapping[str, bool]] = None,
        callback: Optional[Callable[[int, TensorLike, Dict[str, TensorLike]], None]] = None,
    ) -> BehaviorManifoldSolution:
        """Optimize ``x``, ``u``, ``alpha``, and optionally decoder weights.

        Parameters
        ----------
        x_init : torch.Tensor, optional
            Initial state sequence, shape ``(horizon + 1, x_dim)``. Defaults
            to zeros.
        u_init : torch.Tensor, optional
            Initial input sequence, shape ``(horizon, u_dim)``. Defaults to
            zeros.
        alpha_init : torch.Tensor, optional
            Initial latent coordinate, shape ``(alpha_dim,)``. Defaults to
            zeros.
        w_target : torch.Tensor, optional
            Fixed trajectory vector for system identification. Pass this when
            you want to fit ``theta`` and/or ``alpha`` to observed data.
        alpha_samples : torch.Tensor, optional
            Curvature sample points. Shape ``(K, alpha_dim)``.
        freeze : mapping, optional
            Boolean flags for ``"theta"``, ``"x"``, ``"u"``, ``"alpha"``.
            Example: ``{"theta": True, "x": False, "u": False,
            "alpha": False}``.
        callback : callable, optional
            Function called as ``callback(iteration, loss, loss_dict)`` after
            each optimizer step.

        Returns
        -------
        solution : BehaviorManifoldSolution
            Optimized variables and scalar diagnostics.
        """

        x_seq = self._parameter(
            x_init,
            default_shape=(self.horizon + 1, self.x_dim),
            name="x_init",
        )
        u_seq = self._parameter(
            u_init,
            default_shape=(self.horizon, self.u_dim),
            name="u_init",
        )
        alpha = self._parameter(
            alpha_init,
            default_shape=(self.decoder.alpha_dim,),
            name="alpha_init",
        )
        w_target_t = self._optional_tensor(w_target)
        alpha_samples_t = self._optional_tensor(alpha_samples)

        manager = VariableManager(self.decoder, x_seq, u_seq, alpha, freeze=freeze)
        params = manager.get_trainable_params()
        if not params:
            raise ValueError("All variables are frozen; there is nothing to optimize.")

        optimizer = self.optimizer_cls(params, lr=self.lr)
        history: List[Dict[str, float]] = []
        initial_state = x_seq[0].detach().clone()
        best_loss: Optional[float] = None
        stale_iterations = 0
        completed_iterations = 0

        for iteration in range(self.max_iter):
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = self.loss_fn(
                x_seq=x_seq,
                u_seq=u_seq,
                alpha=alpha,
                w_target=w_target_t,
                alpha_samples=alpha_samples_t,
            )
            loss.backward()
            optimizer.step()
            self._project_bounds(x_seq, u_seq)
            with torch.no_grad():
                x_seq[0].copy_(initial_state)
            completed_iterations = iteration + 1

            if self.store_history and iteration % self.store_every == 0:
                history.append(_detach_loss_dict(loss_dict))
            if callback is not None:
                callback(iteration, loss, loss_dict)

            if self.patience is not None:
                current_loss = float(loss.detach())
                if best_loss is None:
                    best_loss = current_loss
                    stale_iterations = 0
                else:
                    scale = max(abs(best_loss), torch.finfo(self.dtype).eps)
                    relative_improvement = (best_loss - current_loss) / scale
                    if relative_improvement > self.relative_loss_tol:
                        best_loss = current_loss
                        stale_iterations = 0
                    else:
                        stale_iterations += 1

                if (
                    completed_iterations >= self.min_iter
                    and stale_iterations >= self.patience
                ):
                    break

        with torch.no_grad():
            final_loss, final_dict = self.loss_fn(
                x_seq=x_seq,
                u_seq=u_seq,
                alpha=alpha,
                w_target=w_target_t,
                alpha_samples=alpha_samples_t,
            )
            final_diagnostics = _detach_loss_dict(final_dict)
            final_diagnostics["iterations"] = float(completed_iterations)
            solution = BehaviorManifoldSolution(
                x=x_seq.detach().clone(),
                u=u_seq.detach().clone(),
                alpha=alpha.detach().clone(),
                w_hat=self.decoder(alpha).detach().clone().reshape(-1),
                loss=float(final_loss.detach().cpu()),
                loss_dict=final_diagnostics,
                history=history,
                iterations=completed_iterations,
            )
        self.last_result = solution
        return solution

    def _as_tensor(
        self,
        value: Optional[TensorLike],
        expected_shape: Tuple[int, ...],
        default: TensorLike,
    ) -> TensorLike:
        tensor = default if value is None else value
        tensor = torch.as_tensor(tensor, device=self.device, dtype=self.dtype)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected tensor shape {expected_shape}, got {tuple(tensor.shape)}."
            )
        return tensor

    def _optional_tensor(self, value: Optional[TensorLike]) -> Optional[TensorLike]:
        if value is None:
            return None
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _parameter(
        self,
        value: Optional[TensorLike],
        default_shape: Tuple[int, ...],
        name: str,
    ) -> TensorLike:
        if value is None:
            tensor = torch.zeros(default_shape, device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
            if tuple(tensor.shape) != default_shape:
                raise ValueError(
                    f"{name} must have shape {default_shape}, got "
                    f"{tuple(tensor.shape)}."
                )
        return nn.Parameter(tensor.detach().clone())

    def _project_bounds(self, x_seq: TensorLike, u_seq: TensorLike) -> None:
        with torch.no_grad():
            if self.x_bounds is not None:
                x_seq.clamp_(self.x_bounds[0], self.x_bounds[1])
            if self.u_bounds is not None:
                u_seq.clamp_(self.u_bounds[0], self.u_bounds[1])


def build_w(x_seq: TensorLike, u_seq: TensorLike) -> TensorLike:
    """Concatenate flattened state and input trajectories.

    Parameters
    ----------
    x_seq : torch.Tensor
        State sequence of any shape.
    u_seq : torch.Tensor
        Input sequence of any shape.

    Returns
    -------
    w : torch.Tensor
        One-dimensional vector ``[vec(x_seq), vec(u_seq)]``.
    """

    return torch.cat([x_seq.reshape(-1), u_seq.reshape(-1)], dim=0)


def unpack_w(
    w: TensorLike,
    *,
    x_dim: int,
    u_dim: int,
    horizon: int,
) -> Tuple[TensorLike, TensorLike]:
    """Unpack flattened behavior vectors into state and input trajectories.

    Parameters
    ----------
    w : torch.Tensor
        Behavior vector with shape ``(w_dim,)`` or batched vectors with shape
        ``(..., w_dim)``.
    x_dim : int
        State dimension.
    u_dim : int
        Input dimension.
    horizon : int
        Number of control intervals.

    Returns
    -------
    x_seq : torch.Tensor
        State trajectory with shape ``(horizon + 1, x_dim)`` or
        ``(..., horizon + 1, x_dim)``.
    u_seq : torch.Tensor
        Input trajectory with shape ``(horizon, u_dim)`` or
        ``(..., horizon, u_dim)``.

    Raises
    ------
    ValueError
        If the trailing behavior-vector dimension is inconsistent with the
        requested dimensions.
    """
    if x_dim <= 0 or u_dim <= 0 or horizon <= 0:
        raise ValueError("x_dim, u_dim, and horizon must be positive.")
    expected = (horizon + 1) * x_dim + horizon * u_dim
    if w.ndim == 0 or w.shape[-1] != expected:
        actual = 1 if w.ndim == 0 else w.shape[-1]
        raise ValueError(f"Expected w_dim {expected}, got {actual}.")
    x_size = (horizon + 1) * x_dim
    prefix = w.shape[:-1]
    return (
        w[..., :x_size].reshape(*prefix, horizon + 1, x_dim),
        w[..., x_size:].reshape(*prefix, horizon, u_dim),
    )


def quadratic_tracking_loss(
    x_seq: TensorLike,
    u_seq: TensorLike,
    Q: TensorLike,
    R: TensorLike,
    x_ref: Optional[TensorLike] = None,
    u_ref: Optional[TensorLike] = None,
) -> TensorLike:
    """Compute ``sum_tau (x_tau-x_ref)^T Q (x_tau-x_ref) + u_tau^T R u_tau``.

    Parameters
    ----------
    x_seq : torch.Tensor
        State sequence with shape ``(..., x_dim)``.
    u_seq : torch.Tensor
        Input sequence with shape ``(..., u_dim)``.
    Q : torch.Tensor
        State cost matrix.
    R : torch.Tensor
        Input cost matrix.
    x_ref : torch.Tensor, optional
        State reference broadcastable to ``x_seq``.
    u_ref : torch.Tensor, optional
        Input reference broadcastable to ``u_seq``.

    Returns
    -------
    loss : torch.Tensor
        Scalar quadratic tracking loss.
    """

    x_err = x_seq if x_ref is None else x_seq - x_ref
    u_err = u_seq if u_ref is None else u_seq - u_ref
    x_cost = torch.einsum("...i,ij,...j->", x_err, Q, x_err)
    u_cost = torch.einsum("...i,ij,...j->", u_err, R, u_err)
    return x_cost + u_cost


@dataclass
class LatentMPCSolution:
    """Result of a latent-only behavior MPC solve.

    Attributes
    ----------
    x : torch.Tensor
        Decoded physical state sequence with shape ``(horizon + 1, x_dim)``.
    u : torch.Tensor
        Decoded physical input sequence with shape ``(horizon, u_dim)``.
    alpha : torch.Tensor
        Optimized latent coordinate with shape ``(alpha_dim,)``.
    w_hat : torch.Tensor
        Decoder output in the trajectory coordinates used during training.
    loss : float
        Final augmented objective value.
    tracking_loss : float
        Final physical-coordinate quadratic tracking objective.
    x0_rmse : float
        Root-mean-square initial-state constraint residual.
    x0_nrmse : float
        Normalized Euclidean initial-state mismatch.
    max_input_violation : float
        Maximum elementwise physical input-bound violation.
    mean_input_violation : float
        Mean elementwise physical input-bound violation.
    dynamics_residual_mean : float or None
        Mean normalized dynamics residual when ``A`` and ``B`` are available.
    dynamics_residual_p95 : float or None
        95th percentile normalized dynamics residual when available.
    feasible : bool
        Whether the final finite initial-state residual satisfies tolerance.
    iterations : int
        Total number of completed inner optimizer steps.
    outer_iterations : int
        Number of augmented-Lagrangian outer iterations completed.
    history : list of dict
        Per-iteration diagnostics, empty unless history storage was requested.
    finite : bool
        Whether the returned objective and latent coordinate are finite.
    """

    x: TensorLike
    u: TensorLike
    alpha: TensorLike
    w_hat: TensorLike
    loss: float
    tracking_loss: float
    x0_rmse: float
    x0_nrmse: float
    max_input_violation: float
    mean_input_violation: float
    dynamics_residual_mean: Optional[float]
    dynamics_residual_p95: Optional[float]
    feasible: bool
    iterations: int
    outer_iterations: int
    history: List[Dict[str, float]]
    finite: bool = True


class LatentBehaviorMPCSolver:
    """Optimize only the latent coordinate of a frozen behavior decoder.

    Decoding and all constraints are differentiable with respect to ``alpha``.
    Costs and residuals are evaluated in physical coordinates, after optional
    trajectory denormalization.

    Parameters
    ----------
    decoder : torch.nn.Module
        Trained trajectory decoder. Its parameters are frozen by the solver.
    x_dim : int
        Physical state dimension.
    u_dim : int
        Physical input dimension.
    horizon : int
        Number of control intervals in each decoded trajectory.
    Q : torch.Tensor, optional
        Stage state-cost matrix. Defaults to identity.
    R : torch.Tensor, optional
        Stage input-cost matrix. Defaults to identity.
    encoder : torch.nn.Module, optional
        Frozen trajectory encoder used only to construct latent warm starts.
    x_ref : torch.Tensor, optional
        Physical state reference. Defaults to zero.
    u_ref : torch.Tensor, optional
        Physical input reference. Defaults to zero.
    Q_terminal : torch.Tensor, optional
        Terminal state-cost matrix. Defaults to ``Q``.
    u_bounds : tuple of tensor-like, optional
        Scalar or per-input physical lower and upper bounds.
    lambda_u_bounds : float, optional
        Weight on squared decoded-input bound violations.
    lambda_alpha : float, optional
        Weight on latent-coordinate regularization.
    alpha_mean, alpha_std : torch.Tensor, optional
        Latent training statistics for standardized regularization.
    A, B : torch.Tensor, optional
        Linear physical dynamics matrices. Both must be supplied together.
    lambda_dynamics : float, optional
        Weight on the mean squared normalized dynamics residual.
    w_mean, w_std : torch.Tensor, optional
        Trajectory normalization statistics. Identity normalization is used
        when these are omitted.
    x_scale : float, optional
        Minimum scale used in normalized initial-state mismatch.
    rho_x0_init : float, optional
        Initial augmented-Lagrangian penalty for the initial state.
    rho_x0_growth : float, optional
        Multiplicative penalty growth factor.
    rho_x0_max : float, optional
        Maximum initial-state penalty.
    constraint_tol : float, optional
        Feasibility tolerance on initial-state RMSE.
    max_outer_iter : int, optional
        Maximum augmented-Lagrangian outer iterations.
    inner_max_iter : int, optional
        Maximum Adam steps in each outer iteration.
    lr : float, optional
        Adam learning rate.
    min_inner_iter : int, optional
        Minimum total Adam steps before early stopping.
    patience : int, optional
        Consecutive non-improving steps allowed after feasibility.
    relative_loss_tol : float, optional
        Relative objective decrease required to reset patience.
    max_grad_norm : float, optional
        Maximum latent gradient norm.
    use_lbfgs_polish : bool, optional
        Whether to polish the best candidate with L-BFGS.
    lbfgs_max_iter : int, optional
        Maximum L-BFGS polishing iterations.
    store_history : bool, optional
        Whether to retain scalar inner-loop history.
    device : torch.device, optional
        Device used for optimization. Defaults to the decoder device.
    dtype : torch.dtype, optional
        Floating-point type used by solver-owned tensors.
    eps : float, optional
        Numerical floor for normalization denominators.
    """

    def __init__(
        self,
        decoder: nn.Module,
        x_dim: int,
        u_dim: int,
        horizon: int,
        Q: Optional[TensorLike] = None,
        R: Optional[TensorLike] = None,
        *,
        encoder: Optional[nn.Module] = None,
        x_ref: Optional[TensorLike] = None,
        u_ref: Optional[TensorLike] = None,
        Q_terminal: Optional[TensorLike] = None,
        u_bounds: Optional[Tuple[TensorLike, TensorLike]] = None,
        lambda_u_bounds: float = 1.0,
        lambda_alpha: float = 0.0,
        alpha_mean: Optional[TensorLike] = None,
        alpha_std: Optional[TensorLike] = None,
        A: Optional[TensorLike] = None,
        B: Optional[TensorLike] = None,
        lambda_dynamics: float = 0.0,
        w_mean: Optional[TensorLike] = None,
        w_std: Optional[TensorLike] = None,
        x_scale: float = 1.0,
        rho_x0_init: float = 1.0,
        rho_x0_growth: float = 10.0,
        rho_x0_max: float = 1e6,
        constraint_tol: float = 1e-3,
        max_outer_iter: int = 5,
        inner_max_iter: int = 200,
        lr: float = 1e-2,
        min_inner_iter: int = 10,
        patience: int = 20,
        relative_loss_tol: float = 1e-6,
        max_grad_norm: float = 10.0,
        use_lbfgs_polish: bool = False,
        lbfgs_max_iter: int = 20,
        store_history: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        """Initialize a frozen-decoder latent MPC solver.

        Notes
        -----
        Parameter meanings, defaults, and coordinate conventions are
        documented on :class:`LatentBehaviorMPCSolver`. Initialization freezes
        the decoder and optional encoder immediately.
        """
        if min(x_dim, u_dim, horizon, max_outer_iter) <= 0:
            raise ValueError("dimensions, horizon, and max_outer_iter must be positive.")
        self.decoder = decoder
        self.encoder = encoder
        self.x_dim, self.u_dim, self.horizon = int(x_dim), int(u_dim), int(horizon)
        try:
            decoder_device = next(decoder.parameters()).device
        except StopIteration:
            decoder_device = torch.device("cpu")
        self.device, self.dtype = device or decoder_device, dtype
        self.w_dim = (horizon + 1) * x_dim + horizon * u_dim
        if getattr(decoder, "w_dim", self.w_dim) != self.w_dim:
            raise ValueError("Decoder output dimension is inconsistent with the horizon.")
        self.Q = self._tensor(Q if Q is not None else torch.eye(x_dim), (x_dim, x_dim))
        self.R = self._tensor(R if R is not None else torch.eye(u_dim), (u_dim, u_dim))
        self.Q_terminal = self._tensor(
            Q_terminal if Q_terminal is not None else self.Q, (x_dim, x_dim)
        )
        self.x_ref = self._tensor(
            x_ref if x_ref is not None else torch.zeros(x_dim), (x_dim,)
        )
        self.u_ref = self._tensor(
            u_ref if u_ref is not None else torch.zeros(u_dim), (u_dim,)
        )
        self.u_bounds = None
        if u_bounds is not None:
            self.u_bounds = (
                torch.as_tensor(u_bounds[0], device=self.device, dtype=dtype),
                torch.as_tensor(u_bounds[1], device=self.device, dtype=dtype),
            )
            for bound in self.u_bounds:
                if bound.numel() not in (1, u_dim):
                    raise ValueError("Input bounds must be scalar or per-input.")
        self.A = None if A is None else self._tensor(A, (x_dim, x_dim))
        self.B = None if B is None else self._tensor(B, (x_dim, u_dim))
        if (self.A is None) != (self.B is None):
            raise ValueError("A and B must be supplied together.")
        self.lambda_u_bounds = float(lambda_u_bounds)
        self.lambda_alpha = float(lambda_alpha)
        self.lambda_dynamics = float(lambda_dynamics)
        self.alpha_mean = self._optional(alpha_mean)
        self.alpha_std = self._optional(alpha_std)
        self.w_mean = self._normalizer(w_mean, 0.0)
        self.w_std = self._normalizer(w_std, 1.0)
        self.x_scale, self.rho_x0_init = float(x_scale), float(rho_x0_init)
        self.rho_x0_growth, self.rho_x0_max = float(rho_x0_growth), float(rho_x0_max)
        self.constraint_tol, self.max_outer_iter = float(constraint_tol), int(max_outer_iter)
        self.inner_max_iter, self.lr = int(inner_max_iter), float(lr)
        self.min_inner_iter, self.patience = int(min_inner_iter), int(patience)
        self.relative_loss_tol, self.max_grad_norm = float(relative_loss_tol), float(max_grad_norm)
        self.use_lbfgs_polish, self.lbfgs_max_iter = bool(use_lbfgs_polish), int(lbfgs_max_iter)
        self.store_history, self.eps = bool(store_history), float(eps)
        decoder.eval()
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        if encoder is not None:
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
        self.last_result: Optional[LatentMPCSolution] = None

    def _tensor(self, value: TensorLike, shape: Tuple[int, ...]) -> TensorLike:
        """Convert and validate a solver tensor.

        Parameters
        ----------
        value : torch.Tensor or array-like
            Value to convert to the solver device and data type.
        shape : tuple of int
            Required output shape.

        Returns
        -------
        torch.Tensor
            Converted tensor.

        Raises
        ------
        ValueError
            If the converted tensor does not have ``shape``.
        """
        result = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tuple(result.shape) != shape:
            raise ValueError(f"Expected shape {shape}, got {tuple(result.shape)}.")
        return result

    def _optional(self, value: Optional[TensorLike]) -> Optional[TensorLike]:
        """Convert an optional value to solver tensor coordinates.

        Parameters
        ----------
        value : torch.Tensor or array-like or None
            Optional value to convert.

        Returns
        -------
        torch.Tensor or None
            Converted tensor, or ``None`` when no value was supplied.
        """
        return None if value is None else torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _normalizer(self, value: Optional[TensorLike], default: float) -> TensorLike:
        """Create and validate a trajectory normalization vector.

        Parameters
        ----------
        value : torch.Tensor or array-like or None
            Explicit vector with shape ``(w_dim,)``.
        default : float
            Fill value used when ``value`` is omitted.

        Returns
        -------
        torch.Tensor
            Normalization vector with shape ``(w_dim,)``.
        """
        result = torch.full((self.w_dim,), default, device=self.device, dtype=self.dtype)
        if value is not None:
            result = self._tensor(value, (self.w_dim,))
        return result

    def normalize_w(self, w: TensorLike) -> TensorLike:
        """Convert physical trajectory vectors to training coordinates.

        Parameters
        ----------
        w : torch.Tensor
            Physical behavior vector with trailing dimension ``w_dim``.

        Returns
        -------
        torch.Tensor
            Normalized behavior vector with the same shape as ``w``.
        """
        w = torch.as_tensor(w, device=self.device, dtype=self.dtype)
        return (w - self.w_mean) / torch.clamp(self.w_std, min=self.eps)

    def denormalize_w(self, w: TensorLike) -> TensorLike:
        """Convert training-coordinate trajectory vectors to physical units.

        Parameters
        ----------
        w : torch.Tensor
            Normalized behavior vector with trailing dimension ``w_dim``.

        Returns
        -------
        torch.Tensor
            Physical behavior vector with the same shape as ``w``.
        """
        return w * self.w_std + self.w_mean

    def encode_initial_trajectory(self, x_init: TensorLike, u_init: TensorLike) -> TensorLike:
        """Encode a physical-unit initialization trajectory.

        Parameters
        ----------
        x_init : torch.Tensor
            State seed with shape ``(horizon + 1, x_dim)``.
        u_init : torch.Tensor
            Input seed with shape ``(horizon, u_dim)``.

        Returns
        -------
        torch.Tensor
            Detached latent warm start with shape ``(alpha_dim,)``.

        Raises
        ------
        ValueError
            If no encoder was supplied or either trajectory has wrong shape.
        """
        if self.encoder is None:
            raise ValueError("An encoder is required for trajectory initialization.")
        x = self._tensor(x_init, (self.horizon + 1, self.x_dim))
        u = self._tensor(u_init, (self.horizon, self.u_dim))
        with torch.no_grad():
            return self.encoder(self.normalize_w(build_w(x, u))).reshape(-1).detach()

    def _decode(self, alpha: TensorLike) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Decode a latent coordinate and unpack its physical trajectory.

        Parameters
        ----------
        alpha : torch.Tensor
            Latent coordinate with shape ``(alpha_dim,)``.

        Returns
        -------
        w_hat : torch.Tensor
            Decoder output in training trajectory coordinates.
        x_seq : torch.Tensor
            Physical state sequence.
        u_seq : torch.Tensor
            Physical input sequence.
        """
        w_hat = self.decoder(alpha).reshape(-1)
        w_physical = self.denormalize_w(w_hat)
        x, u = unpack_w(
            w_physical, x_dim=self.x_dim, u_dim=self.u_dim, horizon=self.horizon
        )
        return w_hat, x, u

    def _components(
        self, alpha: TensorLike, x_current: TensorLike, multiplier: TensorLike, rho: float
    ) -> Tuple[TensorLike, Dict[str, TensorLike]]:
        """Evaluate the differentiable latent MPC objective and components.

        Parameters
        ----------
        alpha : torch.Tensor
            Current latent optimization variable.
        x_current : torch.Tensor
            Measured physical state with shape ``(x_dim,)``.
        multiplier : torch.Tensor
            Initial-state Lagrange multiplier with shape ``(x_dim,)``.
        rho : float
            Current augmented-Lagrangian penalty.

        Returns
        -------
        total : torch.Tensor
            Scalar augmented objective.
        components : dict
            Tensor-valued decoded trajectories, losses, violations, and
            residual diagnostics.
        """
        w_hat, x, u = self._decode(alpha)
        stage = quadratic_tracking_loss(
            x[:-1], u, self.Q, self.R, self.x_ref, self.u_ref
        )
        terminal_error = x[-1] - self.x_ref
        tracking = stage + terminal_error @ self.Q_terminal @ terminal_error
        residual = x[0] - x_current
        augmented = multiplier @ residual + 0.5 * rho * torch.sum(residual**2)
        if self.u_bounds is None:
            violation = torch.zeros_like(u)
        else:
            lower = torch.relu(self.u_bounds[0] - u)
            upper = torch.relu(u - self.u_bounds[1])
            violation = lower + upper
        bound_loss = torch.mean(violation**2)
        if self.alpha_mean is not None and self.alpha_std is not None:
            alpha_reg = torch.mean(
                ((alpha - self.alpha_mean) / torch.clamp(self.alpha_std.abs(), min=self.eps)) ** 2
            )
        else:
            alpha_reg = torch.mean(alpha**2)
        normalized_dynamics = None
        dynamics_loss = alpha.new_tensor(0.0)
        if self.A is not None and self.B is not None:
            residual_d = x[1:] - x[:-1] @ self.A.T - u @ self.B.T
            denominator = (
                torch.linalg.vector_norm(x[1:], dim=-1)
                + torch.linalg.vector_norm(x[:-1] @ self.A.T, dim=-1)
                + torch.linalg.vector_norm(u @ self.B.T, dim=-1)
            )
            normalized_dynamics = torch.linalg.vector_norm(residual_d, dim=-1) / torch.clamp(
                denominator, min=self.eps
            )
            dynamics_loss = torch.mean(normalized_dynamics**2)
        total = (
            tracking + augmented + self.lambda_u_bounds * bound_loss
            + self.lambda_alpha * alpha_reg + self.lambda_dynamics * dynamics_loss
        )
        return total, {
            "w_hat": w_hat, "x": x, "u": u, "tracking": tracking,
            "initial_residual": residual, "violation": violation,
            "bound_loss": bound_loss, "alpha_regularization": alpha_reg,
            "normalized_dynamics": normalized_dynamics,
            "dynamics_loss": dynamics_loss, "augmented_initial_loss": augmented,
        }

    def _solution(
        self, alpha: TensorLike, x_current: TensorLike, multiplier: TensorLike,
        rho: float, iterations: int, outer_iterations: int,
        history: List[Dict[str, float]],
    ) -> LatentMPCSolution:
        """Recompute and package diagnostics for a candidate latent point.

        Parameters
        ----------
        alpha : torch.Tensor
            Candidate latent coordinate.
        x_current : torch.Tensor
            Measured physical initial state.
        multiplier : torch.Tensor
            Current initial-state Lagrange multiplier.
        rho : float
            Current augmented-Lagrangian penalty.
        iterations : int
            Completed inner optimizer steps.
        outer_iterations : int
            Completed augmented-Lagrangian iterations.
        history : list of dict
            Stored scalar iteration history.

        Returns
        -------
        LatentMPCSolution
            Detached candidate and freshly recomputed diagnostics.
        """
        with torch.no_grad():
            loss, c = self._components(alpha, x_current, multiplier, rho)
            residual = c["initial_residual"]
            rmse = torch.sqrt(torch.mean(residual**2))
            nrmse = torch.linalg.vector_norm(residual) / max(
                float(torch.linalg.vector_norm(x_current)), self.x_scale, self.eps
            )
            violation = c["violation"]
            dynamics = c["normalized_dynamics"]
            finite = bool(torch.isfinite(loss) and torch.isfinite(alpha).all())
            return LatentMPCSolution(
                x=c["x"].detach().clone(), u=c["u"].detach().clone(),
                alpha=alpha.detach().clone(), w_hat=c["w_hat"].detach().clone(),
                loss=float(loss), tracking_loss=float(c["tracking"]),
                x0_rmse=float(rmse), x0_nrmse=float(nrmse),
                max_input_violation=float(violation.max()) if violation.numel() else 0.0,
                mean_input_violation=float(violation.mean()) if violation.numel() else 0.0,
                dynamics_residual_mean=None if dynamics is None else float(dynamics.mean()),
                dynamics_residual_p95=None if dynamics is None else float(torch.quantile(dynamics, .95)),
                feasible=finite and float(rmse) <= self.constraint_tol,
                iterations=iterations, outer_iterations=outer_iterations,
                history=history, finite=finite,
            )

    @staticmethod
    def is_better(candidate: LatentMPCSolution, incumbent: Optional[LatentMPCSolution]) -> bool:
        """Compare candidates using feasibility-first ordering.

        Parameters
        ----------
        candidate : LatentMPCSolution
            Candidate being considered.
        incumbent : LatentMPCSolution or None
            Current best solution.

        Returns
        -------
        bool
            ``True`` when ``candidate`` should replace ``incumbent``.
        """
        if incumbent is None:
            return candidate.finite
        if candidate.feasible != incumbent.feasible:
            return candidate.feasible
        if candidate.feasible:
            return candidate.tracking_loss < incumbent.tracking_loss
        return (candidate.x0_rmse, candidate.loss) < (incumbent.x0_rmse, incumbent.loss)

    def solve(
        self,
        x_current: TensorLike,
        *,
        alpha_init: Optional[TensorLike] = None,
        x_init: Optional[TensorLike] = None,
        u_init: Optional[TensorLike] = None,
    ) -> LatentMPCSolution:
        """Solve the latent MPC problem for a measured initial state.

        Parameters
        ----------
        x_current : torch.Tensor
            Measured physical state with shape ``(x_dim,)``.
        alpha_init : torch.Tensor, optional
            Explicit latent initialization. It takes precedence over encoder
            initialization.
        x_init : torch.Tensor, optional
            Physical state seed used with ``u_init`` and the encoder.
        u_init : torch.Tensor, optional
            Physical input seed used with ``x_init`` and the encoder.

        Returns
        -------
        LatentMPCSolution
            Best candidate found using feasibility-first selection.
        """
        current = self._tensor(x_current, (self.x_dim,))
        if alpha_init is None and x_init is not None and u_init is not None and self.encoder is not None:
            alpha_init = self.encode_initial_trajectory(x_init, u_init)
        alpha_dim = int(getattr(self.decoder, "alpha_dim"))
        initial = torch.zeros(alpha_dim, device=self.device, dtype=self.dtype)
        if alpha_init is not None:
            initial = self._tensor(alpha_init, (alpha_dim,))
        alpha = nn.Parameter(initial.detach().clone())
        multiplier = torch.zeros(self.x_dim, device=self.device, dtype=self.dtype)
        rho, previous_rmse = self.rho_x0_init, float("inf")
        history: List[Dict[str, float]] = []
        total_iterations = 0
        best: Optional[LatentMPCSolution] = None
        for outer in range(self.max_outer_iter):
            optimizer = torch.optim.Adam([alpha], lr=self.lr)
            best_inner, stale = float("inf"), 0
            for _ in range(self.inner_max_iter):
                optimizer.zero_grad(set_to_none=True)
                loss, components = self._components(alpha, current, multiplier, rho)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                if alpha.grad is None or not torch.isfinite(alpha.grad).all():
                    break
                torch.nn.utils.clip_grad_norm_([alpha], self.max_grad_norm)
                optimizer.step()
                total_iterations += 1
                current_loss = float(loss.detach())
                rmse = float(torch.sqrt(torch.mean(components["initial_residual"].detach() ** 2)))
                improvement = (
                    float("inf") if not np.isfinite(best_inner)
                    else (best_inner - current_loss) / max(abs(best_inner), self.eps)
                )
                if current_loss < best_inner and improvement > self.relative_loss_tol:
                    best_inner, stale = current_loss, 0
                else:
                    stale += 1
                if self.store_history:
                    history.append({"loss": current_loss, "x0_rmse": rmse, "rho_x0": rho})
                if total_iterations >= self.min_inner_iter and stale >= self.patience and rmse <= self.constraint_tol:
                    break
            candidate = self._solution(alpha, current, multiplier, rho, total_iterations, outer + 1, history)
            if self.is_better(candidate, best):
                best = candidate
            residual = candidate.x[0] - current
            multiplier = multiplier + rho * residual.detach()
            if candidate.x0_rmse <= self.constraint_tol:
                break
            if candidate.x0_rmse > 0.75 * previous_rmse:
                rho = min(rho * self.rho_x0_growth, self.rho_x0_max)
            previous_rmse = candidate.x0_rmse
        if self.use_lbfgs_polish and best is not None and best.finite:
            alpha = nn.Parameter(best.alpha.clone())
            optimizer = torch.optim.LBFGS([alpha], max_iter=self.lbfgs_max_iter)
            def closure() -> TensorLike:
                """Evaluate and differentiate the L-BFGS polishing objective.

                Returns
                -------
                torch.Tensor
                    Scalar latent MPC objective at the current L-BFGS point.
                """
                optimizer.zero_grad(set_to_none=True)
                value, _ = self._components(alpha, current, multiplier, rho)
                value.backward()
                torch.nn.utils.clip_grad_norm_([alpha], self.max_grad_norm)
                return value
            try:
                optimizer.step(closure)
                total_iterations += self.lbfgs_max_iter
                polished = self._solution(
                    alpha, current, multiplier, rho, total_iterations,
                    best.outer_iterations, history
                )
                if self.is_better(polished, best):
                    best = polished
            except RuntimeError:
                pass
        if best is None:
            best = self._solution(initial, current, multiplier, rho, total_iterations, 0, history)
        self.last_result = best
        return best

    def solve_multistart(
        self, x_current: TensorLike, alpha_initializations: Sequence[TensorLike]
    ) -> LatentMPCSolution:
        """Solve multiple latent starts sequentially.

        Parameters
        ----------
        x_current : torch.Tensor
            Measured physical state with shape ``(x_dim,)``.
        alpha_initializations : sequence of torch.Tensor
            Candidate latent initializations.

        Returns
        -------
        LatentMPCSolution
            Feasibility-first best solution across all starts. An empty
            sequence triggers the solver's zero initialization.
        """
        best = None
        for initial in alpha_initializations:
            candidate = self.solve(x_current, alpha_init=initial)
            if self.is_better(candidate, best):
                best = candidate
        if best is None:
            return self.solve(x_current)
        self.last_result = best
        return best


def curvature_penalty_exact(decoder: nn.Module, alpha_samples: TensorLike) -> TensorLike:
    """Estimate ``Omega(M_theta)`` with exact Hessians.

    Parameters
    ----------
    decoder : torch.nn.Module
        Map from latent coordinates to flattened trajectories.
    alpha_samples : torch.Tensor
        Sample points with shape ``(K, alpha_dim)`` or ``(alpha_dim,)``.

    Returns
    -------
    penalty : torch.Tensor
        Mean squared Frobenius norm of the output Hessians over samples.
    """

    samples = _as_sample_batch(alpha_samples)
    penalty = samples.new_tensor(0.0)

    for alpha in samples:
        alpha = alpha.detach().clone().requires_grad_(True)
        w_hat = decoder(alpha).reshape(-1)

        for output_i in range(w_hat.numel()):
            grad_i = torch.autograd.grad(
                w_hat[output_i],
                alpha,
                create_graph=True,
                retain_graph=True,
            )[0]

            for latent_j in range(alpha.numel()):
                hess_row = torch.autograd.grad(
                    grad_i[latent_j],
                    alpha,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                penalty = penalty + torch.sum(hess_row**2)

    return penalty / samples.shape[0]


def curvature_penalty_local(
    decoder: nn.Module,
    alpha_samples: TensorLike,
    eps: float = 1e-2,
) -> TensorLike:
    """Estimate curvature with a local first-order residual.

    Parameters
    ----------
    decoder : torch.nn.Module
        Map from latent coordinates to flattened trajectories.
    alpha_samples : torch.Tensor
        Sample points with shape ``(K, alpha_dim)`` or ``(alpha_dim,)``.
    eps : float, optional
        Standard deviation of the perturbation ``delta``.

    Returns
    -------
    penalty : torch.Tensor
        Mean squared residual of ``phi(alpha + delta) - phi(alpha) -
        J(alpha) delta``.
    """

    samples = _as_sample_batch(alpha_samples)
    penalty = samples.new_tensor(0.0)

    for alpha in samples:
        alpha = alpha.detach().clone().requires_grad_(True)
        delta = float(eps) * torch.randn_like(alpha)
        phi_alpha = decoder(alpha)
        jvp = torch.autograd.functional.jvp(decoder, alpha, delta, create_graph=True)[1]
        residual = decoder(alpha + delta) - phi_alpha - jvp
        penalty = penalty + torch.mean(residual**2)

    return penalty / samples.shape[0]


def _as_sample_batch(alpha_samples: TensorLike) -> TensorLike:
    if alpha_samples.ndim == 1:
        return alpha_samples.reshape(1, -1)
    if alpha_samples.ndim != 2:
        raise ValueError("alpha_samples must have shape (alpha_dim,) or (K, alpha_dim).")
    return alpha_samples


def _to_optional_device(
    tensor: Optional[TensorLike],
    reference: TensorLike,
) -> Optional[TensorLike]:
    if tensor is None:
        return None
    return tensor.to(device=reference.device, dtype=reference.dtype)


def _detach_loss_dict(loss_dict: Mapping[str, TensorLike]) -> Dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in loss_dict.items()
    }


# Inverted Pendulum Callers # TODO: Move them out, this is for core functions only

def manifold_u_caller(
    decoder,
    M,
    H=25,
    dt=0.05,
    x_dim=4,
    u_dim=1,
    alpha_dim=8,
    Q=None,
    R=None,
    x_ref=None,
    u_ref=None,
    umax=10.0,
    lambda_theta=1.0,
    lambda_curvature=1.0,
    lr=1e-2,
    max_iter=1000,
):
    """
    Returns u_caller(t, y), where y is normalized state and u is normalized input.
    """

    device = next(decoder.parameters()).device

    if Q is None:
        Q = torch.diag(torch.tensor([1.0, 1.0, 80.0, 10.0], device=device))
    if R is None:
        R = torch.tensor([[0.1]], device=device)
    if x_ref is None:
        x_ref = torch.zeros(x_dim, device=device)
    else:
        x_ref = torch.as_tensor(x_ref, dtype=torch.float32, device=device)

    if u_ref is None:
        u_ref = torch.zeros(u_dim, device=device)
    else:
        u_ref = torch.as_tensor(u_ref, dtype=torch.float32, device=device)

    for p in decoder.parameters():
        p.requires_grad_(False)

    state = {
        "next_update_t": None,
        "u_seq": np.zeros((H, u_dim)),
        "x_seq": None,
        "alpha": torch.zeros(alpha_dim, device=device),
        "current_u": 0.0,
    }

    def u_caller(t, y):
        y = np.asarray(y, dtype=float).reshape(x_dim)

        if state["next_update_t"] is None or t >= state["next_update_t"] - 1e-12:
            x_init = torch.zeros(H + 1, x_dim, device=device)
            x_init[0] = torch.tensor(y, dtype=torch.float32, device=device)

            # warm start predicted states if available
            if state["x_seq"] is not None:
                x_prev = state["x_seq"]
                x_init[:-1] = torch.tensor(x_prev[1:], dtype=torch.float32, device=device)
                x_init[-1] = x_init[-2]

            u_init = torch.tensor(state["u_seq"], dtype=torch.float32, device=device)
            alpha_init = state["alpha"].detach().clone()

            solver = BehaviorManifoldControlSolver(
                decoder=decoder,
                x_dim=x_dim,
                u_dim=u_dim,
                horizon=H,
                Q=Q,
                R=R,
                x_ref=x_ref,
                u_ref=u_ref,
                lambda_theta=lambda_theta,
                lambda_curvature=lambda_curvature,
                lr=lr,
                max_iter=max_iter,
                u_bounds=(-umax, umax),
                curvature_mode="local",
                device=device,
            )

            sol = solver.solve(
                x_init=x_init,
                u_init=u_init,
                alpha_init=alpha_init,
                freeze={ # Fix the decoder weights which is the trajectory manifold. Alpha is free parameter which is the latent coordinates on the manifold
                    "theta": True, 
                    "x": False,
                    "u": False,
                    "alpha": False,
                },
            )

            u_opt = sol.u.detach().cpu().numpy()
            x_opt = sol.x.detach().cpu().numpy()

            state["current_u"] = float(u_opt[0, 0])
            state["u_seq"] = np.vstack([u_opt[1:], u_opt[-1:]])
            state["x_seq"] = x_opt
            state["alpha"] = sol.alpha.detach().clone()
            state["next_update_t"] = t + dt

        return float(np.clip(state["current_u"], -umax, umax))

    return u_caller


def manifold_F_caller(
    decoder,
    M,
    m,
    g,
    l,
    H=25,
    dt=0.05,
    x_ref=None,
    u_ref=0.0,
    umax=10.0,
    **kwargs,
):
    """
    Returns F_caller(t, y_phys), matching the physical-unit interface expected by simulate().

    ``umax`` is the bound on normalized input ``u = F / (m g)``. The
    corresponding physical force bound is ``umax * m * g``.
    """

    u_caller = manifold_u_caller(
        decoder=decoder,
        M=M / m,
        H=H,
        dt=dt / np.sqrt(l / g),
        x_ref=None if x_ref is None else np.asarray(x_ref) / np.array(
            [l, l / np.sqrt(l / g), 1.0, 1.0 / np.sqrt(l / g)]
        ),
        u_ref=np.array([u_ref]),
        umax=umax,
        **kwargs,
    )

    return wrap_u_caller_as_physical_F_caller(u_caller, m, g, l)


# Linear System Callers

def make_linear_manifold_u_caller(
    decoder: "BehaviorDecoder",
    Q: "torch.Tensor",
    R: "torch.Tensor",
    x_ref: "torch.Tensor",
    u_ref: "torch.Tensor",
    H: int,
    alpha_dim: int,
    umax: float,
    device: "torch.device",
    solve_lr: float = 1e-3,
    max_iter: int = 1000,
    lambda_theta: float = 1.0,
    lambda_curvature: float = 1.0,
    seed: int = 1234,
):
    """Build a receding-horizon manifold controller for a linear discrete system.

    Parameters
    ----------
    decoder : BehaviorDecoder
        Trained behavior decoder mapping latent codes to states/outputs.
    Q : torch.Tensor
        State cost matrix (torch.tensor, x_dim x x_dim).
    R : torch.Tensor
        Input cost matrix (torch.tensor, u_dim x u_dim).
    x_ref : torch.Tensor
        Reference state vector (torch.tensor, x_dim).
    u_ref : torch.Tensor
        Reference input vector (torch.tensor, u_dim).
    H : int
        Planning horizon (number of timesteps).
    alpha_dim : int
        Dimension of the latent code/behavior parameter alpha.
    umax : float
        Maximum (absolute) input value for saturation.
    device : torch.device or str
        Device to run torch tensors on (e.g. 'cpu' or 'cuda').
    solve_lr : float, optional
        Learning rate for the internal solver updating alpha (default 1e-3).
    max_iter : int, optional
        Maximum iterations for the internal solver (default 1000).
    lambda_theta : float, optional
        Regularization weight on parameter deviation (default 1.0).
    lambda_curvature : float, optional
        Regularization weight on curvature of the manifold (default 1.0).

    Returns
    -------
    callable
        Controller called as ``u_caller(k, x_k)`` returning a numpy array control input.
    """
    import numpy as np

    from src.manifold_control import BehaviorManifoldControlSolver

    device = torch.device(device)
    u_dim = u_ref.shape[0]
    x_dim = x_ref.shape[0]
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    state = {
        "u_seq": np.zeros((H, u_dim)),
        "x_seq": None,
        "alpha": 0.1 * torch.randn(
            alpha_dim,
            generator=g,
            device=device,
        ),
    }

    def u_caller(k: int, x: "np.ndarray") -> "np.ndarray":
        x = np.asarray(x, dtype=float).reshape(x_dim)

        x_init = torch.zeros(H + 1, x_dim, device=device)
        x_init[0] = torch.tensor(x, dtype=torch.float32, device=device)

        if state["x_seq"] is not None:
            x_prev = state["x_seq"]
            x_init[:-1] = torch.tensor(x_prev[1:], dtype=torch.float32, device=device)
            x_init[-1] = x_init[-2]

        u_init = torch.tensor(state["u_seq"], dtype=torch.float32, device=device)
        alpha_init = state["alpha"].detach().clone()

        solver = BehaviorManifoldControlSolver( #TODO: Move it out of the u_caller
            decoder=decoder,
            x_dim=x_dim,
            u_dim=u_dim,
            horizon=H,
            Q=Q,
            R=R,
            x_ref=x_ref,
            u_ref=u_ref,
            lambda_theta=lambda_theta,
            lambda_curvature=lambda_curvature,
            lr=solve_lr,
            max_iter=max_iter,
            u_bounds=(-umax, umax),
            curvature_mode="local",
            device=device,
        )

        sol = solver.solve(
            x_init=x_init,
            u_init=u_init,
            alpha_init=alpha_init,
            freeze={"theta": True, "x": False, "u": False, "alpha": False},
        )

        u_opt = sol.u.detach().cpu().numpy()
        x_opt = sol.x.detach().cpu().numpy()

        state["u_seq"] = np.vstack([u_opt[1:], u_opt[-1:]])
        state["x_seq"] = x_opt
        state["alpha"] = sol.alpha.detach().clone()

        return np.clip(u_opt[0], -umax, umax).reshape(u_dim)

    return u_caller


# Helpers
def make_decoder(
    *,
    alpha_dim: int,
    w_dim: int,
    hidden_dims: tuple[int, ...],
    device: "torch.device",
) -> "BehaviorDecoder":
    from torch import nn

    return BehaviorDecoder(
        alpha_dim=alpha_dim,
        w_dim=w_dim,
        hidden_dims=hidden_dims,
        activation=nn.Tanh,
    ).to(device)


def make_encoder(
    *,
    alpha_dim: int,
    w_dim: int,
    hidden_dims: tuple[int, ...],
    device: "torch.device",
) -> "BehaviorEncoder":
    """Construct the mirrored behavior encoder used during training."""
    from torch import nn

    return BehaviorEncoder(
        w_dim=w_dim,
        alpha_dim=alpha_dim,
        hidden_dims=hidden_dims,
        activation=nn.Tanh,
    ).to(device)



def train_decoder(
    W: "torch.Tensor",
    *,
    x_dim: int,
    u_dim: int,
    horizon: int,
    alpha_dim: int,
    hidden_dims: tuple[int, ...],
    epochs: int,
    max_iter: int,
    lr: float,
    print_every: int,
    checkpoint: Path,
    device: "torch.device",
    method: Literal["minibatch", "solver"] = "minibatch",
    batch_size: int = 256,
    lambda_alpha: float = 0.0,
) -> "BehaviorAutoencoder":
    """
    Train a paired behavior encoder and decoder.

    ``method="minibatch"`` (the default) trains an autoencoder with codes
    predicted by the encoder. ``method="solver"`` preserves the legacy
    per-sample latent-table training route and additionally fits the encoder to
    the optimized table.

    Parameters
    ----------
    W : torch.Tensor, shape (n_samples, w_dim)
        Training behavior vectors.

    x_dim : int
        State dimension.

    u_dim : int
        Input dimension.

    horizon : int
        Prediction horizon H.

    alpha_dim : int
        Latent manifold dimension.

    hidden_dims : tuple[int, ...]
        Hidden layer widths of the decoder MLP.

    epochs : int
        Number of passes over the training set.

    lr : float
        Optimizer learning rate.

    print_every : int
        Print progress every ``print_every`` epochs.

    checkpoint : pathlib.Path
        Output path for the trained decoder weights.

    device : torch.device
        Torch device.

    method : {"minibatch", "solver"}, optional
        Decoder trainer.  The minibatch trainer is normally much faster.

    batch_size : int, optional
        Number of trajectories in each minibatch. Ignored by the solver
        trainer. Values larger than the data set produce full-batch training.

    lambda_alpha : float, optional
        Weight of the mean squared latent-coordinate regularizer in minibatch
        mode. The solver mode does not add this term.

    Returns
    -------
    BehaviorAutoencoder
        Trained encoder and decoder pair.
    """
    import torch

    if W.ndim != 2 or W.shape[0] == 0:
        raise ValueError("W must have shape (n_samples, w_dim) with n_samples > 0.")
    expected_w_dim = (horizon + 1) * x_dim + horizon * u_dim
    if W.shape[1] != expected_w_dim:
        raise ValueError(
            f"W has width {W.shape[1]}, but x_dim={x_dim}, u_dim={u_dim}, "
            f"and horizon={horizon} require width {expected_w_dim}."
        )
    if method not in {"minibatch", "solver"}:
        raise ValueError("method must be 'minibatch' or 'solver'.")
    if epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if lambda_alpha < 0:
        raise ValueError("lambda_alpha must be non-negative.")

    W = W.to(device=device)

    decoder = make_decoder(
        alpha_dim=alpha_dim,
        w_dim=W.shape[1],
        hidden_dims=hidden_dims,
        device=device,
    ).to(dtype=W.dtype)
    encoder = make_encoder(
        alpha_dim=alpha_dim,
        w_dim=W.shape[1],
        hidden_dims=hidden_dims,
        device=device,
    ).to(dtype=W.dtype)

    print(
        f"Training W={tuple(W.shape)}, "
        f"method={method}, alpha_dim={alpha_dim}, "
        f"device={device}",
        flush=True,
    )
    if method == "minibatch":
        optimizer = torch.optim.Adam(
            [*encoder.parameters(), *decoder.parameters()], lr=lr
        )
        effective_batch_size = min(batch_size, len(W))

        for epoch in range(epochs):
            permutation = torch.randperm(len(W), device=device)
            epoch_loss_sum = 0.0
            epoch_fit_sum = 0.0

            for start in range(0, len(W), effective_batch_size):
                indices = permutation[start : start + effective_batch_size]
                optimizer.zero_grad(set_to_none=True)
                alpha_pred = encoder(W[indices])
                prediction = decoder(alpha_pred)
                fit_loss = torch.mean((W[indices] - prediction) ** 2)
                alpha_reg = torch.mean(alpha_pred**2)
                loss = fit_loss + lambda_alpha * alpha_reg
                loss.backward()
                optimizer.step()

                n_batch = indices.numel()
                epoch_loss_sum += float(loss.detach()) * n_batch
                epoch_fit_sum += float(fit_loss.detach()) * n_batch

            epoch_loss = epoch_loss_sum / len(W)
            epoch_fit = epoch_fit_sum / len(W)
            if print_every > 0 and epoch % print_every == 0:
                print(
                    f"epoch={epoch:04d}, loss={epoch_loss:.6f}, "
                    f"fit={epoch_fit:.6f}"
                )
    else:
        alpha_table = torch.nn.Parameter(
            0.1
            * torch.randn(
                W.shape[0], alpha_dim, dtype=W.dtype, device=device
            )
        )
        solver = BehaviorManifoldControlSolver(
            decoder=decoder,
            x_dim=x_dim,
            u_dim=u_dim,
            horizon=horizon,
            Q=torch.zeros(x_dim, x_dim, dtype=W.dtype, device=device),
            R=torch.zeros(u_dim, u_dim, dtype=W.dtype, device=device),
            lambda_theta=1.0,
            lambda_curvature=1.0,
            lr=lr,
            max_iter=max_iter,
            curvature_mode="local",
            device=device,
            dtype=W.dtype,
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_fit = 0.0
            for i in range(W.shape[0]):
                sol = solver.solve(
                    x_init=torch.zeros(
                        horizon + 1, x_dim, dtype=W.dtype, device=device
                    ),
                    u_init=torch.zeros(horizon, u_dim, dtype=W.dtype, device=device),
                    alpha_init=alpha_table[i].detach(),
                    w_target=W[i],
                    freeze={
                        "theta": False,
                        "x": True,
                        "u": True,
                        "alpha": False,
                    },
                )
                with torch.no_grad():
                    alpha_table[i].copy_(sol.alpha)
                epoch_loss += sol.loss
                epoch_fit += sol.loss_dict["fit"]

            epoch_loss /= len(W)
            epoch_fit /= len(W)
            if print_every > 0 and epoch % print_every == 0:
                print(
                    f"epoch={epoch:04d}, loss={epoch_loss:.6f}, "
                    f"fit={epoch_fit:.6f}"
                )

        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        for _ in range(max(1, epochs)):
            encoder_optimizer.zero_grad(set_to_none=True)
            encoder_loss = torch.mean((encoder(W) - alpha_table.detach()) ** 2)
            encoder_loss.backward()
            encoder_optimizer.step()

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "alpha_dim": alpha_dim,
            "w_dim": W.shape[1],
            "hidden_dims": tuple(hidden_dims),
        },
        checkpoint,
    )
    print(f"saved autoencoder checkpoint to {checkpoint}")

    return BehaviorAutoencoder(encoder=encoder, decoder=decoder)


def load_autoencoder(
    *,
    checkpoint: Path,
    device: "torch.device",
    alpha_dim: Optional[int] = None,
    w_dim: Optional[int] = None,
    hidden_dims: Optional[tuple[int, ...]] = None,
) -> "BehaviorAutoencoder":
    """Load an encoder-decoder checkpoint and validate architecture metadata.

    Legacy decoder-only state dictionaries cannot reconstruct an encoder and
    therefore produce a clear compatibility error.
    """
    import torch

    payload = torch.load(checkpoint, map_location=device)
    required = {
        "encoder_state_dict",
        "decoder_state_dict",
        "alpha_dim",
        "w_dim",
        "hidden_dims",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(
            "Checkpoint is a legacy decoder-only checkpoint; retrain it to "
            "create a conjugate encoder-decoder checkpoint."
        )

    saved_alpha_dim = int(payload["alpha_dim"])
    saved_w_dim = int(payload["w_dim"])
    saved_hidden_dims = tuple(int(dim) for dim in payload["hidden_dims"])
    requested = (alpha_dim, w_dim, hidden_dims)
    saved = (saved_alpha_dim, saved_w_dim, saved_hidden_dims)
    labels = ("alpha_dim", "w_dim", "hidden_dims")
    for label, expected, actual in zip(labels, requested, saved):
        if expected is not None and expected != actual:
            raise ValueError(
                f"Checkpoint {label}={actual!r} does not match requested "
                f"{label}={expected!r}."
            )

    encoder = make_encoder(
        alpha_dim=saved_alpha_dim,
        w_dim=saved_w_dim,
        hidden_dims=saved_hidden_dims,
        device=device,
    )
    decoder = make_decoder(
        alpha_dim=saved_alpha_dim,
        w_dim=saved_w_dim,
        hidden_dims=saved_hidden_dims,
        device=device,
    )
    encoder.load_state_dict(payload["encoder_state_dict"])
    decoder.load_state_dict(payload["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return BehaviorAutoencoder(encoder=encoder, decoder=decoder)


def load_decoder(
    *,
    checkpoint: Path,
    alpha_dim: int,
    w_dim: int,
    hidden_dims: tuple[int, ...],
    device: "torch.device",
) -> "BehaviorDecoder":
    """Load the decoder from a conjugate autoencoder checkpoint."""
    return load_autoencoder(
        checkpoint=checkpoint,
        alpha_dim=alpha_dim,
        w_dim=w_dim,
        hidden_dims=hidden_dims,
        device=device,
    ).decoder


def split_behavior_matrix(
    W: TensorLike,
    *,
    test_fraction: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[TensorLike, TensorLike]:
    """Reproducibly split behavior rows into training and test sets.

    Parameters
    ----------
    W : torch.Tensor, shape (n_samples, w_dim)
        Behavior matrix with trajectories stored as rows.
    test_fraction : float, optional
        Fraction of rows assigned to the test set.
    seed : int, optional
        Seed for the local permutation generator.

    Returns
    -------
    W_train, W_test : tuple of torch.Tensor
        Nonempty row subsets on the same device as ``W``.
    """

    if W.ndim != 2 or W.shape[0] < 2:
        raise ValueError("W must be two-dimensional with at least two rows.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")

    test_size = min(W.shape[0] - 1, max(1, round(W.shape[0] * test_fraction)))
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    indices = torch.randperm(W.shape[0], generator=generator)
    test_indices = indices[:test_size].to(W.device)
    train_indices = indices[test_size:].to(W.device)
    return W[train_indices], W[test_indices]


def behavior_matrix_rank_summary(
    W: TensorLike,
    *,
    tol: Optional[float] = None,
) -> Dict[str, object]:
    """Summarize centered and uncentered ranks of a row-oriented matrix.

    Parameters
    ----------
    W : array-like, shape (n_trajectories, w_dim)
        Behavior vectors stored as rows.
    tol : float, optional
        Singular-value tolerance passed to ``numpy.linalg.matrix_rank``.

    Returns
    -------
    dict
        Shape, centered and uncentered ranks, and leading singular values.
    """

    matrix = W.detach().cpu().numpy() if torch.is_tensor(W) else np.asarray(W)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("W must be a nonempty two-dimensional matrix.")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rank_kwargs = {} if tol is None else {"tol": tol}
    return {
        "shape": tuple(matrix.shape),
        "rank": int(np.linalg.matrix_rank(matrix, **rank_kwargs)),
        "centered_rank": int(np.linalg.matrix_rank(centered, **rank_kwargs)),
        "leading_singular_values": np.linalg.svd(matrix, compute_uv=False)[:5],
    }


def evaluate_behavior_autoencoder(
    encoder: nn.Module,
    decoder: nn.Module,
    W: TensorLike,
    *,
    A: TensorLike,
    B: TensorLike,
    x_dim: int,
    u_dim: int,
    horizon: int,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Evaluate reconstruction quality and linear-system consistency.

    Parameters
    ----------
    encoder, decoder : torch.nn.Module
        Conjugate behavior encoder and decoder.
    W : torch.Tensor, shape (n_samples, w_dim)
        Behavior vectors stored as rows.
    A : array-like, shape (x_dim, x_dim)
        Linear state transition matrix.
    B : array-like, shape (x_dim, u_dim)
        Linear input matrix.
    x_dim, u_dim, horizon : int
        Trajectory dimensions and finite horizon.
    eps : float, optional
        Positive denominator floor.

    Returns
    -------
    dict of str to float
        Reconstruction NRMSE/R-squared metrics and normalized dynamics
        residual summaries for both data and reconstructions.
    """

    expected_w_dim = (horizon + 1) * x_dim + horizon * u_dim
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] != expected_w_dim:
        raise ValueError(
            f"W must have shape (n_samples, {expected_w_dim}) with n_samples > 0."
        )
    if eps <= 0:
        raise ValueError("eps must be positive.")
    parameter = next(encoder.parameters(), None)
    if parameter is None:
        W_eval = torch.as_tensor(W)
        if not W_eval.is_floating_point():
            W_eval = W_eval.to(dtype=torch.float32)
    else:
        W_eval = torch.as_tensor(W, device=parameter.device, dtype=parameter.dtype)
    A_t = torch.as_tensor(A, device=W_eval.device, dtype=W_eval.dtype)
    B_t = torch.as_tensor(B, device=W_eval.device, dtype=W_eval.dtype)
    if tuple(A_t.shape) != (x_dim, x_dim):
        raise ValueError(f"A must have shape {(x_dim, x_dim)}.")
    if tuple(B_t.shape) != (x_dim, u_dim):
        raise ValueError(f"B must have shape {(x_dim, u_dim)}.")

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        W_hat = decoder(encoder(W_eval))
        error = W_eval - W_hat
        aggregate_nrmse = torch.sqrt(torch.mean(error**2)) / torch.clamp(
            torch.sqrt(torch.mean((W_eval - torch.mean(W_eval)) ** 2)), min=eps
        )
        row_rmse = torch.sqrt(torch.mean(error**2, dim=1))
        row_scale = torch.sqrt(
            torch.mean((W_eval - torch.mean(W_eval, dim=1, keepdim=True)) ** 2, dim=1)
        )
        row_nrmse = row_rmse / torch.clamp(row_scale, min=eps)
        ss_res = torch.sum(error**2)
        ss_tot = torch.sum((W_eval - torch.mean(W_eval, dim=0)) ** 2)
        r2 = 1.0 - ss_res / torch.clamp(ss_tot, min=eps)

        def normalized_dynamics_residuals(matrix: TensorLike) -> TensorLike:
            x_size = (horizon + 1) * x_dim
            x_hat = matrix[:, :x_size].reshape(-1, horizon + 1, x_dim)
            u_hat = matrix[:, x_size:].reshape(-1, horizon, u_dim)
            ax = torch.einsum("ij,bhj->bhi", A_t, x_hat[:, :-1])
            bu = torch.einsum("ij,bhj->bhi", B_t, u_hat)
            residual = torch.linalg.vector_norm(x_hat[:, 1:] - ax - bu, dim=-1)
            denominator = (
                torch.linalg.vector_norm(x_hat[:, 1:], dim=-1)
                + torch.linalg.vector_norm(ax, dim=-1)
                + torch.linalg.vector_norm(bu, dim=-1)
            )
            return residual / torch.clamp(denominator, min=eps)

        data_residual = normalized_dynamics_residuals(W_eval).reshape(-1)
        reconstructed_residual = normalized_dynamics_residuals(W_hat).reshape(-1)

    row_nrmse_np = row_nrmse.cpu().numpy()

    def residual_summary(prefix: str, values: TensorLike) -> Dict[str, float]:
        values_np = values.cpu().numpy()
        return {
            f"{prefix}_mean": float(np.mean(values_np)),
            f"{prefix}_median": float(np.median(values_np)),
            f"{prefix}_p95": float(np.percentile(values_np, 95)),
            f"{prefix}_max": float(np.max(values_np)),
        }

    metrics = {
        "aggregate_nrmse": float(aggregate_nrmse.cpu()),
        "trajectory_nrmse_median": float(np.median(row_nrmse_np)),
        "trajectory_nrmse_p95": float(np.percentile(row_nrmse_np, 95)),
        "r2": float(r2.cpu()),
    }
    metrics.update(residual_summary("data_dynamics_residual", data_residual))
    metrics.update(
        residual_summary("reconstructed_dynamics_residual", reconstructed_residual)
    )
    return metrics


def build_trajectory_training_matrix(
    X_all: Sequence[Array],
    U_all: Sequence[Array],
    *,
    horizon: int,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """
    Build flattened trajectory windows for decoder training.

    Each row is

    ``w = [vec(x_0, ..., x_H), vec(u_0, ..., u_{H-1})]``.

    Parameters
    ----------
    X_all : sequence of np.ndarray
        State trajectories. Each entry has shape ``(x_dim, num_steps + 1)``.

    U_all : sequence of np.ndarray
        Input trajectories. Each entry has shape ``(u_dim, num_steps)``.

    horizon : int
        Window horizon ``H``.

    device : torch.device, optional
        Torch device for the returned tensor.

    dtype : torch.dtype, optional
        Torch dtype for the returned tensor. Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor, shape (n_windows, (H + 1) * x_dim + H * u_dim)
        Training matrix of flattened trajectory windows.
    """
    import torch

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if len(X_all) != len(U_all):
        raise ValueError("X_all and U_all must contain the same number of entries.")

    rows = []

    for X, U in zip(X_all, U_all):
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)

        if X.ndim != 2:
            raise ValueError("each X must have shape (x_dim, num_steps + 1).")
        if U.ndim != 2:
            raise ValueError("each U must have shape (u_dim, num_steps).")
        if X.shape[1] != U.shape[1] + 1:
            raise ValueError("each X must have exactly one more time point than U.")

        num_steps = U.shape[1]
        n_windows = num_steps - horizon + 1

        for k in range(max(0, n_windows)):
            x_seq = X[:, k : k + horizon + 1].T
            u_seq = U[:, k : k + horizon].T
            rows.append(np.concatenate([x_seq.reshape(-1), u_seq.reshape(-1)]))

    if not rows:
        raise RuntimeError(
            "no training windows were generated; try reducing horizon or "
            "increasing num_steps"
        )

    return torch.tensor(
        np.stack(rows),
        dtype=dtype or torch.float32,
        device=device,
    )




