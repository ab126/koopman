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


# Inverted Pendulum Callers

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
    """

    u_caller = manifold_u_caller(
        decoder=decoder,
        M=M / m,
        H=H,
        dt=dt / np.sqrt(l / g),
        x_ref=None if x_ref is None else np.asarray(x_ref) / np.array(
            [l, l / np.sqrt(l / g), 1.0, 1.0 / np.sqrt(l / g)]
        ),
        u_ref=np.array([u_ref / (m * g)]),
        umax=umax / (m * g),
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
) -> "BehaviorDecoder":
    """
    Train the behavior decoder and one latent coordinate per behavior vector.

    ``method="minibatch"`` (the default) jointly trains the decoder and latent
    table with one optimizer step per minibatch.  ``method="solver"`` preserves
    the original behavior: it invokes :class:`BehaviorManifoldControlSolver`
    once per row of ``W`` and performs ``max_iter`` steps in every invocation.

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
    BehaviorDecoder
        Trained decoder.
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
    )

    alpha_table = torch.nn.Parameter(
        torch.randn(
            W.shape[0],
            alpha_dim,
            dtype=W.dtype,
            device=device,
        )
        * 0.1
    )

    print(
        f"Training W={tuple(W.shape)}, "
        f"alpha_table={(W.shape[0], alpha_dim)}, "
        f"device={device}",
        flush=True,
    )
    if method == "minibatch":
        optimizer = torch.optim.Adam(
            [*decoder.parameters(), alpha_table], lr=lr
        )
        effective_batch_size = min(batch_size, len(W))

        for epoch in range(epochs):
            permutation = torch.randperm(len(W), device=device)
            epoch_loss_sum = 0.0
            epoch_fit_sum = 0.0

            for start in range(0, len(W), effective_batch_size):
                indices = permutation[start : start + effective_batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction = decoder(alpha_table[indices])
                fit_loss = torch.sum((W[indices] - prediction) ** 2)
                alpha_reg = torch.mean(alpha_table[indices] ** 2)
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

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), checkpoint)
    print(f"saved decoder checkpoint to {checkpoint}")

    return decoder


def load_decoder(
    *,
    checkpoint: Path,
    alpha_dim: int,
    w_dim: int,
    hidden_dims: tuple[int, ...],
    device: "torch.device",
) -> "BehaviorDecoder":
    import torch

    decoder = make_decoder(
        alpha_dim=alpha_dim,
        w_dim=w_dim,
        hidden_dims=hidden_dims,
        device=device,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    decoder.load_state_dict(state_dict)
    decoder.eval()
    return decoder


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




