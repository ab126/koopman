#!/usr/bin/env python
"""Run the manifold-control inverted-pendulum workflow from the notebook."""

from __future__ import annotations

import argparse
import math
import torch
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np

    from src.manifold_control import BehaviorDecoder


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dims must be positive")
    return dims


def parse_float_list(value: str, expected_len: int, name: str) -> list[float]:
    items = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(items) != expected_len:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {expected_len} comma-separated values"
        )
    return items


def parse_t_span(value: str) -> tuple[float, float]:
    start, stop = parse_float_list(value, 2, "t-span")
    if stop <= start:
        raise argparse.ArgumentTypeError("t-span stop must be greater than start")
    return start, stop


def parse_y0(value: str) -> list[float]:
    return parse_float_list(value, 4, "y0")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate data, train a behavior decoder, and run manifold control."
    )

    parser.add_argument("--M", type=float, default=2.0, help="slab mass")
    parser.add_argument("--m", type=float, default=1.0, help="rod mass")
    parser.add_argument("--l", type=float, default=1.0, help="rod length")
    parser.add_argument("--g", type=float, default=1.0, help="gravity")
    parser.add_argument("--H", type=int, default=25, help="control horizon")

    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--t-span", type=parse_t_span, default=(0.0, 2.0))
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--theta-max", type=float, default=math.pi / 20.0)
    parser.add_argument("--data-method", choices=("ivp", "rk4"), default="ivp")

    parser.add_argument("--alpha-dim", type=int, default=8)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128, 128))
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-alpha", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saves") / "saved_models" / "behavior_decoder_inverse_pendulum.pt",
    )

    parser.add_argument("--solve-lr", type=float, default=1e-2)
    parser.add_argument("--solve-max-iter", type=int, default=2000)
    parser.add_argument("--lambda-theta", type=float, default=1.0)
    parser.add_argument("--lambda-curvature", type=float, default=1.0)
    parser.add_argument("--solve-umax", type=float, default=20.0)
    parser.add_argument("--y0", type=parse_y0, default=[0.0, 0.0, 0.08, 0.0])

    parser.add_argument("--sim-umax", type=float, default=20.0)
    parser.add_argument("--sim-max-iter", type=int, default=1000)
    parser.add_argument("--sim-method", choices=("ivp", "rk4"), default="rk4")
    parser.add_argument("--results-path", type=Path, default=Path("saves") / "simulation_results" / "inverted_pendulum_results.npz")
    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--skip-data-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-control-solve", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device, for example 'cpu' or 'cuda'",
    )
    return parser


def build_training_matrix(
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

    from src.core import _physical_state_scale

    _, state_scale, mg = _physical_state_scale(m, g, l)
    rows = []

    for X, F in zip(X_all, F_all):
        Xn = X / state_scale.reshape(4, 1)
        un = F / mg
        horizon_count = Xn.shape[1] - H - 1

        for k in range(max(0, horizon_count)):
            x_seq = Xn[:, k : k + H + 1].T
            u_seq = un[k : k + H].reshape(H, 1)
            rows.append(np.concatenate([x_seq.reshape(-1), u_seq.reshape(-1)]))

    if not rows:
        raise RuntimeError(
            "no training windows were generated; try reducing H or increasing "
            "num-points/t-span"
        )

    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)


def make_decoder(
    *,
    alpha_dim: int,
    w_dim: int,
    hidden_dims: tuple[int, ...],
    device: "torch.device",
) -> "BehaviorDecoder":
    from torch import nn

    from src.manifold_control import BehaviorDecoder

    return BehaviorDecoder(
        alpha_dim=alpha_dim,
        w_dim=w_dim,
        hidden_dims=hidden_dims,
        activation=nn.Tanh,
    ).to(device)


def train_decoder(
    W: "torch.Tensor",
    *,
    alpha_dim: int,
    hidden_dims: tuple[int, ...],
    epochs: int,
    lr: float,
    lambda_alpha: float,
    print_every: int,
    checkpoint: Path,
    device: "torch.device",
) -> "BehaviorDecoder":
    import torch
    from torch import nn

    decoder = make_decoder(
        alpha_dim=alpha_dim,
        w_dim=W.shape[1],
        hidden_dims=hidden_dims,
        device=device,
    )
    alpha_table = nn.Parameter(0.1 * torch.randn(W.shape[0], alpha_dim, device=device))
    optimizer = torch.optim.Adam(list(decoder.parameters()) + [alpha_table], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        W_hat = decoder(alpha_table)
        recon_loss = torch.mean((W - W_hat) ** 2)
        alpha_reg = torch.mean(alpha_table**2)
        loss = recon_loss + lambda_alpha * alpha_reg
        loss.backward()
        optimizer.step()

        if print_every > 0 and epoch % print_every == 0:
            print(
                f"  epoch={epoch:04d}, "
                f"loss={loss.item():.6f}, "
                f"recon={recon_loss.item():.6f}"
            )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), checkpoint)
    print(f"  saved decoder checkpoint to {checkpoint}")
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


def solve_single_step(args: argparse.Namespace, decoder: "BehaviorDecoder") -> None:
    import torch

    from src.manifold_control import BehaviorManifoldControlSolver

    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    device = torch.device(args.device)
    Q = torch.diag(torch.tensor([1.0, 1.0, 80.0, 10.0], device=device))
    R = torch.tensor([[0.1]], device=device)
    x_ref = torch.zeros(4, device=device)
    u_ref = torch.zeros(1, device=device)

    solver = BehaviorManifoldControlSolver(
        decoder=decoder,
        x_dim=4,
        u_dim=1,
        horizon=args.H,
        Q=Q,
        R=R,
        x_ref=x_ref,
        u_ref=u_ref,
        lambda_theta=args.lambda_theta,
        lambda_curvature=args.lambda_curvature,
        lr=args.solve_lr,
        max_iter=args.solve_max_iter,
        u_bounds=(-args.solve_umax, args.solve_umax),
        curvature_mode="local",
        device=device,
    )

    x_init = torch.zeros(args.H + 1, 4, device=device)
    x_init[0] = torch.tensor(args.y0, dtype=torch.float32, device=device)
    u_init = torch.zeros(args.H, 1, device=device)
    alpha_init = torch.zeros(args.alpha_dim, device=device)

    solution = solver.solve(
        x_init=x_init,
        u_init=u_init,
        alpha_init=alpha_init,
        freeze={"theta": True, "x": False, "u": False, "alpha": False},
    )
    u_plan = solution.u.detach().cpu().numpy()
    print(f"  loss_dict={solution.loss_dict}")
    print(f"  first control={u_plan[0, 0]:.6f}")


def run_simulation(args: argparse.Namespace, decoder: "BehaviorDecoder") -> None:
    import numpy as np

    from src.core import simulate
    from src.manifold_control import manifold_F_caller

    dt = (args.t_span[1] - args.t_span[0]) / (args.num_points -1)
    F = manifold_F_caller(
        decoder=decoder,
        M=args.M,
        m=args.m,
        g=args.g,
        l=args.l,
        H=args.H,
        dt=dt,
        umax=args.sim_umax,
        max_iter=args.sim_max_iter,
    )

    t, x, x_dot, theta, theta_dot, F_seq = simulate(
        F=F,
        M=args.M,
        m=args.m,
        g=args.g,
        l=args.l,
        y0=args.y0,
        t_span=args.t_span,
        num_points=args.num_points,
        method=args.sim_method,
        verbose=True,
    )

    print(
        "  final state="
        f"[x={x[-1]:.6f}, x_dot={x_dot[-1]:.6f}, "
        f"theta={theta[-1]:.6f}, theta_dot={theta_dot[-1]:.6f}]"
    )

    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.results_path,
            t=t,
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            F=F_seq,
        )
        print(f"  saved simulation results to {args.results_path}")

    if args.plot: # Might plot to unintended location 
        from src.plotting import plot_results

        plot_results(t, x, x_dot, theta, theta_dot, F_seq) 


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    import torch
    import numpy as np

    from src.core import gen_max_theta_data

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    w_dim = (args.H + 1) * 4 + args.H
    W = None

    if args.skip_training and not args.checkpoint.exists():
        raise FileNotFoundError(
            f"--skip-training requires an existing checkpoint at {args.checkpoint}"
        )

    if args.skip_data_generation:
        print("Step 1/4: skipping data generation")
    else:
        print("Step 1/4: generating trajectory data")
        t_all, X_all, F_all = gen_max_theta_data(
            args.M,
            args.m,
            args.g,
            args.l,
            sigma=args.sigma,
            theta_max=args.theta_max,
            t_span=args.t_span,
            num_points=args.num_points,
            n_repeats=args.n_repeats,
            method=args.data_method,
        )
        W = build_training_matrix(
            X_all,
            F_all,
            H=args.H,
            m=args.m,
            g=args.g,
            l=args.l,
            device=device,
        )
        w_dim = W.shape[1]
        print(f"  training matrix shape={tuple(W.shape)}")

    if args.skip_training:
        print("Step 2/4: loading decoder checkpoint")
        decoder = load_decoder(
            checkpoint=args.checkpoint,
            alpha_dim=args.alpha_dim,
            w_dim=w_dim,
            hidden_dims=args.hidden_dims,
            device=device,
        )
    else:
        if W is None:
            raise RuntimeError("training requires generated data")
        print("Step 2/4: training manifold decoder")
        decoder = train_decoder(
            W,
            alpha_dim=args.alpha_dim,
            hidden_dims=args.hidden_dims,
            epochs=args.epochs,
            lr=args.train_lr,
            lambda_alpha=args.lambda_alpha,
            print_every=args.print_every,
            checkpoint=args.checkpoint,
            device=device,
        )
        decoder.eval()

    if args.skip_control_solve:
        print("Step 3/4: skipping open-loop control solve")
    else:
        print("Step 3/4: solving frozen-decoder open-loop control problem")
        solve_single_step(args, decoder)

    if args.skip_simulation:
        print("Step 4/4: skipping closed-loop simulation")
    else:
        print("Step 4/4: running closed-loop manifold-control simulation")
        run_simulation(args, decoder)

    print("Done.")


if __name__ == "__main__":
    main()
