"""Run manifold control on a randomly generated discrete-time linear system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.manifold_control import load_decoder, train_decoder
from src.linear_system import (
    finite_horizon_lqr_u_caller,
    generate_linear_trajectory_data,
    is_controllable,
    sample_controllable_linear_system,
    simulate_discrete_closed_loop,
)

from src.manifold_control import build_trajectory_training_matrix

if TYPE_CHECKING:
    import numpy as np
    from src.manifold_control import BehaviorDecoder


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    """
    Parse comma-separated hidden dimensions.

    Parameters
    ----------
    value : str
        Comma-separated list of positive integers.

    Returns
    -------
    tuple of int
        Hidden layer widths.
    """
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dims must be positive")
    return dims


def parse_float_list(value: str, expected_len: int, name: str) -> list[float]:
    """
    Parse a comma-separated list of floats.

    Parameters
    ----------
    value : str
        Comma-separated values.

    expected_len : int
        Required number of entries.

    name : str
        Argument name used in error messages.

    Returns
    -------
    list of float
        Parsed values.
    """
    items = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(items) != expected_len:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {expected_len} comma-separated values"
        )
    return items


def parse_vector(value: str) -> list[float]:
    """
    Parse a comma-separated vector.

    Parameters
    ----------
    value : str
        Comma-separated values.

    Returns
    -------
    list of float
        Parsed vector.
    """
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a random controllable linear system, train a behavior "
            "decoder, and run manifold control."
        )
    )

    parser.add_argument("--x-dim", type=int, default=4)
    parser.add_argument("--u-dim", type=int, default=1)
    parser.add_argument("--H", type=int, default=10, help="control horizon")
    parser.add_argument("--spectral-radius", type=float, default=0.95)

    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--process-noise-std", type=float, default=0.0)

    parser.add_argument("--alpha-dim", type=int, default=14)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saves") / "saved_models" / "behavior_decoder_linear.pt",
    )

    parser.add_argument("--solve-lr", type=float, default=1e-2)
    parser.add_argument("--lambda-theta", type=float, default=10.0)
    parser.add_argument("--lambda-curvature", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--umax", type=float, default=10.0)

    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--r-scale", type=float, default=0.1)
    parser.add_argument("--sim-steps", type=int, default=50)
    parser.add_argument("--x0", type=parse_vector, default=None)
    parser.add_argument("--x-ref", type=parse_vector, default=None)

    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("saves") / "simulation_results" / "linear_manifold_results.npz",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("saves") / "figures" / "linear_manifold_results.png",
    )

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


def make_linear_manifold_u_caller_args(
    args: argparse.Namespace,
    decoder: "BehaviorDecoder",
    Q: "torch.Tensor",
    R: "torch.Tensor",
    x_ref: "torch.Tensor",
    u_ref: "torch.Tensor",
):
    """
    Build a receding-horizon manifold controller for a linear discrete system.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    decoder : BehaviorDecoder
        Trained behavior decoder.

    Q : torch.Tensor
        State cost matrix.

    R : torch.Tensor
        Input cost matrix.

    x_ref : torch.Tensor
        Reference state.

    u_ref : torch.Tensor
        Reference input.

    Returns
    -------
    callable
        Controller called as ``u_caller(k, x_k)``.
    """
    import numpy as np

    from src.manifold_control import BehaviorManifoldControlSolver

    device = torch.device(args.device)

    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    state = {
        "u_seq": np.zeros((args.H, args.u_dim)),
        "x_seq": None,
        "alpha": torch.zeros(args.alpha_dim, device=device),
    }

    def u_caller(k: int, x: "np.ndarray") -> "np.ndarray":
        x = np.asarray(x, dtype=float).reshape(args.x_dim)

        x_init = torch.zeros(args.H + 1, args.x_dim, device=device)
        x_init[0] = torch.tensor(x, dtype=torch.float32, device=device)

        if state["x_seq"] is not None:
            x_prev = state["x_seq"]
            x_init[:-1] = torch.tensor(x_prev[1:], dtype=torch.float32, device=device)
            x_init[-1] = x_init[-2]

        u_init = torch.tensor(state["u_seq"], dtype=torch.float32, device=device)
        alpha_init = state["alpha"].detach().clone()

        solver = BehaviorManifoldControlSolver(
            decoder=decoder,
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            horizon=args.H,
            Q=Q,
            R=R,
            x_ref=x_ref,
            u_ref=u_ref,
            lambda_theta=args.lambda_theta,
            lambda_curvature=args.lambda_curvature,
            lr=args.solve_lr,
            max_iter=args.max_iter,
            u_bounds=(-args.umax, args.umax),
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

        return np.clip(u_opt[0], -args.umax, args.umax).reshape(args.u_dim)

    return u_caller


def solve_single_step(
    args: argparse.Namespace,
    decoder: "BehaviorDecoder",
    Q: "torch.Tensor",
    R: "torch.Tensor",
    x_ref: "torch.Tensor",
    u_ref: "torch.Tensor",
    x0: "np.ndarray",
) -> None:
    """
    Solve one frozen-decoder manifold-control problem.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    decoder : BehaviorDecoder
        Trained behavior decoder.

    Q : torch.Tensor
        State cost matrix.

    R : torch.Tensor
        Input cost matrix.

    x_ref : torch.Tensor
        Reference state.

    u_ref : torch.Tensor
        Reference input.

    x0 : np.ndarray, shape (x_dim,)
        Initial state.
    """
    import torch

    from src.manifold_control import BehaviorManifoldControlSolver

    device = torch.device(args.device)

    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    solver = BehaviorManifoldControlSolver(
        decoder=decoder,
        x_dim=args.x_dim,
        u_dim=args.u_dim,
        horizon=args.H,
        Q=Q,
        R=R,
        x_ref=x_ref,
        u_ref=u_ref,
        lambda_theta=args.lambda_theta,
        lambda_curvature=args.lambda_curvature,
        lr=args.solve_lr,
        max_iter=args.max_iter,
        u_bounds=(-args.umax, args.umax),
        curvature_mode="local",
        device=device,
    )

    x_init = torch.zeros(args.H + 1, args.x_dim, device=device)
    x_init[0] = torch.tensor(x0, dtype=torch.float32, device=device)
    u_init = torch.zeros(args.H, args.u_dim, device=device)
    alpha_init = torch.zeros(args.alpha_dim, device=device)

    solution = solver.solve(
        x_init=x_init,
        u_init=u_init,
        alpha_init=alpha_init,
        freeze={"theta": True, "x": False, "u": False, "alpha": False},
    )

    u_plan = solution.u.detach().cpu().numpy()
    print(f"  loss_dict={solution.loss_dict}")
    print(f"  first control={u_plan[0]}")


def run_simulation(
    args: argparse.Namespace,
    decoder: "BehaviorDecoder",
    A: "np.ndarray",
    B: "np.ndarray",
    Q: "torch.Tensor",
    R: "torch.Tensor",
    x_ref_t: "torch.Tensor",
    u_ref_t: "torch.Tensor",
    x0: "np.ndarray",
    x_ref: "np.ndarray",
) -> None:
    """
    Run closed-loop manifold control and LQR baseline simulations.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    decoder : BehaviorDecoder
        Trained behavior decoder.

    A : np.ndarray, shape (x_dim, x_dim)
        State transition matrix.

    B : np.ndarray, shape (x_dim, u_dim)
        Input matrix.

    Q : torch.Tensor
        State cost matrix.

    R : torch.Tensor
        Input cost matrix.

    x_ref_t : torch.Tensor
        Reference state tensor.

    u_ref_t : torch.Tensor
        Reference input tensor.

    x0 : np.ndarray, shape (x_dim,)
        Initial state.

    x_ref : np.ndarray, shape (x_dim,)
        Reference state.
    """
    import numpy as np

    u_manifold = make_linear_manifold_u_caller_args(
        args=args,
        decoder=decoder,
        Q=Q,
        R=R,
        x_ref=x_ref_t,
        u_ref=u_ref_t,
    )

    X_mani, U_mani = simulate_discrete_closed_loop(
        A,
        B,
        u_manifold,
        x0,
        num_steps=args.sim_steps,
    )

    u_lqr = finite_horizon_lqr_u_caller(
        A,
        B,
        args.q_scale * np.eye(args.x_dim),
        args.r_scale * np.eye(args.u_dim),
        horizon=max(args.sim_steps, args.H),
        x_ref=x_ref,
        umax=args.umax,
    )
    X_lqr, U_lqr = simulate_discrete_closed_loop(
        A,
        B,
        u_lqr,
        x0,
        num_steps=args.sim_steps,
    )

    zero_u = lambda _k, _x: np.zeros(args.u_dim)
    X_zero, U_zero = simulate_discrete_closed_loop(
        A,
        B,
        zero_u,
        x0,
        num_steps=args.sim_steps,
    )

    print(f"  final ||x_manifold - x_ref||={np.linalg.norm(X_mani[:, -1] - x_ref):.6f}")
    print(f"  final ||x_lqr      - x_ref||={np.linalg.norm(X_lqr[:, -1] - x_ref):.6f}")
    print(f"  final ||x_zero     - x_ref||={np.linalg.norm(X_zero[:, -1] - x_ref):.6f}")

    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.results_path,
            A=A,
            B=B,
            x0=x0,
            x_ref=x_ref,
            X_manifold=X_mani,
            U_manifold=U_mani,
            X_lqr=X_lqr,
            U_lqr=U_lqr,
            X_zero=X_zero,
            U_zero=U_zero,
        )
        print(f"  saved simulation results to {args.results_path}")

    if args.plot:
        import matplotlib.pyplot as plt

        t = np.arange(args.sim_steps + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(t, np.linalg.norm(X_mani - x_ref.reshape(-1, 1), axis=0), label="manifold")
        plt.plot(t, np.linalg.norm(X_lqr - x_ref.reshape(-1, 1), axis=0), label="lqr")
        plt.plot(t, np.linalg.norm(X_zero - x_ref.reshape(-1, 1), axis=0), label="zero")
        plt.xlabel("step")
        plt.ylabel("state error norm")
        plt.legend()
        plt.tight_layout()
        plt.show()

        args.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot_path)


def main() -> None:
    """
    Run the full linear manifold-control workflow.

    Returns
    -------
    None
        Results are printed and optionally saved to disk.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    import numpy as np
    import torch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    W = None

    if args.x0 is None:
        x0 = np.zeros(args.x_dim)
        x0[0] = 1.0
    else:
        x0 = np.asarray(args.x0, dtype=float).reshape(args.x_dim)

    if args.x_ref is None:
        x_ref = np.zeros(args.x_dim)
    else:
        x_ref = np.asarray(args.x_ref, dtype=float).reshape(args.x_dim)

    u_ref = np.zeros(args.u_dim)

    Q = args.q_scale * torch.eye(args.x_dim, device=device)
    R = args.r_scale * torch.eye(args.u_dim, device=device)
    x_ref_t = torch.tensor(x_ref, dtype=torch.float32, device=device)
    u_ref_t = torch.tensor(u_ref, dtype=torch.float32, device=device)

    w_dim = (args.H + 1) * args.x_dim + args.H * args.u_dim

    if args.skip_training and not args.checkpoint.exists():
        raise FileNotFoundError(
            f"--skip-training requires an existing checkpoint at {args.checkpoint}"
        )

    print("Step 1/4: sampling controllable linear system")
    A, B = sample_controllable_linear_system(
        args.x_dim,
        args.u_dim,
        spectral_radius=args.spectral_radius,
        seed=args.seed,
    )
    rank, controllable = is_controllable(A, B)
    print(f"  controllability rank={rank}, full={controllable}")

    if args.skip_data_generation:
        print("Step 2/4: skipping trajectory data generation")
    else:
        print("Step 2/4: generating trajectory data")
        _, X_all, U_all = generate_linear_trajectory_data(
            A,
            B,
            num_steps=args.num_steps,
            n_repeats=args.n_repeats,
            process_noise_std=args.process_noise_std,
            seed=args.seed,
        )
        W = build_trajectory_training_matrix(
            X_all,
            U_all,
            horizon=args.H,
            device=device,
            dtype=torch.float32,
        )
        w_dim = W.shape[1]
        print(f"  training matrix shape={tuple(W.shape)}")

    if args.skip_training:
        print("Step 3/4: loading decoder checkpoint")
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
        print("Step 3/4: training manifold decoder")
        decoder = train_decoder(
            W,
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            horizon=args.H,
            alpha_dim=args.alpha_dim,
            hidden_dims=args.hidden_dims,
            epochs=args.epochs,
            max_iter=args.max_iter,
            lr=args.train_lr,
            print_every=args.print_every,
            checkpoint=args.checkpoint,
            device=device,
        )
        decoder.eval()

    if args.skip_control_solve:
        print("Step 4/4: skipping control solve and simulation")
    else:
        print("Step 4/4: solving and simulating manifold controller")
        solve_single_step(
            args=args,
            decoder=decoder,
            Q=Q,
            R=R,
            x_ref=x_ref_t,
            u_ref=u_ref_t,
            x0=x0,
        )

        if not args.skip_simulation:
            run_simulation(
                args=args,
                decoder=decoder,
                A=A,
                B=B,
                Q=Q,
                R=R,
                x_ref_t=x_ref_t,
                u_ref_t=u_ref_t,
                x0=x0,
                x_ref=x_ref,
            )

    print("Done.")


if __name__ == "__main__":
    main()


