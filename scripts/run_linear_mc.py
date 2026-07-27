"""Run manifold control on a randomly generated discrete-time linear system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.manifold_control import (
    behavior_matrix_rank_summary,
    evaluate_behavior_autoencoder,
    load_autoencoder,
    split_behavior_matrix,
    train_decoder,
)
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lambda-alpha", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--rank-tol",
        type=float,
        default=None,
        help="matrix-rank tolerance (default: NumPy's automatic tolerance)",
    )
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
    parser.add_argument("--control-inner-max-iter", type=int, default=200)
    parser.add_argument("--control-outer-max-iter", type=int, default=5)
    parser.add_argument("--control-lr", type=float, default=1e-2)
    parser.add_argument("--control-rho-x0", type=float, default=1.0)
    parser.add_argument("--control-rho-growth", type=float, default=10.0)
    parser.add_argument("--control-rho-max", type=float, default=1e6)
    parser.add_argument("--control-constraint-tol", type=float, default=1e-3)
    parser.add_argument("--control-lambda-u-bounds", type=float, default=100.0)
    parser.add_argument("--control-lambda-alpha", type=float, default=0.0)
    parser.add_argument("--control-lambda-dynamics", type=float, default=0.0)
    parser.add_argument("--control-patience", type=int, default=20)
    parser.add_argument("--control-relative-loss-tol", type=float, default=1e-6)
    parser.add_argument("--control-multistart", type=int, default=1)
    parser.add_argument("--control-lbfgs-polish", action="store_true")

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
    encoder=None,
    A=None,
    B=None,
    fallback_controller=None,
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

    from src.manifold_control import LatentBehaviorMPCSolver

    device = torch.device(args.device)

    solver = LatentBehaviorMPCSolver(
        decoder, args.x_dim, args.u_dim, args.H, Q, R, encoder=encoder,
        x_ref=x_ref, u_ref=u_ref, u_bounds=(-args.umax, args.umax),
        lambda_u_bounds=args.control_lambda_u_bounds,
        lambda_alpha=args.control_lambda_alpha,
        A=A, B=B, lambda_dynamics=args.control_lambda_dynamics,
        rho_x0_init=args.control_rho_x0, rho_x0_growth=args.control_rho_growth,
        rho_x0_max=args.control_rho_max, constraint_tol=args.control_constraint_tol,
        max_outer_iter=args.control_outer_max_iter,
        inner_max_iter=args.control_inner_max_iter, lr=args.control_lr,
        patience=args.control_patience,
        relative_loss_tol=args.control_relative_loss_tol,
        use_lbfgs_polish=args.control_lbfgs_polish, device=device,
    )
    state = {
        "previous_solution": None, "previous_alpha": None,
        "previous_x_plan": None, "previous_u_plan": None,
        "previous_feasible_u_plan": None,
    }
    diagnostics = {key: [] for key in (
        "solver_iterations", "solver_outer_iterations", "solver_feasible",
        "solver_x0_rmse", "solver_x0_nrmse", "solver_max_input_violation",
        "solver_tracking_loss", "solver_dynamics_residual", "fallback_used",
    )}
    generator = torch.Generator(device=device)
    generator.manual_seed(0 if args.seed is None else args.seed)

    def u_caller(k: int, x: "np.ndarray") -> "np.ndarray":
        x = np.asarray(x, dtype=float).reshape(args.x_dim)

        current = torch.tensor(x, dtype=torch.float32, device=device)
        if state["previous_x_plan"] is None:
            x_seed = x_ref.expand(args.H + 1, -1).clone()
            x_seed[0] = current
            u_seed = u_ref.expand(args.H, -1).clone()
        else:
            x_seed = state["previous_x_plan"].clone()
            x_seed[:-1] = state["previous_x_plan"][1:]
            x_seed[-1] = state["previous_x_plan"][-1]
            x_seed[0] = current
            u_seed = state["previous_u_plan"].clone()
            u_seed[:-1] = state["previous_u_plan"][1:]
            u_seed[-1] = state["previous_u_plan"][-1]
        starts = []
        if encoder is not None:
            starts.append(solver.encode_initial_trajectory(x_seed, u_seed))
        if state["previous_alpha"] is not None:
            starts.append(state["previous_alpha"])
        starts.append(torch.zeros(args.alpha_dim, device=device))
        while len(starts) < args.control_multistart:
            base = starts[0]
            starts.append(base + 0.01 * torch.randn(base.shape, generator=generator, device=device))
        sol = solver.solve_multistart(current, starts[:max(1, args.control_multistart)])
        usable = (
            sol.finite
            and sol.x0_rmse <= 10.0 * args.control_constraint_tol
            and sol.max_input_violation <= max(args.control_constraint_tol, 1e-6)
        )
        fallback_used = not usable
        if usable:
            control = sol.u[0].detach().cpu().numpy()
            state["previous_x_plan"], state["previous_u_plan"] = sol.x, sol.u
            state["previous_alpha"], state["previous_solution"] = sol.alpha, sol
            if sol.feasible:
                state["previous_feasible_u_plan"] = sol.u.detach().clone()
        elif fallback_controller is not None:
            control = np.asarray(fallback_controller(k, x), dtype=float)
        elif state["previous_feasible_u_plan"] is not None:
            control = state["previous_feasible_u_plan"][min(1, args.H - 1)].cpu().numpy()
        else:
            control = u_ref.detach().cpu().numpy()
        values = (
            sol.iterations, sol.outer_iterations, sol.feasible, sol.x0_rmse,
            sol.x0_nrmse, sol.max_input_violation, sol.tracking_loss,
            np.nan if sol.dynamics_residual_mean is None else sol.dynamics_residual_mean,
            fallback_used,
        )
        for key, value in zip(diagnostics, values):
            diagnostics[key].append(value)
        return np.clip(control, -args.umax, args.umax).reshape(args.u_dim)

    u_caller.solver = solver
    u_caller.state = state
    u_caller.diagnostics = diagnostics
    return u_caller


def solve_single_step(
    args: argparse.Namespace,
    decoder: "BehaviorDecoder",
    Q: "torch.Tensor",
    R: "torch.Tensor",
    x_ref: "torch.Tensor",
    u_ref: "torch.Tensor",
    x0: "np.ndarray",
    encoder=None,
    A=None,
    B=None,
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

    from src.manifold_control import LatentBehaviorMPCSolver

    device = torch.device(args.device)

    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    solver = LatentBehaviorMPCSolver(
        decoder, args.x_dim, args.u_dim, args.H, Q, R, x_ref=x_ref, u_ref=u_ref,
        u_bounds=(-args.umax, args.umax),
        lambda_u_bounds=args.control_lambda_u_bounds,
        rho_x0_init=args.control_rho_x0, rho_x0_growth=args.control_rho_growth,
        rho_x0_max=args.control_rho_max, constraint_tol=args.control_constraint_tol,
        encoder=encoder, A=A, B=B, lambda_dynamics=args.control_lambda_dynamics,
        max_outer_iter=args.control_outer_max_iter,
        inner_max_iter=args.control_inner_max_iter, lr=args.control_lr, device=device,
    )

    x_init = torch.zeros(args.H + 1, args.x_dim, device=device)
    x_init[0] = torch.tensor(x0, dtype=torch.float32, device=device)
    u_init = torch.zeros(args.H, args.u_dim, device=device)
    alpha_init = torch.zeros(args.alpha_dim, device=device)

    if encoder is not None:
        x_init[1:] = x_ref
        alpha_init = solver.encode_initial_trajectory(x_init, u_init)
    solution = solver.solve(torch.as_tensor(x0, device=device), alpha_init=alpha_init)

    u_plan = solution.u.detach().cpu().numpy()
    print(f"  tracking objective={solution.tracking_loss:.6g}")
    print(f"  initial-state RMSE={solution.x0_rmse:.6g}")
    print(f"  normalized initial-state mismatch={solution.x0_nrmse:.6g}")
    print(f"  maximum input-bound violation={solution.max_input_violation:.6g}")
    print(f"  mean normalized dynamics residual={solution.dynamics_residual_mean}")
    print(f"  95th-percentile normalized dynamics residual={solution.dynamics_residual_p95}")
    print(f"  inner iterations={solution.iterations}")
    print(f"  outer iterations={solution.outer_iterations}")
    print(f"  feasible={solution.feasible}")
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
    encoder=None,
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

    u_lqr = finite_horizon_lqr_u_caller(
        A, B, args.q_scale * np.eye(args.x_dim),
        args.r_scale * np.eye(args.u_dim),
        horizon=max(args.sim_steps, args.H), x_ref=x_ref, umax=args.umax,
    )
    u_manifold = make_linear_manifold_u_caller_args(
        args=args,
        decoder=decoder,
        Q=Q,
        R=R,
        x_ref=x_ref_t,
        u_ref=u_ref_t,
        encoder=encoder, A=A, B=B, fallback_controller=u_lqr,
    )

    X_mani, U_mani = simulate_discrete_closed_loop(
        A,
        B,
        u_manifold,
        x0,
        num_steps=args.sim_steps,
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
    diagnostics = {key: np.asarray(value) for key, value in u_manifold.diagnostics.items()}
    print(f"  feasible solves={100.0 * diagnostics['solver_feasible'].mean():.1f}%")
    print(f"  median/max x0 RMSE={np.median(diagnostics['solver_x0_rmse']):.6g}/"
          f"{np.max(diagnostics['solver_x0_rmse']):.6g}")
    print(f"  median solve iterations={np.median(diagnostics['solver_iterations']):.1f}")
    print(f"  maximum input violation={np.max(diagnostics['solver_max_input_violation']):.6g}")

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
            **diagnostics,
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
    W_train = None
    W_test = None

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
        W_train, W_test = split_behavior_matrix(
            W, test_fraction=args.test_fraction, seed=args.seed
        )
        summaries = {
            "W_all": behavior_matrix_rank_summary(W, tol=args.rank_tol),
            "W_train": behavior_matrix_rank_summary(W_train, tol=args.rank_tol),
            "W_test": behavior_matrix_rank_summary(W_test, tol=args.rank_tol),
        }
        print("behavior matrix:")
        for name, summary in summaries.items():
            print(
                f"  {name} shape={summary['shape']}, "
                f"uncentered rank={summary['rank']}, "
                f"centered rank={summary['centered_rank']}"
            )
            singular_values = np.asarray(summary["leading_singular_values"])
            print(
                "    leading singular values="
                f"{np.array2string(singular_values, precision=3)}"
            )
        expected_behavior_dim = args.x_dim + args.H * args.u_dim
        print(
            "  expected noiseless behavior dimension="
            f"n + Hm={expected_behavior_dim}"
        )
        print(
            "  rank tolerance="
            f"{args.rank_tol if args.rank_tol is not None else 'NumPy default'}"
        )

    if args.skip_training:
        print("Step 3/4: loading autoencoder checkpoint")
        autoencoder = load_autoencoder(
            checkpoint=args.checkpoint,
            alpha_dim=args.alpha_dim,
            w_dim=w_dim,
            hidden_dims=args.hidden_dims,
            device=device,
        )
    else:
        if W_train is None:
            raise RuntimeError("training requires generated data")
        print("Step 3/4: training behavior autoencoder")
        autoencoder = train_decoder(
            W_train,
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
            batch_size=args.batch_size,
            lambda_alpha=args.lambda_alpha,
        )
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    encoder.eval()
    decoder.eval()

    if W_train is not None and W_test is not None:
        def print_autoencoder_metrics(label: str, matrix: "torch.Tensor") -> None:
            metrics = evaluate_behavior_autoencoder(
                encoder,
                decoder,
                matrix,
                A=A,
                B=B,
                x_dim=args.x_dim,
                u_dim=args.u_dim,
                horizon=args.H,
            )
            print(f"{label} autoencoder metrics:")
            print(
                "  reconstruction: "
                f"aggregate NRMSE={metrics['aggregate_nrmse']:.6g}, "
                f"trajectory median={metrics['trajectory_nrmse_median']:.6g}, "
                f"trajectory p95={metrics['trajectory_nrmse_p95']:.6g}, "
                f"R^2={metrics['r2']:.6g}"
            )
            for residual_label, prefix in (
                ("data dynamics residual", "data_dynamics_residual"),
                ("reconstructed dynamics residual", "reconstructed_dynamics_residual"),
            ):
                print(
                    f"  {residual_label}: "
                    f"mean={metrics[prefix + '_mean']:.6g}, "
                    f"median={metrics[prefix + '_median']:.6g}, "
                    f"p95={metrics[prefix + '_p95']:.6g}, "
                    f"max={metrics[prefix + '_max']:.6g}"
                )

        print_autoencoder_metrics("Train", W_train)
        print_autoencoder_metrics("Test", W_test)

    # Future DeePC comparison plan:
    # 1. Generate one shared noiseless persistently exciting trajectory set.
    # 2. Construct horizon-H DeePC Hankel matrices.
    # 3. Match x0, reference, Q/R, bounds, and horizon across all controllers.
    # 4. Compare model-based QR/LQR, linear representation, nonlinear decoder,
    #    and DeePC using first/full input, state, objective, feasibility, and
    #    dynamics differences; verify dimension n + Hm in the exact case.

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
            encoder=encoder, A=A, B=B,
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
                encoder=encoder,
            )

    print("Done.")


if __name__ == "__main__":
    main()


