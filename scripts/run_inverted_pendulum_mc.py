"""Run the normalized inverted-pendulum manifold-control workflow."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from tqdm import tqdm

from src.inverted_pendulum import (
    _physical_state_scale,
    gen_max_theta_data,
    linearize_upright_dynamics,
    lqr_ct,
    rk4_step,
)
from src.manifold_control import (
    LatentBehaviorMPCSolver,
    build_trajectory_training_matrix,
    load_autoencoder,
    split_behavior_matrix,
    train_decoder,
)

if TYPE_CHECKING:
    from src.manifold_control import BehaviorDecoder, BehaviorEncoder


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive hidden-layer widths."""
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dims must be positive integers")
    return dims


def parse_y0(value: str) -> list[float]:
    """Parse a four-component physical initial state."""
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("y0 must contain four comma-separated values")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=float, default=2.0, help="cart mass")
    parser.add_argument("--m", type=float, default=1.0, help="pendulum mass")
    parser.add_argument("--l", type=float, default=1.0, help="pendulum length")
    parser.add_argument("--g", type=float, default=9.81, help="gravity")
    parser.add_argument("--H", type=int, default=25, help="control horizon")
    parser.add_argument("--control-dt", type=float, default=0.02, help="physical controller interval")
    parser.add_argument("--sim-duration", type=float, default=5.0, help="physical simulation duration")
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--theta-max", type=float, default=math.pi / 20.0)
    parser.add_argument("--alpha-dim", type=int, default=29)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128, 128))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lambda-alpha", type=float, default=1e-5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--training-max-iter", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--checkpoint", type=Path, default=Path("saves/saved_models/behavior_decoder_inverse_pendulum.pt"))
    parser.add_argument("--y0", type=parse_y0, default=[0.0, 0.0, math.pi / 40.0, 0.0])
    parser.add_argument("--umax", type=float, default=2.0, help="normalized bound; physical bound is umax*m*g")
    parser.add_argument("--control-lr", type=float, default=5e-3)
    parser.add_argument("--control-inner-max-iter", type=int, default=300)
    parser.add_argument("--control-outer-max-iter", type=int, default=5)
    parser.add_argument("--control-rho-x0", type=float, default=10.0)
    parser.add_argument("--control-rho-growth", type=float, default=10.0)
    parser.add_argument("--control-rho-max", type=float, default=1e7)
    parser.add_argument("--control-constraint-tol", type=float, default=2e-3)
    parser.add_argument("--control-lambda-u-bounds", type=float, default=1000.0)
    parser.add_argument("--control-lambda-alpha", type=float, default=1e-4)
    parser.add_argument("--control-patience", type=int, default=30)
    parser.add_argument("--control-relative-loss-tol", type=float, default=1e-6)
    parser.add_argument("--control-multistart", type=int, default=3)
    parser.add_argument("--control-lbfgs-polish", action="store_true")
    parser.add_argument("--results-path", type=Path, default=Path("saves/simulation_results/inverted_pendulum_results.npz"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=Path("saves/figures/inverted_pendulum_mc.png"))
    parser.add_argument("--skip-data-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-control-solve", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def simulate_discrete_inverted_pendulum(u_caller, M_ratio, y0, dt, num_steps, umax=np.inf):
    """Simulate normalized dynamics with one held controller call per step."""
    y = np.asarray(y0, dtype=float).reshape(4)
    X = np.empty((4, num_steps + 1), dtype=float)
    U = np.empty((1, num_steps), dtype=float)
    X[:, 0] = y
    for k in tqdm(range(num_steps), desc="Simulating"):
        u = float(np.asarray(u_caller(k, y)).reshape(-1)[0])
        u = float(np.clip(u, -umax, umax))
        U[0, k] = u
        y = rk4_step(y, u, dt=dt, M=M_ratio)
        X[:, k + 1] = y
    return np.arange(num_steps + 1, dtype=float) * dt, X, U


def evaluate_inverted_pendulum_autoencoder(encoder, decoder, W, *, horizon, dt, M_ratio, eps=1e-12):
    """Return reconstruction and nonlinear RK4 residual diagnostics."""
    with torch.no_grad():
        reconstructed = decoder(encoder(W))
    data = W.detach().cpu().numpy()
    recon = reconstructed.detach().cpu().numpy()
    error = recon - data
    aggregate = np.linalg.norm(error) / max(np.linalg.norm(data), eps)
    per_row = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(data, axis=1), eps)
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((data - data.mean(axis=0, keepdims=True)) ** 2)

    def residuals(matrix):
        X = matrix[:, : (horizon + 1) * 4].reshape(-1, horizon + 1, 4)
        U = matrix[:, (horizon + 1) * 4 :].reshape(-1, horizon, 1)
        values = []
        for trajectory, inputs in zip(X, U):
            for k in range(horizon):
                predicted = rk4_step(trajectory[k], inputs[k, 0], dt, M_ratio)
                denominator = max(np.linalg.norm(trajectory[k + 1]) + np.linalg.norm(predicted), eps)
                values.append(np.linalg.norm(trajectory[k + 1] - predicted) / denominator)
        values = np.asarray(values)
        return {name: float(function(values)) for name, function in (
            ("mean", np.mean), ("median", np.median), ("p95", lambda x: np.percentile(x, 95)), ("max", np.max)
        )}

    return {
        "aggregate_nrmse": float(aggregate),
        "trajectory_nrmse_median": float(np.median(per_row)),
        "trajectory_nrmse_p95": float(np.percentile(per_row, 95)),
        "r2": float(1.0 - ss_res / max(ss_tot, eps)),
        "data_residual": residuals(data),
        "reconstructed_residual": residuals(recon),
    }


def _print_metrics(label, metrics):
    print(f"{label} reconstruction: aggregate NRMSE={metrics['aggregate_nrmse']:.6g}, "
          f"trajectory median={metrics['trajectory_nrmse_median']:.6g}, "
          f"trajectory p95={metrics['trajectory_nrmse_p95']:.6g}, R^2={metrics['r2']:.6g}")
    for key, title in (("data_residual", "original data"), ("reconstructed_residual", "reconstructed")):
        value = metrics[key]
        print(f"  {title} dynamics residual: mean={value['mean']:.6g}, median={value['median']:.6g}, "
              f"p95={value['p95']:.6g}, max={value['max']:.6g}")


def make_inverted_pendulum_manifold_u_caller_args(
    args, decoder, Q, R, x_ref, u_ref, encoder=None, fallback_controller=None,
):
    """Build a persistent normalized latent-MPC controller."""
    device = torch.device(args.device)
    solver = LatentBehaviorMPCSolver(
        decoder, 4, 1, args.H, Q, R, encoder=encoder, x_ref=x_ref, u_ref=u_ref,
        Q_terminal=5.0 * Q, u_bounds=(-args.umax, args.umax),
        lambda_u_bounds=args.control_lambda_u_bounds,
        lambda_alpha=args.control_lambda_alpha, rho_x0_init=args.control_rho_x0,
        rho_x0_growth=args.control_rho_growth, rho_x0_max=args.control_rho_max,
        constraint_tol=args.control_constraint_tol,
        max_outer_iter=args.control_outer_max_iter,
        inner_max_iter=args.control_inner_max_iter, lr=args.control_lr,
        patience=args.control_patience, relative_loss_tol=args.control_relative_loss_tol,
        use_lbfgs_polish=args.control_lbfgs_polish, device=device,
    )
    state = {"previous_alpha": None, "previous_x_plan": None,
             "previous_u_plan": None, "previous_feasible_u_plan": None}
    keys = ("solver_iterations", "solver_outer_iterations", "solver_feasible",
            "solver_x0_rmse", "solver_x0_nrmse", "solver_max_input_violation",
            "solver_tracking_loss", "fallback_used")
    diagnostics = {key: [] for key in keys}
    generator = torch.Generator(device=device)
    generator.manual_seed(0 if args.seed is None else args.seed)

    def u_caller(k, x):
        current = torch.as_tensor(np.asarray(x).reshape(4), dtype=torch.float32, device=device)
        if state["previous_x_plan"] is None:
            x_seed = x_ref.expand(args.H + 1, -1).clone()
            u_seed = u_ref.expand(args.H, -1).clone()
        else:
            x_seed = torch.cat((state["previous_x_plan"][1:], state["previous_x_plan"][-1:])).clone()
            u_seed = torch.cat((state["previous_u_plan"][1:], state["previous_u_plan"][-1:])).clone()
        x_seed[0] = current
        starts = []
        if encoder is not None:
            starts.append(solver.encode_initial_trajectory(x_seed, u_seed))
        if state["previous_alpha"] is not None:
            starts.append(state["previous_alpha"])
        starts.append(torch.zeros(args.alpha_dim, device=device))
        while len(starts) < args.control_multistart:
            starts.append(starts[0] + 0.01 * torch.randn(starts[0].shape, generator=generator, device=device))
        sol = solver.solve_multistart(current, starts[:max(1, args.control_multistart)])
        usable = (sol.finite and sol.max_input_violation <= max(args.control_constraint_tol, 1e-6)
                  and sol.x0_rmse <= 10.0 * args.control_constraint_tol)
        if usable:
            control = sol.u[0].detach().cpu().numpy()
            state["previous_alpha"] = sol.alpha.detach().clone()
            state["previous_x_plan"], state["previous_u_plan"] = sol.x.detach().clone(), sol.u.detach().clone()
            if sol.feasible:
                state["previous_feasible_u_plan"] = sol.u.detach().clone()
        else:
            control = np.asarray(fallback_controller(k, x), dtype=float) if fallback_controller else np.array([np.nan])
            if not np.all(np.isfinite(control)) and state["previous_feasible_u_plan"] is not None:
                control = state["previous_feasible_u_plan"][min(1, args.H - 1)].cpu().numpy()
            if not np.all(np.isfinite(control)):
                control = np.zeros(1)
        values = (sol.iterations, sol.outer_iterations, sol.feasible, sol.x0_rmse,
                  sol.x0_nrmse, sol.max_input_violation, sol.tracking_loss, not usable)
        for key, value in zip(keys, values):
            diagnostics[key].append(value)
        return np.clip(control, -args.umax, args.umax).reshape(1)

    u_caller.solver, u_caller.state, u_caller.diagnostics = solver, state, diagnostics
    return u_caller


def _normalized_lqr_controller(args, x_ref):
    A_ct, B_ct = linearize_upright_dynamics(args.M / args.m)
    gain = lqr_ct(A_ct, B_ct, np.diag([1.0, 1.0, 100.0, 10.0]), np.array([[0.1]]))
    return lambda _k, x: np.clip(-gain @ (np.asarray(x) - x_ref), -args.umax, args.umax)


def solve_single_step(args, decoder, encoder, Q, R, x_ref, u_ref, y0_normalized):
    """Solve and print one normalized latent-MPC step."""
    solver = LatentBehaviorMPCSolver(
        decoder, 4, 1, args.H, Q, R, encoder=encoder, x_ref=x_ref, u_ref=u_ref,
        Q_terminal=5.0 * Q, u_bounds=(-args.umax, args.umax),
        lambda_u_bounds=args.control_lambda_u_bounds, lambda_alpha=args.control_lambda_alpha,
        rho_x0_init=args.control_rho_x0, rho_x0_growth=args.control_rho_growth,
        rho_x0_max=args.control_rho_max, constraint_tol=args.control_constraint_tol,
        max_outer_iter=args.control_outer_max_iter, inner_max_iter=args.control_inner_max_iter,
        lr=args.control_lr, patience=args.control_patience,
        relative_loss_tol=args.control_relative_loss_tol, device=torch.device(args.device),
    )
    x_init = x_ref.expand(args.H + 1, -1).clone()
    x_init[0] = torch.as_tensor(y0_normalized, dtype=torch.float32, device=args.device)
    u_init = u_ref.expand(args.H, -1).clone()
    alpha = solver.encode_initial_trajectory(x_init, u_init)
    sol = solver.solve(x_init[0], alpha_init=alpha)
    print(f"  tracking objective={sol.tracking_loss:.6g}")
    print(f"  initial-state RMSE={sol.x0_rmse:.6g}")
    print(f"  normalized initial-state mismatch={sol.x0_nrmse:.6g}")
    print(f"  maximum input-bound violation={sol.max_input_violation:.6g}")
    print(f"  inner/outer iterations={sol.iterations}/{sol.outer_iterations}")
    print(f"  feasible={sol.feasible}; first normalized control={sol.u[0].detach().cpu().numpy()}")


def run_simulation(args, decoder, encoder, Q, R, x_ref_t, u_ref_t, y0_normalized):
    """Run, summarize, save, and optionally plot three normalized controllers."""
    num_steps = round(args.sim_duration / args.control_dt)
    dt_norm = args.control_dt / math.sqrt(args.l / args.g)
    x_ref = x_ref_t.cpu().numpy()
    lqr_controller = _normalized_lqr_controller(args, x_ref)
    manifold = make_inverted_pendulum_manifold_u_caller_args(
        args, decoder, Q, R, x_ref_t, u_ref_t, encoder, lqr_controller)
    controllers = {"manifold": manifold, "lqr": lqr_controller, "zero": lambda _k, _x: np.zeros(1)}
    results = {name: simulate_discrete_inverted_pendulum(
        caller, args.M / args.m, y0_normalized, dt_norm, num_steps, args.umax)
        for name, caller in controllers.items()}
    for name, (_, X, _) in results.items():
        print(f"  final normalized state error ({name})={np.linalg.norm(X[:, -1] - x_ref):.6g}")
    diagnostics = {key: np.asarray(value) for key, value in manifold.diagnostics.items()}
    print(f"  feasible solve percentage={100 * diagnostics['solver_feasible'].mean():.1f}%")
    print(f"  fallback fraction={diagnostics['fallback_used'].mean():.6g}")
    print(f"  median/max x0 RMSE={np.median(diagnostics['solver_x0_rmse']):.6g}/{np.max(diagnostics['solver_x0_rmse']):.6g}")
    print(f"  median solver iterations={np.median(diagnostics['solver_iterations']):.1f}")
    print(f"  maximum normalized input violation={np.max(diagnostics['solver_max_input_violation']):.6g}")
    print(f"  maximum absolute normalized input={max(np.max(np.abs(item[2])) for item in results.values()):.6g}")
    print(f"  implied maximum physical force={args.umax * args.m * args.g:.6g}")

    t0, state_scale, mg = _physical_state_scale(args.m, args.g, args.l)
    payload = {"x_ref_normalized": x_ref, "y0_normalized": y0_normalized}
    for name, (t_norm, X_norm, U_norm) in results.items():
        payload.update({f"t_normalized_{name}": t_norm, f"X_normalized_{name}": X_norm,
                        f"U_normalized_{name}": U_norm, f"t_physical_{name}": t_norm * t0,
                        f"X_physical_{name}": X_norm * state_scale[:, None],
                        f"F_physical_{name}": U_norm * mg})
    payload.update(diagnostics)
    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.results_path, **payload)
        print(f"  saved simulation results to {args.results_path}")
    if args.plot:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for name, (t_norm, X, U) in results.items():
            time = t_norm * t0
            axes[0].plot(time, np.linalg.norm(X - x_ref[:, None], axis=0), label=name)
            axes[1].plot(time, X[2], label=name)
            axes[2].step(time[:-1], U[0], where="post", label=name)
        axes[0].set_ylabel("normalized state error")
        axes[1].set_ylabel("theta [rad]")
        axes[2].set_ylabel("normalized input"); axes[2].set_xlabel("physical time [s]")
        axes[0].legend(); figure.tight_layout()
        args.plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot_path)


def main() -> None:
    """Run data generation, training, diagnostics, and control."""
    args = build_arg_parser().parse_args()
    if args.alpha_dim != 4 + args.H:
        warnings.warn("alpha_dim differs from 4 + H, the full deterministic trajectory-manifold dimension for four states and one input.")
    if args.control_dt <= 0 or args.sim_duration <= 0:
        raise ValueError("control-dt and sim-duration must be positive")
    if args.seed is not None:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    num_steps = round(args.sim_duration / args.control_dt)
    num_points = num_steps
    _, state_scale, mg = _physical_state_scale(args.m, args.g, args.l)
    y0_normalized = np.asarray(args.y0, dtype=float) / state_scale
    x_ref_t = torch.zeros(4, dtype=torch.float32, device=device)
    u_ref_t = torch.zeros(1, dtype=torch.float32, device=device)
    Q = torch.diag(torch.tensor([1.0, 1.0, 100.0, 10.0], device=device))
    R = torch.tensor([[0.1]], device=device)
    W_train = W_test = None
    w_dim = (args.H + 1) * 4 + args.H
    if args.skip_training and not args.checkpoint.exists():
        raise FileNotFoundError(f"--skip-training requires {args.checkpoint}")
    if args.skip_data_generation:
        print("Step 1/4: skipping data generation")
    else:
        print("Step 1/4: generating RK4 trajectory data")
        _, X_all, F_all = gen_max_theta_data(
            args.M, args.m, args.g, args.l, sigma=args.sigma, theta_max=args.theta_max,
            t_span=(0.0, args.sim_duration), num_points=num_points,
            n_repeats=args.n_repeats, umax=args.umax, control_dt=args.control_dt,
            seed=args.seed, method="rk4")
        for X, F in zip(X_all, F_all):
            assert X.shape[1] == np.asarray(F).size + 1
        X_norm_all = [X / state_scale.reshape(4, 1) for X in X_all]
        U_norm_all = [np.asarray(F, dtype=float).reshape(1, -1) / mg for F in F_all]
        W = build_trajectory_training_matrix(X_norm_all, U_norm_all, horizon=args.H,
                                             device=device, dtype=torch.float32)
        assert W.shape[1] == (args.H + 1) * 4 + args.H
        W_train, W_test = split_behavior_matrix(W, test_fraction=args.test_fraction, seed=args.seed)
        w_dim = W.shape[1]
        print(f"  behavior matrix shape={tuple(W.shape)}; train={len(W_train)}, test={len(W_test)}")
    if args.skip_training:
        print("Step 2/4: loading autoencoder checkpoint")
        autoencoder = load_autoencoder(checkpoint=args.checkpoint, alpha_dim=args.alpha_dim,
                                       w_dim=w_dim, hidden_dims=args.hidden_dims, device=device)
    else:
        if W_train is None:
            raise RuntimeError("training requires generated data")
        print("Step 2/4: training behavior autoencoder")
        autoencoder = train_decoder(
            W_train, x_dim=4, u_dim=1, horizon=args.H, alpha_dim=args.alpha_dim,
            hidden_dims=args.hidden_dims, epochs=args.epochs, max_iter=args.training_max_iter,
            lr=args.train_lr, print_every=args.print_every, checkpoint=args.checkpoint,
            device=device, batch_size=args.batch_size, lambda_alpha=args.lambda_alpha)
    encoder, decoder = autoencoder.encoder, autoencoder.decoder
    encoder.eval(); decoder.eval()
    if W_train is not None:
        dt_norm = args.control_dt / math.sqrt(args.l / args.g)
        _print_metrics("Train", evaluate_inverted_pendulum_autoencoder(
            encoder, decoder, W_train, horizon=args.H, dt=dt_norm, M_ratio=args.M / args.m))
        _print_metrics("Test", evaluate_inverted_pendulum_autoencoder(
            encoder, decoder, W_test, horizon=args.H, dt=dt_norm, M_ratio=args.M / args.m))
    if args.skip_control_solve:
        print("Step 3/4: skipping one-step control solve")
    else:
        print("Step 3/4: solving normalized one-step latent MPC")
        solve_single_step(args, decoder, encoder, Q, R, x_ref_t, u_ref_t, y0_normalized)
    if args.skip_simulation:
        print("Step 4/4: skipping closed-loop simulation")
    else:
        print("Step 4/4: running normalized closed-loop comparison")
        run_simulation(args, decoder, encoder, Q, R, x_ref_t, u_ref_t, y0_normalized)
    print("Done.")


if __name__ == "__main__":
    main()
