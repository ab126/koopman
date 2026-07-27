import torch
from torch import nn

from src.manifold_control import (
    BehaviorDecoder,
    BehaviorEncoder,
    LatentBehaviorMPCSolver,
    build_w,
    unpack_w,
)


def test_unpack_w_inverts_build_w_and_supports_batches():
    x = torch.randn(5, 3)
    u = torch.randn(4, 2)
    x_out, u_out = unpack_w(build_w(x, u), x_dim=3, u_dim=2, horizon=4)
    assert torch.equal(x_out, x)
    assert torch.equal(u_out, u)
    batch = torch.stack((build_w(x, u), build_w(x + 1, u + 1)))
    assert unpack_w(batch, x_dim=3, u_dim=2, horizon=4)[0].shape == (2, 5, 3)


def test_solver_freezes_decoder_and_decoded_plan_is_exact():
    decoder = BehaviorDecoder(2, 7, hidden_dims=(4,))
    solver = LatentBehaviorMPCSolver(
        decoder, x_dim=1, u_dim=1, horizon=3, inner_max_iter=2, max_outer_iter=1
    )
    solution = solver.solve(torch.zeros(1))
    x, u = unpack_w(decoder(solution.alpha), x_dim=1, u_dim=1, horizon=3)
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert torch.equal(solution.x, x)
    assert torch.equal(solution.u, u)
    assert solution.u[0].shape == (1,)


def test_encoder_warm_start_shape_and_physical_denormalization():
    decoder = BehaviorDecoder(2, 5, hidden_dims=())
    encoder = BehaviorEncoder(5, 2, hidden_dims=())
    mean = torch.arange(5, dtype=torch.float32)
    solver = LatentBehaviorMPCSolver(
        decoder, 1, 1, 2, encoder=encoder, w_mean=mean, w_std=torch.ones(5),
        inner_max_iter=1, max_outer_iter=1,
    )
    alpha = solver.encode_initial_trajectory(torch.zeros(3, 1), torch.zeros(2, 1))
    assert alpha.shape == (2,)
    with torch.no_grad():
        decoder.net[0].weight.zero_()
        decoder.net[0].bias.zero_()
    solution = solver.solve(torch.zeros(1), alpha_init=alpha)
    assert torch.equal(build_w(solution.x, solution.u), mean)


class LinearTrajectoryDecoder(nn.Module):
    alpha_dim = 1
    w_dim = 5

    def forward(self, alpha):
        # x = [a, a, a], u = [0, 0], so x_{k+1}=x_k exactly.
        return torch.stack((alpha[0], alpha[0], alpha[0], alpha[0] * 0, alpha[0] * 0))


def test_augmented_constraint_and_linear_dynamics_residual():
    solver = LatentBehaviorMPCSolver(
        LinearTrajectoryDecoder(), 1, 1, 2, A=torch.ones(1, 1), B=torch.zeros(1, 1),
        rho_x0_init=10.0, inner_max_iter=100, max_outer_iter=3, lr=0.1,
        constraint_tol=1e-2,
    )
    solution = solver.solve(torch.ones(1))
    assert solution.x0_rmse < 0.1
    assert solution.dynamics_residual_mean < 1e-6


def test_input_bound_violation_diagnostics():
    decoder = LinearTrajectoryDecoder()
    solver = LatentBehaviorMPCSolver(
        decoder, 1, 1, 2, u_bounds=(-0.1, 0.1), inner_max_iter=0, max_outer_iter=1
    )
    # This decoder's inputs are zero, hence exactly within bounds.
    solution = solver.solve(torch.zeros(1), alpha_init=torch.ones(1))
    assert solution.max_input_violation == 0.0
