import numpy as np
import torch
from torch import nn

from src.manifold_control import (
    BehaviorDecoder,
    BehaviorEncoder,
    behavior_matrix_rank_summary,
    evaluate_behavior_autoencoder,
    load_autoencoder,
    split_behavior_matrix,
    train_decoder,
)


def _exact_linear_behaviors(num_samples=8, horizon=3):
    generator = torch.Generator().manual_seed(7)
    x0 = torch.randn(num_samples, 1, generator=generator)
    u = torch.randn(num_samples, horizon, 1, generator=generator)
    states = [x0]
    for k in range(horizon):
        states.append(0.8 * states[-1] + 0.5 * u[:, k])
    x = torch.stack(states, dim=1).squeeze(2)
    return torch.cat([x, u.reshape(num_samples, -1)], dim=1)


def test_encoder_shapes_and_mirrored_hidden_dimensions():
    encoder = BehaviorEncoder(w_dim=11, alpha_dim=3, hidden_dims=(7, 5))
    linears = [layer for layer in encoder.net if isinstance(layer, nn.Linear)]

    assert encoder(torch.zeros(11)).shape == (3,)
    assert encoder(torch.zeros(4, 11)).shape == (4, 3)
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (11, 5),
        (5, 7),
        (7, 3),
    ]


def test_encoder_decoder_output_shape():
    encoder = BehaviorEncoder(9, 2, (6, 4))
    decoder = BehaviorDecoder(2, 9, (6, 4))
    assert decoder(encoder(torch.zeros(5, 9))).shape == (5, 9)


def test_perfect_reconstruction_metrics_and_exact_dynamics():
    W = _exact_linear_behaviors()
    metrics = evaluate_behavior_autoencoder(
        nn.Identity(),
        nn.Identity(),
        W,
        A=np.array([[0.8]]),
        B=np.array([[0.5]]),
        x_dim=1,
        u_dim=1,
        horizon=3,
    )

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["aggregate_nrmse"] == 0.0
    assert metrics["trajectory_nrmse_p95"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["data_dynamics_residual_max"] < 1e-6
    assert metrics["reconstructed_dynamics_residual_max"] < 1e-6


def test_perturbed_reconstruction_has_positive_dynamics_residual():
    class PerturbState(nn.Module):
        def forward(self, W):
            result = W.clone()
            result[:, 1] += 0.25
            return result

    W = _exact_linear_behaviors()
    metrics = evaluate_behavior_autoencoder(
        nn.Identity(),
        PerturbState(),
        W,
        A=np.array([[0.8]]),
        B=np.array([[0.5]]),
        x_dim=1,
        u_dim=1,
        horizon=3,
    )
    assert metrics["reconstructed_dynamics_residual_mean"] > 0.0


def test_split_is_deterministic_and_rank_uses_rows():
    W = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    train_a, test_a = split_behavior_matrix(W, test_fraction=0.25, seed=19)
    train_b, test_b = split_behavior_matrix(W, test_fraction=0.25, seed=19)

    torch.testing.assert_close(train_a, train_b)
    torch.testing.assert_close(test_a, test_b)
    summary = behavior_matrix_rank_summary(W)
    assert summary["shape"] == (12, 5)
    assert summary["rank"] == np.linalg.matrix_rank(W.numpy())
    assert summary["centered_rank"] == np.linalg.matrix_rank(
        W.numpy() - W.numpy().mean(axis=0, keepdims=True)
    )


def test_checkpoint_round_trip_reproduces_both_networks(tmp_path):
    torch.manual_seed(11)
    W = torch.randn(10, 3)
    checkpoint = tmp_path / "autoencoder.pt"
    trained = train_decoder(
        W,
        x_dim=1,
        u_dim=1,
        horizon=1,
        alpha_dim=2,
        hidden_dims=(5, 4),
        epochs=2,
        max_iter=1,
        lr=1e-2,
        print_every=0,
        checkpoint=checkpoint,
        device=torch.device("cpu"),
        batch_size=4,
    )
    loaded = load_autoencoder(checkpoint=checkpoint, device=torch.device("cpu"))

    with torch.no_grad():
        torch.testing.assert_close(loaded.encoder(W), trained.encoder(W))
        alpha = torch.randn(6, 2)
        torch.testing.assert_close(loaded.decoder(alpha), trained.decoder(alpha))
