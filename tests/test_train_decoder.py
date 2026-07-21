from pathlib import Path

import pytest
import torch

from src.manifold_control import BehaviorManifoldControlSolver, make_decoder, train_decoder


def _train_args(checkpoint: Path) -> dict:
    return {
        "x_dim": 1,
        "u_dim": 1,
        "horizon": 1,
        "alpha_dim": 2,
        "hidden_dims": (8,),
        "epochs": 2,
        "max_iter": 1,
        "lr": 1e-2,
        "print_every": 0,
        "checkpoint": checkpoint,
        "device": torch.device("cpu"),
    }


def test_minibatch_is_default_and_does_not_use_solver(tmp_path, monkeypatch):
    W = torch.randn(7, 3)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("minibatch training must not invoke the solver")

    monkeypatch.setattr(BehaviorManifoldControlSolver, "solve", fail_if_called)
    checkpoint = tmp_path / "decoder.pt"
    autoencoder = train_decoder(W, batch_size=3, **_train_args(checkpoint))

    assert checkpoint.exists()
    assert autoencoder.decoder(torch.zeros(4, 2)).shape == (4, 3)
    assert autoencoder.encoder(W[:4]).shape == (4, 2)


def test_minibatch_updates_decoder_parameters(tmp_path):
    W = torch.randn(6, 3)
    torch.manual_seed(123)
    initial = make_decoder(
        alpha_dim=2,
        w_dim=3,
        hidden_dims=(8,),
        device=torch.device("cpu"),
    )
    initial_params = [parameter.detach().clone() for parameter in initial.parameters()]

    torch.manual_seed(123)
    autoencoder = train_decoder(
        W, batch_size=2, **_train_args(tmp_path / "decoder.pt")
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_params, autoencoder.decoder.parameters())
    )


def test_solver_training_remains_available(tmp_path):
    W = torch.randn(2, 3)
    autoencoder = train_decoder(
        W,
        method="solver",
        **_train_args(tmp_path / "solver-decoder.pt"),
    )
    assert autoencoder.decoder(torch.zeros(2)).shape == (3,)


def test_train_decoder_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="method"):
        train_decoder(
            torch.randn(2, 3),
            method="unknown",
            **_train_args(tmp_path / "decoder.pt"),
        )
