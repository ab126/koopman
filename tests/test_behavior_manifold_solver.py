import torch

import src.manifold_control as manifold_control
from src.manifold_control import BehaviorDecoder, BehaviorManifoldControlSolver


def _solver(**kwargs):
    horizon = 2
    x_dim = 1
    u_dim = 1
    decoder = BehaviorDecoder(
        alpha_dim=2,
        w_dim=(horizon + 1) * x_dim + horizon * u_dim,
        hidden_dims=(4,),
    )
    defaults = {
        "decoder": decoder,
        "x_dim": x_dim,
        "u_dim": u_dim,
        "horizon": horizon,
        "lambda_curvature": 0.0,
        "max_iter": 5,
        "device": torch.device("cpu"),
    }
    defaults.update(kwargs)
    return BehaviorManifoldControlSolver(**defaults)


def test_early_stopping_respects_min_iter_and_reports_iterations():
    callback_iterations = []
    solver = _solver(
        lr=0.0,
        max_iter=50,
        min_iter=4,
        patience=2,
        relative_loss_tol=1e-6,
    )

    solution = solver.solve(
        freeze={"theta": True, "x": False, "u": False, "alpha": False},
        callback=lambda iteration, loss, parts: callback_iterations.append(iteration),
    )

    assert solution.iterations == 4
    assert solution.loss_dict["iterations"] == 4.0
    assert callback_iterations == [0, 1, 2, 3]


def test_initial_state_is_fixed_and_other_values_obey_bounds():
    solver = _solver(
        lr=0.1,
        x_bounds=(-0.1, 0.1),
        u_bounds=(-0.2, 0.2),
        max_iter=4,
    )
    x_init = torch.tensor([[2.0], [4.0], [-4.0]])
    u_init = torch.tensor([[3.0], [-3.0]])

    solution = solver.solve(x_init=x_init, u_init=u_init)

    torch.testing.assert_close(solution.x[0], x_init[0])
    assert torch.all(solution.x[1:].abs() <= 0.1)
    assert torch.all(solution.u.abs() <= 0.2)


def test_history_is_disabled_by_default():
    solution = _solver(max_iter=3).solve()

    assert solution.history == []
    assert solution.iterations == 3


def test_history_honors_store_every():
    solution = _solver(max_iter=5, store_history=True, store_every=2).solve()

    assert len(solution.history) == 3
    assert all(set(item) == {"qr", "fit", "curvature", "total"} for item in solution.history)


def test_zero_curvature_weight_skips_curvature_computation(monkeypatch):
    def unexpected_curvature_call(*args, **kwargs):
        raise AssertionError("curvature computation should have been skipped")

    monkeypatch.setattr(
        manifold_control, "curvature_penalty_exact", unexpected_curvature_call
    )
    _solver(lambda_curvature=0.0, curvature_mode="exact", max_iter=1).solve()


def test_solution_output_shapes_are_unchanged():
    solution = _solver(max_iter=1).solve()

    assert solution.x.shape == (3, 1)
    assert solution.u.shape == (2, 1)
    assert solution.alpha.shape == (2,)
    assert solution.w_hat.shape == (5,)
