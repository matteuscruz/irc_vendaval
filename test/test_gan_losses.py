"""Testes das perdas WGAN-GP (src/gan/losses.py)."""

import torch

from src.gan.losses import (
    get_crit_loss,
    get_gen_loss,
    get_gradient,
    gradient_penalty,
)
from src.gan.models import Critic


def test_gradient_penalty_zero_for_unit_norm():
    # Gradiente com norma exatamente 1 -> penalidade ≈ 0
    grad = torch.zeros(8, 1)
    grad[:, 0] = 1.0
    assert gradient_penalty(grad).item() == 0.0


def test_gradient_penalty_positive_for_nonunit_norm():
    grad = torch.full((8, 1), 3.0)
    assert gradient_penalty(grad).item() > 0.0


def test_get_gen_loss_is_negative_mean():
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    assert torch.isclose(get_gen_loss(pred), torch.tensor(-2.0))


def test_get_crit_loss_signs():
    fake = torch.tensor([[2.0], [2.0]])
    real = torch.tensor([[1.0], [1.0]])
    gp = torch.tensor(0.5)
    loss = get_crit_loss(fake, real, gp, c_lambda=10.0)
    # mean(fake) - mean(real) + c_lambda*gp = 2 - 1 + 5 = 6
    assert torch.isclose(loss, torch.tensor(6.0))


def test_get_gradient_runs_and_shapes():
    torch.manual_seed(0)
    crit = Critic(n_classes=3, hidden_dim=8)
    real = torch.randn(5, 1, requires_grad=False)
    fake = torch.randn(5, 1)
    one_hot = torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])]
    eps = torch.rand(5, 1, requires_grad=True)
    grad = get_gradient(crit, real, fake, eps, one_hot)
    assert grad.shape == (5, 1)


def test_minibatch_std_appends_feature():
    from src.gan.models import minibatch_std

    x = torch.randn(8, 4)
    out = minibatch_std(x)
    assert out.shape == (8, 5)  # uma feature extra
    # Entrada constante -> std da feature anexada ≈ 0 (detecta colapso)
    const = torch.ones(8, 4)
    assert minibatch_std(const)[:, -1].abs().max().item() < 1e-6
