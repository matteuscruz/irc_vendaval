"""LSTM com gate suave, attention pooling e branch estático (PyTorch).

Arquitetura:
    seq dinâmico (B, L, n_dyn) ─► LSTM ─► attention pool ─┐
                                                          ├─► shared ─┬─► head_mean  (Huber)
    estático (B, n_stat) ─► MLP ──────────────────────────┘           ├─► head_extreme (Pinball)
                                                                       └─► head_gate (σ)

A predição final é uma combinação suave (gate aprendido) das duas cabeças:
    blended = (1 - gate) * mean + gate * extreme
eliminando a descontinuidade do roteamento binário. O branch estático (lat/lon + percentis
da estação) condiciona a sequência e é reinjetado após o pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttnPool(nn.Module):
    """Atenção aditiva (Bahdanau) sobre os timesteps da sequência."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.w = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        # out: (B, L, H) → contexto ponderado (B, H)
        scores = self.v(torch.tanh(self.w(out)))   # (B, L, 1)
        alpha = torch.softmax(scores, dim=1)
        return (alpha * out).sum(dim=1)


class GatedAttnLSTM(nn.Module):
    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        hidden: int = 64,
        static_hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.static_mlp = nn.Sequential(
            nn.Linear(n_static, static_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            n_dynamic + static_hidden,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = _AttnPool(hidden)
        self.bn = nn.BatchNorm1d(hidden + static_hidden)
        self.shared = nn.Sequential(
            nn.Linear(hidden + static_hidden, 16),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.head_mean = nn.Linear(16, 1)      # regime normal
        self.head_extreme = nn.Linear(16, 1)   # cauda
        self.head_gate = nn.Linear(16, 1)      # peso suave da combinação

    def forward(
        self, seq: torch.Tensor, static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # seq: (B, L, n_dyn) | static: (B, n_stat)
        s = self.static_mlp(static)                     # (B, static_hidden)
        s_tiled = s.unsqueeze(1).expand(-1, seq.size(1), -1)
        out, _ = self.lstm(torch.cat([seq, s_tiled], dim=-1))
        ctx = self.attn(out)                            # (B, hidden)
        z = self.shared(self.bn(torch.cat([ctx, s], dim=-1)))
        mean = self.head_mean(z).squeeze(-1)
        ext = self.head_extreme(z).squeeze(-1)
        gate = torch.sigmoid(self.head_gate(z)).squeeze(-1)
        return mean, ext, gate


def blended_prediction(
    mean: torch.Tensor, ext: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Combinação suave das cabeças — usada no treino e na inferência."""
    return (1.0 - gate) * mean + gate * ext


def gated_loss(
    mean: torch.Tensor,
    ext: torch.Tensor,
    gate: torch.Tensor,
    target: torch.Tensor,
    tau: float,
    w_normal: float,
    w_extreme: float,
    huber_delta: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Huber (head_mean) + Pinball (head_extreme) + Huber (blended).

    target é a razão INMET/ERA5. weights são pesos por amostra (extreme weighting).
    l_mean/l_ext especializam cada cabeça; l_blend calibra a saída final.
    """
    blended = blended_prediction(mean, ext, gate)
    l_mean = F.huber_loss(mean, target, delta=huber_delta, reduction="none")
    err = target - ext
    l_ext = torch.where(err >= 0, tau * err, (tau - 1.0) * err)
    l_blend = F.huber_loss(blended, target, delta=huber_delta, reduction="none")
    per = w_normal * l_mean + w_extreme * l_ext + l_blend
    return (per * weights).sum() / weights.sum().clamp(min=1e-8)
