"""Alvo da LSTM: rajada máxima diária em m/s.

A LSTM dual-head anterior previa a RAZÃO rajada/ERA5 e reconstruía o valor
absoluto multiplicando pelo proxy ERA5 do dia-alvo — reconstrução repetida em
6+ lugares (trainer, métricas, plots, injeção sintética, inferência). O alvo
agora é a própria rajada em m/s (escalonada só para o treino), e a única
inversão válida é esta.
"""
from __future__ import annotations

import numpy as np

TARGET_CLIP = (0.0, 80.0)  # faixa física (m/s)


def inverse_target(scaler_y, y_scaled, clip: bool = False) -> np.ndarray:
    """Desfaz o escalonamento do alvo e devolve vetor 1-D em m/s.

    `clip=True` para PREDIÇÕES (limita à faixa física); o y observado não deve
    ser clipado, senão as métricas mascaram erro real de medição.
    """
    y = scaler_y.inverse_transform(
        np.asarray(y_scaled, dtype=float).reshape(-1, 1)
    ).ravel()
    return np.clip(y, *TARGET_CLIP) if clip else y
