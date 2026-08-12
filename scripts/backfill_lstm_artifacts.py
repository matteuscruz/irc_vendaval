"""Wrapper local do backfill de artefatos LSTM (predictions + architecture).

Uso
---
    uv run python scripts/backfill_lstm_artifacts.py <exp_dir>
    # ex.: uv run python scripts/backfill_lstm_artifacts.py \\
    #        artifacts/lstm_pytorch_modal/lstm_pytorch/lstm_v2

A lógica vive em src/pipelines/lstm_backfill.py (também usada pelo Modal).
"""
from __future__ import annotations

import sys

from src.pipelines.lstm_backfill import backfill

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python scripts/backfill_lstm_artifacts.py <exp_dir>")
        sys.exit(1)
    backfill(sys.argv[1])
