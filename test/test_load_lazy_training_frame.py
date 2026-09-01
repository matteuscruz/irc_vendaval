"""Teste estreito pra `load_lazy_training_frame` (T1.6 do plano de
decomposição Kedro) — extraída de `cluster_lazy.run()` sem mudança de
lógica. Confirma que produz um DataFrame não-vazio com a coluna
`cluster_id` esperada, usando os fixtures sintéticos já existentes.
"""
from __future__ import annotations

from pathlib import Path


def test_load_lazy_training_frame_builds_flat_dataframe(
    synthetic_raw_dir: Path, synthetic_shp_dir: Path,
):
    from src.pipelines.cluster_lazy import load_lazy_training_frame

    df = load_lazy_training_frame(
        str(synthetic_raw_dir), str(synthetic_shp_dir), feature_groups="original",
    )

    assert not df.empty
    assert "cluster_id" in df.columns
    assert df["cluster_id"].nunique() >= 1
