"""Teste estreito pra `load_mlp_training_frame` (T1.7 do plano de
decomposição Kedro) — extraída de `cluster_mlp.run()` sem mudança de
lógica. Split deliberadamente raso: só a etapa de carregamento, não o loop
por cluster (ver T3.1 do plano — `cluster_mlp.py` não tem
`_process_one_cluster` equivalente pra decompor mais fundo).
"""
from __future__ import annotations

from pathlib import Path


def test_load_mlp_training_frame_builds_frame_and_stations_meta(
    synthetic_raw_dir: Path, synthetic_shp_dir: Path,
):
    from src.pipelines.cluster_mlp import load_mlp_training_frame

    df, stations_meta = load_mlp_training_frame(
        str(synthetic_raw_dir), str(synthetic_shp_dir), feature_groups="original",
    )

    assert not df.empty
    assert "cluster_id" in df.columns
    assert not stations_meta.empty
    assert {"estacao", "cluster_id", "latitude", "longitude"}.issubset(stations_meta.columns)


def test_run_accepts_precomputed_frame_without_reloading(
    synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path, monkeypatch,
):
    """Prova que `precomputed_frame` (Kedro) realmente é usado — não só
    aceito e ignorado — monkeypatchando `load_mlp_training_frame` pra
    explodir se `run()` tentar recarregar."""
    from src.pipelines import cluster_mlp

    df, stations_meta = cluster_mlp.load_mlp_training_frame(
        str(synthetic_raw_dir), str(synthetic_shp_dir), feature_groups="original",
    )

    def _boom(*a, **kw):
        raise AssertionError("run() recarregou os dados apesar de precomputed_frame ter sido passado")

    monkeypatch.setattr(cluster_mlp, "load_mlp_training_frame", _boom)

    output_dir = tmp_path / "mlp_precomputed"
    cluster_mlp.run(
        raw_dir=str(synthetic_raw_dir), shp_dir=str(synthetic_shp_dir),
        output_dir=str(output_dir), exp_name="smoke", feature_groups="original",
        precomputed_frame=(df, stations_meta),
    )

    results_path = output_dir / "smoke" / "results.csv"
    assert results_path.exists()
