"""Testes estreitos pro fix de persistência/uso de modelos lazy por
trimestre (Fase 2 do plano de correção ERA5 2000-2024):

(a) `cluster_lazy.run(stratify_seasons=True)` grava tanto o modelo pooled
    (season=None) quanto os campeões por trimestre — antes só o pooled era
    persistido, mesmo quando a tabela de vencedores elegia um campeão
    específico de trimestre.
(b) `SpatialCorrector.predict_stations()` escolhe o artifact do trimestre
    certo quando ambos (pooled e por-trimestre) existem pro mesmo cluster,
    em vez de aplicar o último carregado a todas as linhas (bug de colisão
    de chave que existia antes do fix).

Rodar isolado (não a suíte inteira): pytest test/test_lazy_season_models.py -q
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def test_cluster_lazy_persists_pooled_and_per_season_models(
    synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path,
):
    from src.pipelines import cluster_lazy

    output_dir = tmp_path / "lazy_clusters"
    cluster_lazy.run(
        raw_dir=str(synthetic_raw_dir),
        shp_dir=str(synthetic_shp_dir),
        output_dir=str(output_dir),
        exp_name="smoke_seasons",
        feature_groups="original",
        stratify_seasons=True,
    )

    models_dir = output_dir / "smoke_seasons" / "fitted_models"
    saved = sorted(p.name for p in models_dir.glob("best_model_c*.joblib"))
    assert saved, f"Nenhum modelo salvo em {models_dir}"

    pooled = [n for n in saved if "_" not in n.removeprefix("best_model_c")]
    per_season = [n for n in saved if n not in pooled]
    assert pooled, f"Nenhum modelo pooled (season=None) salvo: {saved}"
    assert per_season, (
        f"Nenhum modelo por trimestre salvo (gap que este fix resolve): {saved}"
    )

    # Artifact por-trimestre carrega a chave "season" explícita, não-None.
    artifact = joblib.load(models_dir / per_season[0])
    assert artifact["season"] in {"DJF", "MAM", "JJA", "SON"}

    # Artifact pooled carrega season=None explícito.
    pooled_artifact = joblib.load(models_dir / pooled[0])
    assert pooled_artifact["season"] is None


class _ConstantModel:
    """Modelo fake: sempre prediz o mesmo valor, pra rastrear qual artifact
    (cluster, season) foi de fato aplicado a cada linha."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


def _fake_artifact(cid: int, season: str | None, value: float) -> dict:
    x_fit = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = RobustScaler().fit(x_fit)
    return {
        "model": _ConstantModel(value),
        "model_name": "Constant",
        "scaler": scaler,
        "features": ["x"],
        "cluster_id": cid,
        "season": season,
        "target_kind": "absolute",
    }


def test_spatial_corrector_prefers_season_specific_model():
    from src.inference.spatial_correction import SpatialCorrector

    corr = SpatialCorrector.__new__(SpatialCorrector)  # bypass __init__ (sem raw_dir real)
    # Valores dentro do range físico [0, 80] a que predict_stations() clipa
    # (senão o teste quebraria pelo clip, não pela lógica de seleção).
    corr.cluster_models = {
        (1, None): _fake_artifact(1, None, value=1.0),
        (1, "DJF"): _fake_artifact(1, "DJF", value=50.0),
    }

    # 4 linhas do cluster 1: uma em cada trimestre (jan=DJF, abr=MAM,
    # jul=JJA, out=SON) — só DJF tem modelo específico.
    df_all = pd.DataFrame({
        "time": pd.to_datetime(["2021-01-15", "2021-04-15", "2021-07-15", "2021-10-15"]),
        "estacao": ["A1", "A1", "A1", "A1"],
        "latitude": [-20.0] * 4,
        "longitude": [-45.0] * 4,
        "cluster_id": [1, 1, 1, 1],
        "x": [1.0, 1.0, 1.0, 1.0],
        "daily_wind_gust_max": [15.0, 15.0, 15.0, 15.0],
        "wind_mag_max": [10.0, 10.0, 10.0, 10.0],
    })
    corr.df_all = df_all

    out = corr.predict_stations(("2021-01-01", "2021-12-31"))
    out = out.merge(df_all[["time"]].assign(_month=df_all["time"].dt.month), on="time")

    djf_row = out[out["_month"] == 1]
    other_rows = out[out["_month"] != 1]

    assert (djf_row["rajada_corrigida"] == 50.0).all(), (
        "Linha de DJF deveria usar o modelo específico do trimestre (valor 50), "
        f"mas usou: {djf_row['rajada_corrigida'].tolist()}"
    )
    assert (other_rows["rajada_corrigida"] == 1.0).all(), (
        "Linhas fora de DJF deveriam cair no modelo pooled (valor 1), "
        f"mas usaram: {other_rows['rajada_corrigida'].tolist()}"
    )
