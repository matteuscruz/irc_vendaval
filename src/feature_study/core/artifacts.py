"""Persistência do campeão do estudo como `.joblib` reaproveitável.

O estudo de features descarta os modelos ajustados: ele mede efeitos, não
entrega estimadores. Isso torna impossível gerar o mapa espacial em GRADE a
partir dele — `SpatialCorrector` (`src/inference/spatial_correction.py:150`)
aborta quando `fitted_models/` está vazio, e a única saída seria retreinar o
arm vencedor na pipeline de produção.

Este módulo grava o campeão de cada trimestre no MESMO schema que o
`cluster_lazy` já produz (`cluster_lazy.py:769`), para que a inferência em
grade funcione sem nenhum caso especial e sem reajustar nada.

Escopo deliberadamente estreito — rank-1 de cada trimestre, só o arm âncora,
só a seed base ⇒ 4 arquivos. Com ~680 colunas, um ExtraTrees/RandomForest de
100 árvores passa de 100 MB, então tudo sai com `compress=3`.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.preprocessing import RobustScaler

# Nome esperado pelo glob de `SpatialCorrector.prepare` (`best_model_c*.joblib`).
# Fugir dele faria o arquivo ser ignorado em silêncio, e o mapa em grade
# continuaria abortando por "nenhum modelo encontrado".
FILENAME = "best_model_c{cluster_id}_{season}.joblib"
COMPRESS = 3


def champion_artifact(*, model, model_name: str, features, x_train_raw,
                      cluster_id, season: str, r2: float | None = None,
                      study_tag: str = "", seed: int | None = None) -> dict:
    """Artefato no schema de `cluster_lazy`, com o scaler reconstruído.

    O `fit_arm` treina sobre o frame JÁ escalado por `preprocess_df`
    (RobustScaler ajustado só no treino), então o artefato precisa carregar um
    scaler equivalente — a inferência aplica `artifact["scaler"]` antes de
    `artifact["model"].predict`. Reajustá-lo aqui sobre o MESMO treino bruto
    reproduz exatamente a transformação usada no ajuste.

    `target_kind="absolute"`: o LazyPredict prevê m/s, não uma razão sobre o
    ERA5. Sem essa chave a inferência decidiria pelo nome do modelo e trataria
    errado um campeão que por acaso fosse `MLPRegressor`.
    """
    return {
        "model": model,
        "model_name": model_name,
        "scaler": RobustScaler().fit(x_train_raw),
        "features": list(features),
        "cluster_id": cluster_id,
        "season": season,
        "climatology": {"method": "harmonic", "n_harmonics": 3},
        "r2": None if r2 is None else float(r2),
        "target_kind": "absolute",
        # Procedência: sem isto, um joblib do ESTUDO fica indistinguível de um
        # da produção dentro de `fitted_models/`.
        "source": "feature_study",
        "study_tag": study_tag,
        "seed": seed,
    }


def dump_champion_pipeline(artifact: dict, out_dir, *, compress: int = COMPRESS) -> Path:
    """Grava em `<out_dir>/fitted_models/` e devolve o caminho."""
    import joblib

    destino = Path(out_dir) / "fitted_models"
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / FILENAME.format(cluster_id=artifact["cluster_id"], season=artifact["season"])
    joblib.dump(artifact, caminho, compress=compress)
    tamanho = caminho.stat().st_size / 1e6
    print(f"[artifacts] {caminho.name}: {artifact['model_name']}, "
          f"{len(artifact['features'])} features, {tamanho:.1f} MB", flush=True)
    return caminho
