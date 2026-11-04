"""Modal (GPU): a LSTM nos mesmos desenhos do ML regional/global (clusters 1–4 e 12).

Pré-requisito local: `scripts/treinar_global.py --stage base` e `--stage janelas` (gera
`artifacts/feature_study/global/{base_global.parquet, janelas.npy}`).

    MODAL_PROFILE=<perfil> modal run src/modal/global_lstm.py --stage upload      # ~2 GB para o volume
    MODAL_PROFILE=<perfil> modal run src/modal/global_lstm.py --stage fit         # 49 unidades (12 desenhos × 4 trim. + GA)
    MODAL_PROFILE=<perfil> modal run src/modal/global_lstm.py --stage fit --chaves GA   # só o modelo anual
    MODAL_PROFILE=<perfil> modal run src/modal/global_lstm.py --stage download    # traz as previsões
    uv run python scripts/treinar_global.py --stage avaliar                       # métricas ML × LSTM

Cada unidade = (desenho, trimestre): treina a LSTM do estudo (`worker_lstm`, LSTM(96), Huber, early stopping) nas
linhas de treino dos clusters do desenho, ajusta o EQM na validação deles e prevê as linhas de teste e de 2025 dos
clusters avaliados. Idempotente: unidade já gravada é pulada (use `--force` para refazer).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

APP_NAME = "irc-vendaval-global-lstm"
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"
DATASET_VOLUME_NAME = "irc-vendaval-dataset"
REMOTE_DATASET_DIR = "/dataset"
# ERA5 bruto de cada região (local → volume de dados), para as janelas da grade ano a ano
RAW_REGIOES = {"rs_sc": ("dataset/raw/era5_rs_sc", "raw/era5_rs_sc"),
               "c12": ("dataset/raw/new_features/cluster12", "raw/new_features/cluster12"),
               "c5": ("dataset/raw/new_features/cluster5/vendaval_val12_global.nc",
                      "raw/new_features/cluster5/vendaval_val12_global.nc")}
REMOTE_APP_DIR = "/app"
REMOTE_ARTIFACTS_DIR = "/artifacts"
VOL_DIR = "feature_study/global"                 # relativo à raiz do volume
SEASONS = ("DJF", "MAM", "JJA", "SON")
_local_root = Path(__file__).parent.parent.parent
LOCAL_DIR = _local_root / "artifacts" / VOL_DIR

# mesma imagem da LSTM do estudo de features (TensorFlow 2.16.1 com GPU; NumPy < 2 nos dois pip_install)
lstm_image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.16.1-gpu")
    .apt_install("libhdf5-dev", "libnetcdf-dev", "gdal-bin", "libgdal-dev", "libproj-dev")
    .pip_install(
        "numpy>=1.24.0,<2", "pandas>=2.0.0,<2.3", "scikit-learn>=1.6.0", "matplotlib>=3.9.0",
        "xarray>=2024.1.0", "netCDF4>=1.6.0", "geopandas>=0.14.0", "pyshp>=2.3.0",
        "scipy>=1.10.0", "shapely>=2.0.0", "pyarrow>=14.0.0", "pyyaml>=6.0.3", "pydantic>=2.13.3",
    )
    .pip_install("numpy>=1.24.0,<2", "lazypredict[boost]>=0.3.0", "xgboost>=2.0.0", "lightgbm>=4.0.0")
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_file(str(_local_root / "main.py"), remote_path=f"{REMOTE_APP_DIR}/main.py", copy=True)
    .pip_install("numpy>=1.24.0,<2", "catboost>=1.2")
    .add_local_dir(str(_local_root / "src"), remote_path=f"{REMOTE_APP_DIR}/src", copy=True)
    .add_local_dir(str(_local_root / "scripts"), remote_path=f"{REMOTE_APP_DIR}/scripts", copy=True)
    .add_local_dir(str(_local_root / "config"), remote_path=f"{REMOTE_APP_DIR}/config", copy=True)
)

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)


@app.function(image=lstm_image, volumes={REMOTE_ARTIFACTS_DIR: artifact_volume}, gpu="T4", timeout=5400,
              memory=16384, cpu=4, max_containers=24, retries=1)
def fit_unit(chave: str, season: str, force: bool = False) -> dict:
    if REMOTE_APP_DIR not in sys.path:
        sys.path.insert(0, REMOTE_APP_DIR)
    import numpy as np
    import pandas as pd

    from src.feature_study.diagnostics import regional as rg
    from src.feature_study.diagnostics import regional_lstm as rl

    artifact_volume.reload()
    d = Path(REMOTE_ARTIFACTS_DIR) / VOL_DIR
    out = d / "lstm" / f"{chave}__{season}.parquet"
    if out.exists() and not force:
        return {"chave": chave, "season": season, "pulada": True}
    des = {x["chave"]: x for x in rg.designs()}[chave]
    base = pd.read_parquet(d / "base_global.parquet")
    X = np.load(d / "janelas.npy", mmap_mode="r")
    t0 = time.time()
    anual = season == rg.ANUAL                       # modelo único do ano todo, com o dia do ano como coluna
    extra = rl.doy_features(base) if anual else None
    fit = rl.fit_season(X, base, des["train"], None if anual else season, extra=extra)
    alvo = (base.cluster_id.isin(list(des["avalia"])) & base.janela_ok & base._split.isin(["test", "novo"]))
    if not anual:
        alvo &= base.season == season
    idx = np.flatnonzero(alvo.to_numpy())
    bruto, eqm = rl.predict_rows(fit, X, idx, extra=extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": base.row_id.to_numpy()[idx], "bruto": bruto, "eqm": eqm}).to_parquet(out, index=False)
    rl.save_fit(fit, d / "modelos" / f"{chave}__{season}")          # modelo salvo para reuso (mapas, produção)
    m = fit["metrics"]
    info = {"chave": chave, "season": season, "n_train": fit["n_train"], "n_val": fit["n_val"], "n_prev": len(idx),
            "rmse_val": float(m[m.split == "val"].RMSE.iloc[0]), "segundos": round(time.time() - t0, 1)}
    (out.with_suffix(".json")).write_text(pd.Series(info).to_json())
    artifact_volume.commit()
    print(f"[modal] ✓ {chave}/{season}: {info}", flush=True)
    return info


@app.function(image=lstm_image, volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
              gpu="T4", timeout=3600, memory=24576, cpu=4, max_containers=26, retries=1)
def grade_ano(ano: int, cols: list, celulas: dict, force: bool = False) -> dict:
    """Mapa da LSTM num ano: janelas de 24 h de todas as células (`celulas` = região → [(lat, lon, cluster)]),
    previstas pela LSTM global anual (GA) e pela local do cluster e trimestre (B<c>). Grava só as previsões
    (com o mapeamento de quantis) em `lstm_grade_anos/<ano>.parquet`; as janelas não são guardadas."""
    if REMOTE_APP_DIR not in sys.path:
        sys.path.insert(0, REMOTE_APP_DIR)
    import numpy as np
    import pandas as pd

    from src.feature_study.data import era5_features as ef
    from src.feature_study.diagnostics import regional_lstm as rl

    d = Path(REMOTE_ARTIFACTS_DIR) / VOL_DIR
    out = d / "lstm_grade_anos" / f"{ano}.parquet"
    if out.exists() and not force:
        return {"ano": ano, "pulado": True}
    t0 = time.time()
    metas, xs = [], []
    for reg, cel in celulas.items():
        cl_de = {(round(a, 3), round(b, 3)): int(c) for a, b, c in cel}
        meta, x = ef.peak_windows_year(Path(REMOTE_DATASET_DIR) / RAW_REGIOES[reg][1], [(a, b) for a, b, _ in cel],
                                       ano, cols, window=24)
        meta["cluster_id"] = [cl_de[(round(a, 3), round(b, 3))] for a, b in zip(meta.lat, meta.lon)]
        metas.append(meta); xs.append(x)
    meta = pd.concat(metas, ignore_index=True)
    X = np.concatenate(xs)
    del xs
    meta["season"] = meta.time.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                             6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    meta["gust10fg"] = X[:, -1, cols.index("gust10fg")]
    todos = np.arange(len(meta))
    _, meta["ls_anual"] = rl.predict_rows(rl.load_fit(d / "modelos" / "GA__ALL"), X, todos, extra=rl.doy_features(meta))
    meta["ls_local"] = np.nan
    for (c, s_), g in meta.groupby(["cluster_id", "season"]):
        _, e = rl.predict_rows(rl.load_fit(d / "modelos" / f"B{c}__{s_}"), X, g.index.to_numpy())
        meta.loc[g.index, "ls_local"] = e
    out.parent.mkdir(parents=True, exist_ok=True)
    meta[["lat", "lon", "time", "cluster_id", "gust10fg", "ls_local", "ls_anual"]].to_parquet(out, index=False)
    artifact_volume.commit()
    info = {"ano": ano, "linhas": len(meta), "segundos": round(time.time() - t0, 1)}
    print(f"[modal] ✓ grade {info}", flush=True)
    return info


@app.function(image=lstm_image, volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
              timeout=6 * 3600, memory=49152, cpu=8)
def etapa_remota(nome: str) -> str:
    """Roda no Modal uma etapa dos scripts locais, sem mudar os caminhos deles: /app/artifacts e /app/dataset
    apontam para os volumes. `nome`: adiciona-c5 | ml | avaliar | mapas | exportar."""
    import os

    if REMOTE_APP_DIR not in sys.path:
        sys.path.insert(0, REMOTE_APP_DIR)
    for link, alvo in (("artifacts", REMOTE_ARTIFACTS_DIR), ("dataset", REMOTE_DATASET_DIR)):
        if not os.path.exists(f"{REMOTE_APP_DIR}/{link}"):
            os.symlink(alvo, f"{REMOTE_APP_DIR}/{link}")
    artifact_volume.reload()
    t0 = time.time()
    from scripts import treinar_global as tg

    if nome == "adiciona-c5":
        tg.etapa_adiciona_c5(range(2000, 2026))
    elif nome == "ml":
        tg.etapa_ml()
    elif nome == "avaliar":
        tg.etapa_avaliar()
    elif nome == "mapas":
        from scripts import mapa_extrapolacao as me

        me.main()
    elif nome == "exportar":
        from scripts import exportar_predicoes_nc as ex

        ex.main()
    else:
        raise ValueError(nome)
    artifact_volume.commit()
    return f"{nome}: {time.time() - t0:.0f}s"


@app.function(image=lstm_image, volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
              timeout=3600, memory=4096, cpu=2)
def juntar_partes(destino: str, tamanho: int) -> str:
    """Remonta `<destino>.partes/*` em `<destino>` (envio em partes: o cliente do Modal esgota a memória local com
    arquivos grandes). `destino` é absoluto no container; confere o tamanho antes de apagar as partes."""
    import shutil

    d = Path(destino)
    partes = sorted(Path(destino + ".partes").iterdir())
    d.parent.mkdir(parents=True, exist_ok=True)
    with open(d, "wb") as out:
        for p in partes:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 8 * 1024 * 1024)
    if d.stat().st_size != tamanho:
        raise RuntimeError(f"{d}: {d.stat().st_size} bytes, esperados {tamanho}")
    shutil.rmtree(destino + ".partes")
    (artifact_volume if destino.startswith(REMOTE_ARTIFACTS_DIR) else dataset_volume).commit()
    return f"{d}: {tamanho} bytes de {len(partes)} partes"


def _confere_volume() -> None:
    """Falha cedo (antes de subir GPUs) se a base e as janelas não estiverem no volume, ou tiverem outro tamanho."""
    try:
        no_vol = {Path(e.path).name: e.size for e in artifact_volume.listdir(VOL_DIR)}
    except Exception:
        no_vol = {}
    for nome in ("base_global.parquet", "janelas.npy"):
        if nome not in no_vol:
            raise SystemExit(f"{VOL_DIR}/{nome} não está no volume: rode `--stage upload` (e não interrompa).")


@app.local_entrypoint()
def main(stage: str = "fit", chaves: str = "", force: bool = False, only_grade: bool = False, anos: str = "2000-2025",
         regioes: str = ""):
    """`upload` envia base e janelas; `fit` treina todas as unidades (ou só `--chaves B1,G`); `download` traz.
    `upload-raw` envia o ERA5 bruto das regiões; `grade-anos` prevê a grade ano a ano com as LSTMs salvas."""
    from src.feature_study.diagnostics import regional as rg

    todas = [x["chave"] for x in rg.designs()]
    sel = [c for c in chaves.split(",") if c] or todas
    if stage == "upload":
        extras = [n for n in ("grade_meta.parquet", "janelas_grade.npy") if (LOCAL_DIR / n).exists()]
        for nome in ("base_global.parquet", "janelas.npy"):
            if not (LOCAL_DIR / nome).exists():
                raise SystemExit(f"{LOCAL_DIR / nome} não existe: rode scripts/treinar_global.py --stage base/janelas")
        so_grade = only_grade
        nomes = extras if so_grade else ["base_global.parquet", "janelas.npy", *extras]
        tam = sum((LOCAL_DIR / n).stat().st_size for n in nomes) / 1e9
        print(f"enviando {tam:.2f} GB para {ARTIFACT_VOLUME_NAME}:{VOL_DIR} — leva alguns minutos, sem barra de "
              "progresso. NÃO interrompa (Ctrl+C cancela o envio inteiro).", flush=True)
        with artifact_volume.batch_upload(force=True) as b:
            for nome in nomes:
                b.put_file(str(LOCAL_DIR / nome), f"{VOL_DIR}/{nome}")
        _confere_volume()
        print("upload concluído e conferido no volume.")
    elif stage == "fit":
        _confere_volume()
        anuais = {x["chave"] for x in rg.designs() if x.get("anual")}
        args = [(c, s, force) for c in sel for s in ((rg.ANUAL,) if c in anuais else SEASONS)]
        print(f"{len(args)} unidades: {sel}")
        for r in fit_unit.starmap(args, return_exceptions=True):
            print(r)
    elif stage == "etapa":
        print(etapa_remota.remote(chaves))                # ex.: --stage etapa --chaves adiciona-c5
    elif stage == "juntar":
        destino, tam = chaves.rsplit(":", 1)              # ex.: --chaves /dataset/raw/x.nc:12345
        print(juntar_partes.remote(destino, int(tam)))
    elif stage == "upload-apoio":
        # o que as etapas remotas leem e só existe localmente
        A = _local_root / "artifacts/feature_study"
        with artifact_volume.batch_upload(force=True) as b:
            b.put_directory(str(A / "cluster3_groups/data"), "feature_study/cluster3_groups/data")
            for n in ("estacoes.csv",):
                b.put_file(str(LOCAL_DIR / n), f"{VOL_DIR}/{n}")
        print("apoio no volume")
    elif stage == "upload-raw":
        print("enviando o ERA5 bruto — leva vários minutos. NÃO interrompa.", flush=True)
        with dataset_volume.batch_upload(force=True) as b:
            for reg in ([r for r in regioes.split(",") if r] or RAW_REGIOES):
                loc, rem = RAW_REGIOES[reg]
                if (_local_root / loc).is_file():
                    b.put_file(str(_local_root / loc), rem)
                else:
                    b.put_directory(str(_local_root / loc), rem)
        print("ERA5 bruto no volume", DATASET_VOLUME_NAME)
    elif stage == "grade-anos":
        import pandas as pd

        from scripts.treinar_global import colunas
        from src.feature_study.data import era5_features as ef

        meta = pd.read_parquet(LOCAL_DIR / "grade_meta.parquet").drop_duplicates(["lat", "lon"])
        celulas = {"rs_sc": [(float(a), float(b), int(c)) for a, b, c in
                             meta[~meta.cluster_id.isin([5, 12])][["lat", "lon", "cluster_id"]].itertuples(index=False)],
                   "c12": [(float(a), float(b), int(c)) for a, b, c in
                           meta[meta.cluster_id == 12][["lat", "lon", "cluster_id"]].itertuples(index=False)],
                   "c5": [(a, b, 5) for a, b in ef.points_of_cluster(_local_root / RAW_REGIOES["c5"][0], 5)]}
        i, f = (int(x) for x in anos.split("-"))
        cols = colunas()
        print(f"{f - i + 1} anos, {sum(len(v) for v in celulas.values())} células, {len(cols)} colunas")
        for r in grade_ano.starmap([(a, cols, celulas, force) for a in range(i, f + 1)], return_exceptions=True):
            print(r)
    elif stage == "download":
        try:
            artifact_volume.listdir(f"{VOL_DIR}/lstm")
        except Exception:
            raise SystemExit(f"{VOL_DIR}/lstm não existe no volume: o `fit` ainda não gravou nenhuma unidade")
        dest = LOCAL_DIR / "lstm"
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for sub in ("lstm",):
            try:
                ents = artifact_volume.listdir(f"{VOL_DIR}/{sub}")
            except Exception:
                continue
            (LOCAL_DIR / sub).mkdir(parents=True, exist_ok=True)
            for e in ents:
                with open(LOCAL_DIR / sub / Path(e.path).name, "wb") as f:
                    for parte in artifact_volume.read_file(e.path):
                        f.write(parte)
                n += 1
        print(f"{n} arquivos em {dest}")
    else:
        raise SystemExit("stage: upload | fit | download | upload-raw | grade-anos")
