"""Gera `resultados_preliminares/`: os dados que contam COMO a seleção de variáveis do
Cluster 3 foi feita (pôster). Nada aqui altera `src/` nem os resultados do estudo.

Etapas (`--etapas`, padrão todas):
  permutacao  importância por permutação por VARIÁVEL (o instrumento que escolheu as 12
              variáveis de `config/selected_features_val12.json`), com as predições de cada
              modelo para cada variável embaralhada → 01_*, 02_*, 03_*
  grupos      cada grupo (1..4) SOZINHO, triagem com todos os regressores do estudo, top-5 e
              campeão por trimestre na validação, top-5 refeito nas 5 seeds → 08_*
  compilar    junta o que o estudo já gravou (efeitos dos grupos, predições e métricas dos
              arms base / sel__val12 / full) → 04_*, 05_*, 06_*

Reproduz exatamente a seção 9 de `scripts/build_notebook_group_importance.py` (LightGBM,
seed 42, permutação com `np.random.default_rng(42)`, 5 repetições, piso = `ctrl_noise`)
e confere que a seleção refeita é a do JSON congelado.

    uv run python scripts/gerar_resultados_preliminares.py
    uv run python scripts/gerar_resultados_preliminares.py --etapas grupos
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.feature_study.core.analysis import EQUIV_TOL_P90, EQUIV_TOL_R2  # noqa: E402
from src.feature_study.core.config import CHAMPION_SLACK, CLIP_RANGE, MODEL_SEEDS  # noqa: E402
from src.feature_study.core.worker import select_regressors  # noqa: E402
from src.feature_study.selection.selected import columns_of, load_selection, variable_of  # noqa: E402
from src.pipelines.common import compute_metrics, preprocess_df  # noqa: E402

warnings.filterwarnings("ignore")

ESTUDO = RAIZ / "resultados_lfs/feature_study/cluster3_groups"          # dados da seleção
ESTUDO_M = RAIZ / "resultados_lfs/feature_study/cluster3_groups_modal"  # estudo de referência (5 seeds)
SAIDA = RAIZ / "resultados_preliminares"
SEASONS = ("DJF", "MAM", "JJA", "SON")
ALVO, REF, RUIDO = "daily_wind_gust_max", "era5_gust_max", "ctrl_noise"
N_REP, SEED = 5, 42
ROTULO = {"grupo1": "grupo 1", "grupo2": "grupo 2", "grupo3": "grupo 3", "grupo4": "grupo 4"}
DESCR_GRUPO = {"grupo1": "superfície/camada limite (vento, rajada, BLH, T, Td, p, hora/dia)",
               "grupo2": "convecção (CAPE, MCPR, cisalhamento, lapse rates)",
               "grupo3": "sinótico (MSLP e gradiente, w850/w500, z500, vorticidade, omega)",
               "grupo4": "relevo e máscara terra-mar (estáticas por estação)"}
PARQ = dict(index=False, compression="zstd")


# ── dados ──────────────────────────────────────────────────────────────────────

def carregar():
    meta = json.loads((ESTUDO / "data/meta.json").read_text())
    arms = {a["name"]: a for a in json.loads((ESTUDO / "data/arms.json").read_text())}
    full = arms["full"]["features"]
    amostra = pd.read_parquet(ESTUDO / "data/sample_full.parquet")
    teste = pd.read_parquet(ESTUDO / "data/test.parquet").assign(_split="test")
    return meta, full, amostra[amostra._split == "train"], amostra[amostra._split == "val"], teste


def blocos_por_variavel(full, grupo_de):
    """Mesma redução coluna → variável do notebook (seção 1) e do JSON congelado."""
    blocos, grupo_var = {}, {}
    for c in full:
        g = grupo_de.get(c)
        if g is None:
            continue
        v = variable_of(c)
        blocos.setdefault(v, []).append(c)
        grupo_var[v] = g
    return blocos, grupo_var


def linhas(df):
    cols = ["row_id", "estacao", "time", "season", "year", "month", "_split",
            "latitude", "longitude", ALVO, REF]
    return df[cols].rename(columns={"_split": "split", ALVO: "y_true", REF: "era5_gust_max"})


# ── modelos ────────────────────────────────────────────────────────────────────

def ajustar_lazy(x_tr, x_va, y_tr, y_va, nomes, seed):
    """Mesmo LazyRegressor do estágio `fit` (`worker.fit_arm`) → {nome: pipeline}."""
    from lazypredict.Supervised import LazyRegressor

    reg = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True,
                        random_state=seed, regressors=select_regressors(nomes))
    reg.fit(x_tr, x_va, y_tr, y_va)
    return reg.provide_models(x_tr, x_va, y_tr, y_va)


def _pred(modelo, x, clip=True):
    p = np.asarray(modelo.predict(x), float).ravel()
    return np.clip(p, *CLIP_RANGE) if clip else p


def metricas(y, p):
    if not np.all(np.isfinite(p)):
        return {k: np.nan for k in ("R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90")}
    return compute_metrics(np.asarray(y, float), np.asarray(p, float))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def rmse_p90(y, p):
    y, p = np.asarray(y), np.asarray(p)
    m = y >= np.percentile(y, 90)
    return rmse(y[m], p[m])


# ── 1. permutação por variável ─────────────────────────────────────────────────

def permutar(predizer, x, y, blocos, seed=SEED):
    """`permutacao_por_bloco` (diagnostics/group_importance.py) devolvendo também a
    predição de cada (bloco, repetição). Mesma sequência de sorteios do notebook."""
    rng = np.random.default_rng(seed)
    p0 = predizer(x)
    r0, r0_90 = rmse(y, p0), rmse_p90(y, p0)
    resumo, preds = [], {}
    for nome, cols in blocos.items():
        for r in range(N_REP):
            emb = x.copy()
            ordem = rng.permutation(len(x))
            emb[cols] = x[cols].to_numpy()[ordem]
            p = predizer(emb)
            resumo.append({"variavel": nome, "repeticao": r, "n_colunas": len(cols),
                           "rmse_sem_permutar": r0, "rmse_permutado": rmse(y, p),
                           "d_rmse": rmse(y, p) - r0, "d_rmse_p90": rmse_p90(y, p) - r0_90})
            preds[(nome, r)] = p.astype("float32")
    return p0.astype("float32"), pd.DataFrame(resumo), preds


def etapa_permutacao(dados, sel):
    from lightgbm import LGBMRegressor

    meta, full, treino, val, teste = dados
    blocos, grupo_var = blocos_por_variavel(full, meta["group_of"])
    blocos_r = {**blocos, RUIDO: [RUIDO]}
    cols = full + [RUIDO]
    top5 = json.loads((ESTUDO_M / "summary/top_models.json").read_text())["by_season"]
    modelos_todos = sorted({"LGBMRegressor (seleção)"} | {m for ms in top5.values() for m in ms})
    print(f"[permutação] {len(blocos)} variáveis + ruído | modelos: {modelos_todos}", flush=True)

    resumo, preds_por_modelo = [], {m: [] for m in modelos_todos}
    for s in SEASONS:
        tr = treino[treino.season == s]
        avaliar = {"val": val[val.season == s], "test": teste[teste.season == s]}
        y_tr = tr[ALVO].to_numpy(float)

        # (a) o instrumento da seleção: LightGBM padrão, sem truncamento, como no notebook
        lgbm = LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=4).fit(tr[cols], y_tr)
        preditores = {"LGBMRegressor (seleção)": lambda x, m=lgbm: np.asarray(m.predict(x), float)}

        # (b) o top-5 do estudo de referência naquele trimestre (pipelines do LazyPredict,
        #     RobustScaler ajustado no treino, saída truncada em [0, 80] como no estudo)
        t0 = time.time()
        va_s = avaliar["val"]
        x_tr, x_va = preprocess_df(tr[cols], va_s[cols])
        ajustados = ajustar_lazy(x_tr, x_va, y_tr, va_s[ALVO].to_numpy(float), top5[s], SEED)
        from sklearn.preprocessing import RobustScaler
        esc = RobustScaler().fit(tr[cols])
        for nome, pipe in ajustados.items():
            preditores[nome] = (lambda x, p=pipe: _pred(p, pd.DataFrame(esc.transform(x), columns=cols)))
        print(f"  {s}: top-5 ajustado em {time.time() - t0:.0f}s", flush=True)

        for nome, predizer in preditores.items():
            for split, df in avaliar.items():
                t0 = time.time()
                x, y = df[cols].reset_index(drop=True), df[ALVO].to_numpy(float)
                p0, r, pr = permutar(predizer, x, y, blocos_r)
                r = r.assign(modelo=nome, season=s, split=split)
                resumo.append(r)
                base = pd.DataFrame({"row_id": df.row_id.to_numpy(), "y_true": y.astype("float32"),
                                     "y_pred_sem_permutar": p0})
                partes = [base.assign(variavel=v, repeticao=np.int8(rep), y_pred=p)
                          for (v, rep), p in pr.items()]
                preds_por_modelo[nome].append(pd.concat(partes, ignore_index=True)
                                              .assign(season=s, split=split))
                print(f"    {s} {split:<4} {nome:<32} {time.time() - t0:5.0f}s", flush=True)

    resumo = pd.concat(resumo, ignore_index=True)
    resumo["grupo"] = resumo.variavel.map(grupo_var).fillna("controle (ruído)")
    resumo["na_selecao_val12"] = resumo.variavel.isin(sel["variables"])
    resumo = resumo[["modelo", "season", "split", "variavel", "grupo", "n_colunas", "repeticao",
                     "rmse_sem_permutar", "rmse_permutado", "d_rmse", "d_rmse_p90", "na_selecao_val12"]]
    resumo.to_parquet(SAIDA / "02_permutacao_importancia.parquet", **PARQ)

    for nome, partes in preds_por_modelo.items():
        df = pd.concat(partes, ignore_index=True)
        df["grupo"] = df.variavel.map(grupo_var).fillna("controle (ruído)")
        for c in ("season", "split", "variavel", "grupo"):
            df[c] = df[c].astype("category")
        df = df[["season", "split", "variavel", "grupo", "repeticao",
                 "row_id", "y_true", "y_pred_sem_permutar", "y_pred"]]
        arq = nome.replace(" (seleção)", "_selecao")
        df.to_parquet(SAIDA / f"01_permutacao_predicoes_{arq}.parquet", **PARQ)

    linhas(pd.concat([val, teste])).to_parquet(SAIDA / "01_linhas_val_teste.parquet", **PARQ)
    decisao(resumo, sel, grupo_var)


def decisao(resumo, sel, grupo_var):
    """Regra congelada no JSON: d_rmse médio > piso do ruído (maior entre trimestres) e
    positivo em >= 3 de 4 trimestres. Aplicada por modelo e split; a oficial é LightGBM/val."""
    linhas_dec = []
    for (modelo, split), d in resumo.groupby(["modelo", "split"]):
        piso = d[d.variavel == RUIDO].groupby("season").d_rmse.mean().max()
        pv = d[d.variavel != RUIDO].groupby(["variavel", "season"]).d_rmse.mean().unstack()
        pv = pv.reindex(columns=[s for s in SEASONS if s in pv.columns])
        n_pos = (pv > 0).sum(axis=1)
        media = pv.mean(axis=1)
        dec = np.select([(media > piso) & (n_pos >= 3), media > piso], ["manter", "inconclusiva"], "descartar")
        out = pd.DataFrame({"modelo": modelo, "split": split, "variavel": pv.index,
                            "grupo": pv.index.map(grupo_var), "d_rmse_medio": media.values,
                            "trimestres_positivos": n_pos.values, "piso_ruido": piso, "decisao": dec,
                            **{f"d_rmse_{s}": pv[s].values for s in pv.columns}})
        linhas_dec.append(out)
    dec = pd.concat(linhas_dec, ignore_index=True)
    dec["na_selecao_val12"] = dec.variavel.isin(sel["variables"])
    manter = dec[dec.decisao == "manter"]
    concord = manter[manter.split == "val"].groupby("variavel").modelo.nunique()
    dec["n_modelos_que_mantem_val"] = dec.variavel.map(concord).fillna(0).astype(int)
    dec = dec.sort_values(["modelo", "split", "d_rmse_medio"], ascending=[True, True, False])
    dec.to_parquet(SAIDA / "03_decisao_selecao.parquet", **PARQ)

    oficial = dec[(dec.modelo == "LGBMRegressor (seleção)") & (dec.split == "val")]
    refeita = set(oficial[oficial.decisao == "manter"].variavel)
    piso = oficial.piso_ruido.iloc[0]
    print(f"[decisão] piso do ruído na validação: {piso:.5f} (JSON: {sel['noise_floor_val']})")
    if refeita != set(sel["variables"]):
        print(f"[AVISO] seleção refeita DIFERE do JSON: só aqui {sorted(refeita - set(sel['variables']))}, "
              f"só no JSON {sorted(set(sel['variables']) - refeita)}")
    else:
        print(f"[decisão] seleção refeita == JSON congelado ({len(refeita)} variáveis)")


# ── 8. cada grupo sozinho ──────────────────────────────────────────────────────

def ranking_val(met, k=5, slack=CHAMPION_SLACK):
    """Regra do `screen` (`analysis.top_models_by_season`): entre os modelos a até `slack`
    do melhor R² de validação, menor RMSE_P90; completa por R²; equivalentes contam uma vez."""
    g = met[(met.split == "val") & met.R2.notna() & met.RMSE_P90.notna()]
    eleg = g[g.R2 >= g.R2.max() - slack].sort_values("RMSE_P90")
    cand = list(eleg.model) + list(g[~g.model.isin(eleg.model)].sort_values("R2", ascending=False).model)
    pm = g.set_index("model")
    escolhidos, omit = [], {}
    for m in cand:
        gem = next((e for e in escolhidos if abs(pm.R2[e] - pm.R2[m]) <= EQUIV_TOL_R2
                    and abs(pm.RMSE_P90[e] - pm.RMSE_P90[m]) <= EQUIV_TOL_P90), None)
        if gem:
            omit.setdefault(gem, []).append(m)
        else:
            escolhidos.append(m)
    return escolhidos, omit, k


def bootstrap_blocos(df, col_pred, n=1000, seed=42):
    """IC 95 % do RMSE de teste com bootstrap em blocos (ano, mês), como no estudo."""
    bloco = (df.year * 100 + df.month).to_numpy()
    ub, inv = np.unique(bloco, return_inverse=True)
    e2 = (df[col_pred].to_numpy() - df.y_true.to_numpy()) ** 2
    soma = np.bincount(inv, weights=e2)
    cont = np.bincount(inv)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(ub), np.ones(len(ub)) / len(ub), size=n)
    r = np.sqrt((w @ soma) / (w @ cont))
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def etapa_grupos(dados):
    meta, full, treino, val, teste = dados
    arms = {f"so_{g}": [c for c in full if meta["group_of"].get(c) == g] for g in ROTULO}
    arms["so_ruido"] = [RUIDO]
    todos = [c.__name__ for c in select_regressors("all")]
    print(f"[grupos] {len(todos)} regressores | colunas por arm: "
          f"{ {a: len(c) for a, c in arms.items()} }", flush=True)

    mets, preds = [], []
    for arm, cols in arms.items():
        for s in SEASONS:
            t0 = time.time()
            tr, va, te = (d[d.season == s] for d in (treino, val, teste))
            y_tr, y_va, y_te = (d[ALVO].to_numpy(float) for d in (tr, va, te))
            x_tr, x_va = preprocess_df(tr[cols], va[cols])
            _, x_te = preprocess_df(tr[cols], te[cols])
            ajust = ajustar_lazy(x_tr, x_va, y_tr, y_va, todos, SEED)
            for nome, pipe in ajust.items():
                for split, x, y, d in (("val", x_va, y_va, va), ("test", x_te, y_te, te)):
                    try:
                        p = _pred(pipe, x)
                    except Exception as exc:  # um modelo quebrado não derruba o arm
                        print(f"   [AVISO] {nome} falhou ({type(exc).__name__})")
                        continue
                    mets.append({"arm": arm, "season": s, "model": nome, "model_seed": SEED,
                                 "split": split, "n": len(y), **metricas(y, p)})
                    preds.append(pd.DataFrame({"arm": arm, "season": s, "model": nome, "model_seed": SEED,
                                               "split": split, "row_id": d.row_id.to_numpy(),
                                               "y_pred": p.astype("float32")}))
            print(f"  {arm:<12} {s}: {len(ajust)} modelos em {time.time() - t0:.0f}s", flush=True)

    met = pd.DataFrame(mets)
    # top-5 e campeão por arm × trimestre, escolhidos na VALIDAÇÃO
    met["rank_val"] = np.nan
    met["equivalentes_omitidos"] = ""
    top = {}
    for (arm, s), g in met.groupby(["arm", "season"]):
        escolhidos, omit, k = ranking_val(g)
        top[(arm, s)] = escolhidos[:k]
        for r, m in enumerate(escolhidos[:k], 1):
            idx = (met.arm == arm) & (met.season == s) & (met.model == m)
            met.loc[idx, "rank_val"] = r
            met.loc[idx, "equivalentes_omitidos"] = ", ".join(omit.get(m, []))
    met["top5"] = met.rank_val.notna()
    met["campeao"] = met.rank_val == 1

    # top-5 nas outras seeds (só modelos estocásticos mudam)
    extra_m, extra_p = [], []
    for (arm, s), modelos in top.items():
        cols = arms[arm]
        tr, va, te = (d[d.season == s] for d in (treino, val, teste))
        y_tr, y_va = tr[ALVO].to_numpy(float), va[ALVO].to_numpy(float)
        x_tr, x_va = preprocess_df(tr[cols], va[cols])
        _, x_te = preprocess_df(tr[cols], te[cols])
        for seed in [x for x in MODEL_SEEDS if x != SEED]:
            ajust = ajustar_lazy(x_tr, x_va, y_tr, y_va, modelos, seed)
            for nome, pipe in ajust.items():
                for split, x, d in (("val", x_va, va), ("test", x_te, te)):
                    p = _pred(pipe, x)
                    extra_m.append({"arm": arm, "season": s, "model": nome, "model_seed": seed,
                                    "split": split, "n": len(d), **metricas(d[ALVO].to_numpy(float), p)})
                    extra_p.append(pd.DataFrame({"arm": arm, "season": s, "model": nome, "model_seed": seed,
                                                 "split": split, "row_id": d.row_id.to_numpy(),
                                                 "y_pred": p.astype("float32")}))
        print(f"  seeds extras {arm:<12} {s} ok", flush=True)
    if extra_m:
        em = pd.DataFrame(extra_m)
        chave = met[met.model_seed == SEED][["arm", "season", "model", "split", "rank_val",
                                             "equivalentes_omitidos", "top5", "campeao"]]
        met = pd.concat([met, em.merge(chave, on=["arm", "season", "model", "split"], how="left")],
                        ignore_index=True)
        preds += extra_p

    met["grupo"] = met.arm.str.replace("so_", "")
    met.to_parquet(SAIDA / "08_grupos_isolados_metricas.parquet", **PARQ)
    pr = pd.concat(preds, ignore_index=True)
    lin = linhas(pd.concat([val, teste]))
    pr = pr.merge(lin[["row_id", "estacao", "time", "y_true", "era5_gust_max"]], on="row_id", how="left")
    for c in ("arm", "season", "model", "split"):
        pr[c] = pr[c].astype("category")
    pr.to_parquet(SAIDA / "08_grupos_isolados_predicoes.parquet", **PARQ)

    melhores(met, pr, lin)


def melhores(met, pr, lin):
    """Campeão e top-5 de cada grupo × trimestre (média ± dp das 5 seeds) + ERA5 e ruído,
    e o agregado dos 4 trimestres (campeão de cada trimestre) com IC por blocos."""
    te = met[(met.split == "test") & met.top5]
    agg = (te.groupby(["arm", "season", "model", "rank_val"])
             .agg(RMSE_test=("RMSE", "mean"), RMSE_test_dp_seeds=("RMSE", "std"),
                  RMSE_P90_test=("RMSE_P90", "mean"), Bias_test=("Bias", "mean"),
                  Bias_P90_test=("Bias_P90", "mean"), R2_test=("R2", "mean"), n_seeds=("model_seed", "nunique"))
             .reset_index())
    va = met[(met.split == "val") & met.top5 & (met.model_seed == SEED)][["arm", "season", "model", "R2", "RMSE_P90"]]
    agg = agg.merge(va.rename(columns={"R2": "R2_val", "RMSE_P90": "RMSE_P90_val"}),
                    on=["arm", "season", "model"], how="left")
    agg["campeao"] = agg.rank_val == 1

    # referências por trimestre: ERA5 bruto
    t = lin[lin.split == "test"]
    era = [{"arm": "era5_bruto", "season": s, "model": "ERA5 (rajada bruta)", "rank_val": 1, "campeao": True,
            **{f"{k}_test": v for k, v in metricas(d.y_true.to_numpy(float), d.era5_gust_max.to_numpy(float)).items()}}
           for s, d in t.groupby("season")]
    agg = pd.concat([agg, pd.DataFrame(era)], ignore_index=True)
    agg["escopo"] = "trimestre"

    # agregado: campeão de cada trimestre (seed 42) junto, IC bootstrap em blocos (ano, mês)
    camp = met[(met.split == "test") & met.campeao & (met.model_seed == SEED)][["arm", "season", "model"]]
    p = pr[(pr.split == "test") & (pr.model_seed == SEED)]
    p = p.merge(camp, on=["arm", "season", "model"]).merge(lin[["row_id", "year", "month"]], on="row_id")
    tot = []
    for arm, d in p.groupby("arm", observed=True):
        lo, hi = bootstrap_blocos(d, "y_pred")
        tot.append({"arm": arm, "season": "todos", "escopo": "agregado (campeão de cada trimestre)",
                    "model": " / ".join(f"{s}:{m}" for s, m in camp[camp.arm == arm][["season", "model"]].values),
                    **{f"{k}_test": v for k, v in metricas(d.y_true.to_numpy(float), d.y_pred.to_numpy(float)).items()},
                    "RMSE_test_ic95_inf": lo, "RMSE_test_ic95_sup": hi})
    t2 = t.assign(y_pred=t.era5_gust_max)
    lo, hi = bootstrap_blocos(t2, "y_pred")
    tot.append({"arm": "era5_bruto", "season": "todos", "escopo": "agregado", "model": "ERA5 (rajada bruta)",
                **{f"{k}_test": v for k, v in metricas(t2.y_true.to_numpy(float), t2.y_pred.to_numpy(float)).items()},
                "RMSE_test_ic95_inf": lo, "RMSE_test_ic95_sup": hi})
    out = pd.concat([agg, pd.DataFrame(tot)], ignore_index=True)
    out["grupo"] = out.arm.str.replace("so_", "")
    out["descricao_grupo"] = out.grupo.map(DESCR_GRUPO)
    out.to_parquet(SAIDA / "08_grupos_isolados_melhores.parquet", **PARQ)
    print(out[out.escopo != "trimestre"][["arm", "RMSE_test", "RMSE_test_ic95_inf",
                                          "RMSE_test_ic95_sup", "R2_test"]].to_string(index=False))


# ── compilar o que o estudo já gravou ──────────────────────────────────────────

def etapa_compilar(dados):
    meta = dados[0]
    # 04: efeitos dos grupos (árvores, 5 seeds, Modal) + LSTM (seed 42)
    partes = []
    for origem, pasta in (("arvores_top5_5seeds", ESTUDO_M / "summary/main"),
                          ("lstm_seed42", ESTUDO / "summary/lstm")):
        for nome in ("effects", "effects_by_season", "ranking_groups"):
            f = pasta / f"{nome}.csv"
            if f.exists():
                partes.append(pd.read_csv(f).assign(origem=origem, tabela=nome))
    ef = pd.concat(partes, ignore_index=True)
    for c in ef.columns:
        if ef[c].dtype == object:
            ef[c] = ef[c].astype("string")
    ef.to_parquet(SAIDA / "04_grupos_efeitos.parquet", **PARQ)

    # 05 / 06: predições e métricas dos arms no teste (estudo de referência, 5 seeds)
    teste = pd.read_parquet(ESTUDO_M / "data/test.parquet")
    lin = linhas(teste.assign(_split="test")).set_index("row_id")
    arms_p = ("base", "sel__val12", "full", "drop_grp__grupo1", "add_grp__grupo2",
              "add_grp__grupo3", "add_grp__grupo4", "ctrl__noise", "ctrl__perm_static")
    preds, mets = [], []
    for pasta in sorted((ESTUDO_M / "units").iterdir()):
        tag = pasta.name
        if tag.endswith("_triage") or "lstm" in tag:
            continue
        seed = 42 if tag == "full" else int(tag.split("_s")[-1])
        for f in pasta.glob("metrics__*.parquet"):
            mets.append(pd.read_parquet(f).assign(model_seed=seed))
        for f in pasta.glob("resid__*.parquet"):
            _, s, arm = f.stem.split("__", 2)
            if arm not in arms_p:
                continue
            r = pd.read_parquet(f)
            y = lin.loc[r.row_id, "y_true"].to_numpy()
            for m in r.columns[1:]:
                preds.append(pd.DataFrame({"arm": arm, "season": s, "model": m, "model_seed": seed,
                                           "row_id": r.row_id.to_numpy(),
                                           "y_pred": (r[m].to_numpy() + y).astype("float32")}))
    pr = pd.concat(preds, ignore_index=True)
    pr = pr.merge(lin[["estacao", "time", "year", "month", "latitude", "longitude", "y_true",
                       "era5_gust_max"]].reset_index(), on="row_id", how="left")
    for c in ("arm", "season", "model"):
        pr[c] = pr[c].astype("category")
    pr.to_parquet(SAIDA / "05_comparacao_arms_teste.parquet", **PARQ)

    me = pd.concat(mets, ignore_index=True)
    keep = ["arm", "season", "tag", "model", "model_seed", "split", "n", "R2", "RMSE", "Bias",
            "Bias_P90", "RMSE_P90", "n_nonfinite"]
    me[keep].to_parquet(SAIDA / "06_metricas_arms.parquet", **PARQ)

    # conferência: RMSE refeito das predições == RMSE gravado (antes do truncamento pode diferir
    # só em modelos degenerados)
    chk = pr.groupby(["arm", "season", "model", "model_seed"], observed=True).apply(
        lambda d: rmse(d.y_true, d.y_pred)).rename("rmse_pred").reset_index()
    chk = chk.merge(me[me.split == "test"], on=["arm", "season", "model", "model_seed"])
    print(f"[compilar] |RMSE(preds) − RMSE(metrics)| máx = {(chk.rmse_pred - chk.RMSE).abs().max():.2e}")
    print(f"[compilar] {len(pr):,} predições de arms, {len(me):,} linhas de métricas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapas", default="compilar,permutacao,grupos")
    etapas = ap.parse_args().etapas.split(",")
    SAIDA.mkdir(exist_ok=True)
    dados = carregar()
    sel = load_selection(RAIZ / "config/selected_features_val12.json")
    assert len(columns_of(sel["variables"], dados[1])) == sel["n_columns"]
    if "compilar" in etapas:
        etapa_compilar(dados)
    if "permutacao" in etapas:
        etapa_permutacao(dados, sel)
    if "grupos" in etapas:
        etapa_grupos(dados)


if __name__ == "__main__":
    main()
