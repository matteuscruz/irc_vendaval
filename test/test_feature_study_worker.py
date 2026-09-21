"""Worker do estágio `fit` (src/feature_study/worker.py) e `prepare` ponta a ponta.

Roda o LazyPredict de verdade, mas com 2 regressores baratos e dados minúsculos
(nada de TF nem dos 39 modelos: rodar tudo localmente já travou esta máquina).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from src.feature_study.analysis import compare_units
from src.feature_study.arms import Arm
from src.feature_study.config import (
    LOSS_MODELS, MAIN_TAGS, MODEL_SEED, MODEL_SEEDS, seed_tag,
)
from src.feature_study.worker import fit_arm, load_unit_frames, run_unit, select_regressors
from src.pipelines.common import TARGET_VAR

REGS = [Ridge, ExtraTreesRegressor]


def _frame(n, split, seasons=("DJF",), seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "row_id": np.arange(n) + (0 if split != "test" else 10_000),
        "estacao": rng.choice(list("AB"), n),
        "time": pd.Timestamp("2010-01-01") + pd.to_timedelta(np.arange(n), unit="D"),
        "season": rng.choice(list(seasons), n),
        "year": 2010, "month": 1, "_split": split,
        "f_util": rng.normal(size=n), "f_ruido": rng.normal(size=n),
    })
    df[TARGET_VAR] = 5.0 + 2.0 * df["f_util"] + rng.normal(0, 0.3, n)   # só f_util explica o alvo
    return df


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    sample = pd.concat([_frame(300, "train", ("DJF", "MAM"), 1), _frame(80, "val", ("DJF", "MAM"), 2)])
    sample.to_parquet(d / "sample_r0.parquet", index=False)
    _frame(120, "test", ("DJF", "MAM"), 3).to_parquet(d / "test.parquet", index=False)
    return d


ARM_BASE = Arm("base", ("f_ruido",), "base", "core")
ARM_FULL = Arm("full", ("f_ruido", "f_util"), "full", "core")


def _nan_on_test(cls, name):
    """Regressor que prediz normalmente na validação (1ª chamada) e devolve NaN
    depois — o que um modelo divergente faz num dado fora da faixa de treino."""
    calls = {"n": 0}

    def predict(self, X):
        calls["n"] += 1
        out = super(klass, self).predict(X)
        return out if calls["n"] == 1 else np.full_like(out, np.nan)

    klass = type(name, (cls,), {"predict": predict})
    return klass


def test_a_model_with_nonfinite_test_predictions_is_recorded_not_fatal(data_dir):
    """Um regressor divergente (entre 39) derrubou um lote inteiro de arms. Tem
    de virar uma linha marcada como falha, sem afetar os demais e sem NaN
    preenchido nem truncado."""
    train, val, test = load_unit_frames(data_dir, "r0", "DJF")
    bad = _nan_on_test(ExtraTreesRegressor, "ExtraTreesRegressor")   # nome de referência
    metrics, resid, _ = fit_arm(train, val, test, ["f_util", "f_ruido"], [Ridge, bad])

    et = metrics[(metrics["model"] == "ExtraTreesRegressor") & (metrics["split"] == "test")].iloc[0]
    ok = metrics[(metrics["model"] == "Ridge") & (metrics["split"] == "test")].iloc[0]
    assert et["n_nonfinite"] == len(test) and np.isnan(et["RMSE"])
    assert ok["n_nonfinite"] == 0 and np.isfinite(ok["RMSE"])
    assert "ExtraTreesRegressor" not in resid.columns      # sem resíduo: a análise o descarta
    assert "Ridge" in resid.columns


def test_infinite_prediction_is_a_failure_not_clipped_to_the_physical_maximum():
    from src.feature_study.worker import _safe_metrics

    m, bad = _safe_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.inf, 3.0]))
    assert m is None and bad == 1
    m, bad = _safe_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1e9, 3.0]))   # finito ⇒ truncado
    assert bad == 0 and m["RMSE"] > 0


def test_unit_uses_only_the_requested_season(data_dir):
    train, val, test = load_unit_frames(data_dir, "r0", "MAM")
    assert set(train["season"]) == set(val["season"]) == set(test["season"]) == {"MAM"}
    assert set(train["_split"]) == {"train"} and set(test["_split"]) == {"test"}


def test_fit_arm_scores_val_and_the_full_test_and_keeps_reference_residuals(data_dir):
    train, val, test = load_unit_frames(data_dir, "r0", "DJF")
    metrics, resid, secs = fit_arm(train, val, test, ["f_util", "f_ruido"], REGS)

    assert set(metrics["split"]) == {"val", "test"}
    assert set(metrics["model"]) == {"Ridge", "ExtraTreesRegressor"}
    assert metrics[metrics["split"] == "test"]["n"].eq(len(test)).all()     # teste COMPLETO
    assert list(resid.columns) == ["row_id", "Ridge", "ExtraTreesRegressor"]
    assert resid["row_id"].tolist() == test["row_id"].tolist()
    assert secs > 0


def test_residual_is_prediction_minus_observation_with_the_project_sign(data_dir):
    """Bias = mean(pred − y): o resíduo guardado tem de ter o MESMO sinal, senão
    a análise leria o viés invertido."""
    train, val, test = load_unit_frames(data_dir, "r0", "DJF")
    metrics, resid, _ = fit_arm(train, val, test, ["f_util"], REGS)
    row = metrics[(metrics["split"] == "test") & (metrics["model"] == "Ridge")].iloc[0]

    assert resid["Ridge"].astype(float).mean() == pytest.approx(row["Bias"], abs=1e-4)
    assert np.sqrt((resid["Ridge"].astype(float) ** 2).mean()) == pytest.approx(row["RMSE"], abs=1e-4)


def test_the_arm_with_the_informative_feature_is_better(data_dir):
    train, val, test = load_unit_frames(data_dir, "r0", "DJF")
    m_base, _, _ = fit_arm(train, val, test, ARM_BASE.features, REGS)
    m_full, _, _ = fit_arm(train, val, test, ARM_FULL.features, REGS)

    def rmse(m):
        return m[(m["split"] == "test") & (m["model"] == "Ridge")]["RMSE"].iloc[0]

    assert rmse(m_full) < 0.5 * rmse(m_base)


def test_unit_writes_deterministic_files_and_is_idempotent(data_dir, tmp_path):
    out = tmp_path / "out"
    first = run_unit(data_dir, out, "r0", "DJF", [ARM_BASE, ARM_FULL], regressors=REGS)
    names = sorted(p.name for p in first)
    assert names == ["metrics__DJF__base.parquet", "metrics__DJF__full.parquet",
                     "resid__DJF__base.parquet", "resid__DJF__full.parquet"]

    mtimes = {p: p.stat().st_mtime_ns for p in first}
    run_unit(data_dir, out, "r0", "DJF", [ARM_BASE, ARM_FULL], regressors=REGS)
    assert {p: p.stat().st_mtime_ns for p in first} == mtimes           # não reajustou

    meta = pd.read_parquet(first[0])
    assert {"arm", "season", "tag", "fit_seconds", "peak_rss_mb", "model_seed"} <= set(meta.columns)
    assert (meta["model_seed"] == 42).all()                              # seed FIXA


def test_existing_units_from_another_model_set_are_refit_not_skipped(data_dir, tmp_path):
    """O piloto grava `units/full/` com 3 modelos; o run completo grava a mesma
    pasta com 39. Se o `fit` idempotente pulasse por o arquivo existir, o run
    completo herdaria as métricas incompletas do piloto sem aviso algum."""
    out = tmp_path / "out"
    written = run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)
    m_path = written[0]
    assert (pd.read_parquet(m_path)["models_mode"] == "custom").all()

    # Simula o arquivo do piloto: mesmo nome, outro conjunto de modelos.
    old = pd.read_parquet(m_path).assign(models_mode="fast3")
    old.to_parquet(m_path, index=False)
    before = m_path.stat().st_mtime_ns

    run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)
    assert m_path.stat().st_mtime_ns != before                       # refez
    assert (pd.read_parquet(m_path)["models_mode"] == "custom").all()

    # Arquivo antigo, sem a coluna (gerado antes da regra), também é refeito.
    pd.read_parquet(m_path).drop(columns=["models_mode"]).to_parquet(m_path, index=False)
    before = m_path.stat().st_mtime_ns
    run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)
    assert m_path.stat().st_mtime_ns != before


def test_same_unit_run_twice_is_deterministic(data_dir, tmp_path):
    """Seeds fixas ⇒ reproduzível. É o que o piloto confirma no Modal."""
    run_unit(data_dir, tmp_path / "o1", "r0", "DJF", [ARM_FULL], regressors=REGS)
    run_unit(data_dir, tmp_path / "o2", "r0", "DJF", [ARM_FULL], regressors=REGS)
    import shutil
    shutil.copytree(tmp_path / "o2" / "units" / "r0", tmp_path / "o1" / "units" / "r0_bis")

    assert compare_units(tmp_path / "o1", "r0", "r0_bis") == 0.0


def test_unit_without_test_or_train_rows_fails_loudly(data_dir, tmp_path):
    with pytest.raises(ValueError, match="sem treino"):
        run_unit(data_dir, tmp_path / "o", "r0", "JJA", [ARM_BASE], regressors=REGS)


def test_select_regressors_modes_and_errors():
    assert len(select_regressors("all")) == 39
    assert {c.__name__ for c in select_regressors("fast3")} == {
        "CatBoostRegressor", "LGBMRegressor", "HistGradientBoostingRegressor"}
    assert len(select_regressors("reference")) == 7
    with pytest.raises(ValueError, match="models inválido"):
        select_regressors("todos")


# ── prepare ponta a ponta ───────────────────────────────────────────────────

def test_prepare_end_to_end_on_the_synthetic_cluster(synthetic_nf_raw_dir, synthetic_shp_dir, tmp_path):
    from src.feature_study.prepare import prepare

    meta = prepare(str(synthetic_nf_raw_dir), str(synthetic_shp_dir), tmp_path / "study",
                   eps=0.15, seeds=(1, 2), pilot_sizes=(60,), cluster_id=9)
    data = tmp_path / "study" / "data"

    for f in ("meta.json", "arms.json", "test.parquet", "sample_full.parquet", "sample_r0.parquet",
              "sample_r1.parquet", "sample_n60_r0.parquet", "representativeness.csv"):
        assert (data / f).exists(), f

    # Cobertura: o cluster 10 (sem features novas) foi descartado; o 9 ficou.
    assert "10" in meta["clusters_dropped_by_coverage"]
    assert meta["cluster_id"] == 9
    # Sem as duplicatas conhecidas.
    assert "nf_ws10_max" not in meta["new_features"] and "nf_ws10_std" in meta["new_features"]

    test = pd.read_parquet(data / "test.parquet")
    r0 = pd.read_parquet(data / "sample_r0.parquet")
    full = pd.read_parquet(data / "sample_full.parquet")
    assert set(test["_split"]) == {"test"} and set(r0["_split"]) <= {"train", "val"}
    assert not (set(zip(r0["estacao"], r0["time"])) & set(zip(test["estacao"], test["time"])))
    assert len(r0) < len(full)                                           # de fato amostrou
    assert full["row_id"].is_unique and test["row_id"].is_unique
    assert set(full["row_id"]).isdisjoint(set(test["row_id"]))

    # Réplicas diferentes ⇒ amostras diferentes; mesmas linhas para qualquer arm.
    r1 = pd.read_parquet(data / "sample_r1.parquet")
    assert set(r0["row_id"]) != set(r1["row_id"])

    arms = [Arm.from_dict(d) for d in json.loads((data / "arms.json").read_text())]
    assert {"base", "full", "add__nf_ws10_std", "drop_grp__cape", "ctrl__noise"} <= {a.name for a in arms}
    cols = set(r0.columns)
    assert all(set(a.features) <= cols for a in arms)                    # todo arm cabe na amostra
    assert r0[sorted(cols - {"time"})].select_dtypes("number").notna().all().all()


def test_prepare_defaults_to_the_full_training_set_and_removes_stale_samples(
    synthetic_nf_raw_dir, synthetic_shp_dir, tmp_path,
):
    """Decisão: treino COMPLETO, sem amostragem. Uma amostra de execução
    anterior ao lado de `sample_full` convida a analisar a errada — é removida."""
    from src.feature_study.prepare import prepare

    study = tmp_path / "study"
    data = study / "data"
    data.mkdir(parents=True)
    for stale in ("sample_r0.parquet", "sample_n2000_r0.parquet"):
        (data / stale).write_bytes(b"velho")
    (data / "representativeness.csv").write_text("velho")

    meta = prepare(str(synthetic_nf_raw_dir), str(synthetic_shp_dir), study, cluster_id=9)

    assert meta["sampling"] is False and meta["n_target_per_season"] is None
    assert sorted(p.name for p in data.glob("sample_*.parquet")) == ["sample_full.parquet"]
    assert not (data / "representativeness.csv").exists()
    assert meta["flags"] == {}
    assert set(meta["train_rows_per_season"]) == {"DJF", "MAM", "JJA", "SON"}
    full = pd.read_parquet(data / "sample_full.parquet")
    assert set(full["_split"]) == {"train", "val"} and len(full) == meta["n_trainval_rows"]


def test_pilot_sizes_without_seeds_is_an_error(synthetic_nf_raw_dir, synthetic_shp_dir, tmp_path):
    from src.feature_study.prepare import prepare

    with pytest.raises(ValueError, match="exige ao menos uma seed"):
        prepare(str(synthetic_nf_raw_dir), str(synthetic_shp_dir), tmp_path / "s",
                pilot_sizes=(50,), cluster_id=9)


# ── Seeds fixas do modelo (uma réplica de treino por seed) ──────────────────

def test_the_five_model_seeds_are_fixed_distinct_and_keep_the_previous_run_as_replicate_one():
    assert len(MODEL_SEEDS) == 5 and len(set(MODEL_SEEDS)) == 5
    assert MODEL_SEEDS[0] == MODEL_SEED == 42
    assert seed_tag(MODEL_SEED) == "full"                     # o run anterior continua valendo
    assert len(set(MAIN_TAGS)) == 5 and MAIN_TAGS[0] == "full"


def test_each_seed_writes_its_own_folder_and_records_its_seed(data_dir, tmp_path):
    out = tmp_path / "out"
    for seed in (42, 43):
        run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS, seed=seed, out_tag=f"r0_s{seed}")
    for seed in (42, 43):
        meta = pd.read_parquet(out / "units" / f"r0_s{seed}" / "metrics__DJF__full.parquet")
        assert (meta["model_seed"] == seed).all()


def test_only_stochastic_models_change_with_the_seed(data_dir, tmp_path):
    """Ridge é determinístico (a seed não o afeta); ExtraTrees sorteia. É por isso
    que repetir seeds só acrescenta informação nos modelos que sorteiam."""
    out = tmp_path / "out"
    rmse = {}
    for seed in (42, 43):
        run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS, seed=seed, out_tag=f"s{seed}")
        m = pd.read_parquet(out / "units" / f"s{seed}" / "metrics__DJF__full.parquet")
        rmse[seed] = m[m["split"] == "test"].set_index("model")["RMSE"]

    assert rmse[42]["Ridge"] == rmse[43]["Ridge"]
    assert rmse[42]["ExtraTreesRegressor"] != rmse[43]["ExtraTreesRegressor"]


def test_a_unit_from_another_seed_is_refit_not_skipped(data_dir, tmp_path):
    out = tmp_path / "out"
    written = run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS, seed=42, out_tag="x")
    m_path, before = written[0], written[0].stat().st_mtime_ns
    run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS, seed=42, out_tag="x")
    assert m_path.stat().st_mtime_ns == before                # mesma seed: pula

    run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS, seed=43, out_tag="x")
    assert m_path.stat().st_mtime_ns != before                # outra seed: refaz
    assert (pd.read_parquet(m_path)["model_seed"] == 43).all()


# ── Eixo de perda ───────────────────────────────────────────────────────────

ARM_LOSS = Arm("full__exp90", ("f_ruido", "f_util"), "loss", "losses",
               subject="exp90", reference="full", role="add", loss="exp90", axis="loss")


def test_a_parquet_from_before_the_loss_column_is_still_considered_current(data_dir, tmp_path):
    """O volume do Modal já tem o estudo inteiro (37 arms × 4 trimestres × 5
    seeds) gravado ANTES da coluna `loss`. Se `_is_current` a exigisse junto das
    outras, cada um desses arquivos levantaria, devolveria False e seria refeito
    em silêncio — refazendo (e pagando) o estudo todo."""
    from src.feature_study.worker import _is_current

    out = tmp_path / "out"
    m_path = run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)[0]
    antigo = pd.read_parquet(m_path).drop(columns=["loss"])
    antigo.to_parquet(m_path, index=False)

    assert _is_current(m_path, "custom", 42, "mse")
    assert not _is_current(m_path, "custom", 42, "exp90")


def test_two_different_losses_do_not_skip_each_other(data_dir, tmp_path):
    """Mesmo `models_mode` e mesma seed: sem a perda na chave de frescor, o
    segundo braço herdaria as métricas do primeiro sem nenhum aviso."""
    from src.feature_study.worker import _is_current

    out = tmp_path / "out"
    m_path = run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)[0]

    assert _is_current(m_path, "custom", 42, "mse")
    assert not _is_current(m_path, "custom", 42, "huber44")


def test_each_arm_gets_the_regressors_of_its_own_loss(data_dir, tmp_path):
    """A lista de regressores era montada UMA vez por unidade. Com dois eixos no
    mesmo lote, o arm de perda precisa das subclasses e o arm MSE não."""
    out = tmp_path / "out"
    run_unit(data_dir, out, "r0", "DJF", [ARM_FULL, ARM_LOSS])

    mse = pd.read_parquet(out / "units" / "r0" / "metrics__DJF__full.parquet")
    perda = pd.read_parquet(out / "units" / "r0" / "metrics__DJF__full__exp90.parquet")

    assert (mse["loss"] == "mse").all()
    assert (perda["loss"] == "exp90").all()
    assert set(perda["model"]) == set(LOSS_MODELS)          # só os 3 que trocam de perda
    assert set(mse["model"]) > set(perda["model"])          # o arm MSE roda os 39


def test_the_loss_arm_writes_residuals_under_its_own_suffixed_name(data_dir, tmp_path):
    """O nome com SUFIXO é o que impede o glob `resid__*__full.parquet` de
    capturar também os resíduos da perda."""
    out = tmp_path / "out"
    run_unit(data_dir, out, "r0", "DJF", [ARM_LOSS])
    escritos = {p.name for p in (out / "units" / "r0").glob("*.parquet")}

    assert escritos == {"metrics__DJF__full__exp90.parquet", "resid__DJF__full__exp90.parquet"}


def test_select_regressors_narrows_to_the_models_that_can_change_loss():
    """`reference` pede 7 modelos, mas só 3 aceitam trocar a perda. Isso NÃO é
    erro: o eixo é definido sobre um subconjunto, e pedir os 7 aqui derrubaria a
    unidade inteira."""
    assert [c.__name__ for c in select_regressors("all", "exp70")] == list(LOSS_MODELS)
    assert [c.__name__ for c in select_regressors("reference", "exp70")] == list(LOSS_MODELS)
    assert len(select_regressors("all")) == 39


def test_the_loss_axis_runs_end_to_end_from_prepare_to_aggregate(
        synthetic_nf_raw_dir, synthetic_shp_dir, tmp_path):
    """prepare(losses) → fit → aggregate no dado sintético. Os testes por módulo
    não veem a FIAÇÃO: arms.json carregando os campos novos, o worker escolhendo
    os regressores pelo `arm.loss` e a análise achando os resíduos pelo nome com
    sufixo. É o ensaio local do que vai rodar no Modal."""
    from src.feature_study.analysis import load_arms, run_aggregate
    from src.feature_study.prepare import prepare

    study = tmp_path / "study"
    meta = prepare(str(synthetic_nf_raw_dir), str(synthetic_shp_dir), study,
                   arm_sets=("core", "losses"), cluster_id=9)
    assert meta["loss_arms"] and meta["loss_models"] == list(LOSS_MODELS)

    arms = load_arms(study / "data")
    escolhidos = [a for a in arms if a.name in ("base", "full", "full__exp90")]
    # Os quatro trimestres: `load_residuals` exige o vetor COMPLETO do teste, e
    # um trimestre faltando descarta o arm inteiro da análise.
    for season in sorted(pd.read_parquet(study / "data" / "test.parquet")["season"].unique()):
        run_unit(study / "data", study, "full", season, escolhidos, models="fast3")

    res = run_aggregate(study, study / "data", ["full"], label="t", n_boot=50)
    fr = res["loss_frontier"]

    assert not fr.empty and set(fr["comparison"]) == {"full__exp90"}
    eff = res["effects"]
    assert set(eff["axis"]) == {"features", "loss"}
    # Cada eixo é julgado na SUA métrica primária e tem o próprio SESOI.
    julgado = eff[eff["verdict"] != ""].groupby("axis")["metric"].unique()
    assert set(julgado["features"]) == {"rmse"} and set(julgado["loss"]) == {"rmse_p90"}
    assert eff.groupby("axis")["sesoi"].nunique().eq(1).all()
