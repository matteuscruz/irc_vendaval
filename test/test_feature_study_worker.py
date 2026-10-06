"""Worker do estágio `fit` (src/feature_study/worker.py) e `prepare` ponta a ponta.

Roda o LazyPredict de verdade, mas com 2 regressores baratos e dados minúsculos
(nada de TF nem dos 39 modelos: rodar tudo localmente já travou esta máquina).
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from src.feature_study.core.analysis import compare_units
from src.feature_study.core.arms import Arm
from src.feature_study.core.worker import _features_fingerprint
from src.feature_study.core.config import (
    MAIN_TAGS, MODEL_SEED, MODEL_SEEDS, seed_tag,
)
from src.feature_study.core.worker import (
    _models_fingerprint, fit_arm, load_unit_frames, run_unit, select_regressors,
)
from src.pipelines.common import TARGET_VAR

REGS = [Ridge, ExtraTreesRegressor]
# Conjunto custom de modelos carimba `custom:<impressão digital do conjunto>`,
# não a string literal `"custom"`: dois conjuntos diferentes precisam de
# `models_mode` diferentes, senão o `skip_existing` reaproveita o ajuste errado.
MODO_CUSTOM = "custom:" + _models_fingerprint(REGS)


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
    from src.feature_study.core.worker import _safe_metrics

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
    # `preds__*` só sai para os arms de PRED_DUMP_ARMS (base e full), e é o
    # insumo da série temporal e do mapa espacial por estação.
    assert names == ["metrics__DJF__base.parquet", "metrics__DJF__full.parquet",
                     "preds__DJF__base.parquet", "preds__DJF__full.parquet",
                     "resid__DJF__base.parquet", "resid__DJF__full.parquet"]
    assert first[0].name.startswith("metrics__")        # contrato: métricas primeiro

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
    assert (pd.read_parquet(m_path)["models_mode"] == MODO_CUSTOM).all()

    # Simula o arquivo do piloto: mesmo nome, outro conjunto de modelos.
    old = pd.read_parquet(m_path).assign(models_mode="fast3")
    old.to_parquet(m_path, index=False)
    before = m_path.stat().st_mtime_ns

    run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)
    assert m_path.stat().st_mtime_ns != before                       # refez
    assert (pd.read_parquet(m_path)["models_mode"] == MODO_CUSTOM).all()

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











# ── Fingerprint de features (protege contra reuso indevido) ─────────────────

def test_the_same_arm_name_with_a_different_feature_set_is_refit_not_reused(data_dir, tmp_path):
    """O caso real que motivou isto: uma nova rodada de `prepare` engorda o
    `full` (mais colunas em `new_features`) sem trocar o NOME do arm. Sem o
    fingerprint, `_is_current` reaproveitaria em silêncio um ajuste feito com
    as features ANTIGAS — 20 dos 37 arms do estudo real caem nesse caso."""
    out = tmp_path / "out"
    written = run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)
    m_path, before = written[0], written[0].stat().st_mtime_ns

    # Mesmas colunas de ARM_FULL, só em outra ordem: o fingerprint (que ordena
    # antes de gerar o hash) tem de ver isso como o MESMO arm — não é a ordem
    # que caracteriza uma mudança de composição, é o conjunto.
    full_reordenado = Arm("full", ("f_util", "f_ruido"), "full", "core")
    run_unit(data_dir, out, "r0", "DJF", [full_reordenado], regressors=REGS)
    assert m_path.stat().st_mtime_ns == before

    # Mesmo NOME ("full"), composição REALMENTE diferente — o caso real do
    # estudo, em que uma nova rodada de `prepare` engorda `full`.
    full_menor = Arm("full", ("f_ruido",), "full", "core")
    run_unit(data_dir, out, "r0", "DJF", [full_menor], regressors=REGS)
    assert m_path.stat().st_mtime_ns != before
    assert (pd.read_parquet(m_path)["features_fp"] ==
            _features_fingerprint(("f_ruido",))).all()


def test_a_parquet_from_before_the_features_fingerprint_is_not_considered_current(data_dir, tmp_path):
    """Não existe um valor seguro para assumir em arquivo antigo: foi exatamente
    uma mudança de composição sem aviso que motivou criar a coluna. Ausente ⇒
    refaz, sempre."""
    from src.feature_study.core.worker import _is_current

    out = tmp_path / "out"
    m_path = run_unit(data_dir, out, "r0", "DJF", [ARM_FULL], regressors=REGS)[0]
    antigo = pd.read_parquet(m_path).drop(columns=["features_fp"])
    antigo.to_parquet(m_path, index=False)

    fp = _features_fingerprint(ARM_FULL.features)
    assert not _is_current(m_path, MODO_CUSTOM, 42, fp)


def test_features_fingerprint_is_stable_and_order_independent():
    a = _features_fingerprint(("nf_fg10_max", "nf_cape_max", "wind_mag"))
    b = _features_fingerprint(("wind_mag", "nf_fg10_max", "nf_cape_max"))
    c = _features_fingerprint(("nf_fg10_max", "nf_cape_max"))

    assert a == b
    assert a != c
    assert _features_fingerprint(a.split()) == _features_fingerprint(a.split())  # determinístico








def test_triage_and_study_never_share_an_output_folder(data_dir, tmp_path):
    """Triagem (39 modelos) e estudo (5) gravam o MESMO nome de arquivo para o
    arm `base`. Na mesma pasta, o estudo sobrescrevia a leaderboard da triagem —
    acontecido de verdade no `cluster3_raw` — e rodar o `screen` depois escolhia
    entre os 5 que já tinham sido escolhidos (um top-5 encolheu para 4)."""
    from src.feature_study.core.config import seed_tag, triage_tag

    assert triage_tag(42) != seed_tag(42) and triage_tag(43) != seed_tag(43)

    out = tmp_path / "out"
    triagem = run_unit(data_dir, out, "r0", "DJF", [ARM_BASE], regressors=[Ridge, ExtraTreesRegressor],
                       out_tag=triage_tag(42))[0]
    antes = pd.read_parquet(triagem).model.nunique()

    run_unit(data_dir, out, "r0", "DJF", [ARM_BASE], regressors=[Ridge], out_tag=seed_tag(42))

    depois = pd.read_parquet(triagem)
    assert depois.model.nunique() == antes == 2              # a triagem continua com os dois modelos
