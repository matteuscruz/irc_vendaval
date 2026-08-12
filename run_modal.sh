#!/bin/bash
# run_ablation_modal_all.sh
#
# Roda a matriz de ablation completa no Modal (nuvem) para as 3 pipelines de
# uma vez só: gera os dados sintéticos (cluster_gan) uma única vez e treina
# os 6 braços — original / +sintético / +features novas (ERA5-18UTC + BT55) /
# tudo junto / +ERA5-Basin (grade regional) / tudo junto + Basin — em
# cluster_lazy, cluster_mlp e cluster_lstm (nessa ordem, mesma ordem de
# run_all_modal.sh). 23 passos no total (1 GAN + 6×3 treinos + 3 agregações
# por pipeline + 1 comparativo cruzado final).
#
# Retomável: antes de cada passo, checa se o results.csv daquele braço já
# existe localmente (baixado de uma run anterior, completa ou interrompida)
# e PULA o treino se sim — não precisa começar do zero toda vez. Além disso,
# consulta o volume Modal (irc-vendaval-artifacts) como fonte de verdade
# remota: se um braço já terminou de treinar na nuvem mas ainda não foi
# baixado pra esta máquina, sincroniza (--only-download) em vez de re-treinar.
# Use FORCE=1 pra ignorar tudo isso (local e remoto) e re-rodar do zero. Rode
# scripts/check_ablation_status.py antes pra ver o que já está pronto localmente.
#
# Mapas espaciais desabilitados em todos (foco em comparar métricas). Ao
# final, agrega cada pipeline no seu próprio comparativo (scripts/run_ablation*.py)
# e depois compara as 18 combinações (3 pipelines × 6 braços) juntas, de forma
# automática (scripts/compare_ablation_all.py).
#
# Uso:
#   ./run_ablation_modal_all.sh [lstm_config.yaml] [gan_exp_name] [existing_synthetic_csv]
#   FORCE=1 ./run_ablation_modal_all.sh   # ignora o que já existe
#
# lstm_config.yaml é opcional — default: config/experiment_cluster_lstm_modal.yaml
# (o mesmo default hardcoded em src/modal/cluster_lstm.py e usado por todo o
# ferramental de ablation). Só precisa ser informado se quiser rodar a matriz
# contra outro config LSTM.
#
# Se existing_synthetic_csv for informado e existir localmente, reaproveita
# esse CSV em vez de treinar o GAN de novo.
#
# Requer: modal CLI autenticado, dataset em dataset/raw e dataset/shp.

set -e

DEFAULT_LSTM_CONFIG="config/experiment_cluster_lstm_modal.yaml"
LSTM_CONFIG="${1:-$DEFAULT_LSTM_CONFIG}"
GAN_EXP="${2:-gan_v1}"
EXISTING_SYNTH="${3:-}"
FORCE="${FORCE:-0}"

if [ ! -f "$LSTM_CONFIG" ]; then
    echo "ERRO: LSTM config não encontrado em '${LSTM_CONFIG}'." >&2
    exit 1
fi

# src/modal/cluster_lstm.py monta "config/<config_name>" dentro do container —
# --config precisa ser só o nome do arquivo, não o caminho completo local
# (que já inclui "config/" pra o yaml.safe_load local acima funcionar).
LSTM_CONFIG_NAME="$(basename "$LSTM_CONFIG")"

GAN_LOCAL_DIR="artifacts/gan_modal"
LAZY_LOCAL_DIR="artifacts/lazy_modal"
LAZY_OUT_DIR="${LAZY_LOCAL_DIR}/lazy_clusters"
MLP_LOCAL_DIR="artifacts/mlp_modal"
MLP_OUT_DIR="${MLP_LOCAL_DIR}/mlp_clusters"
LSTM_LOCAL_DIR="artifacts"
LSTM_OUT_DIR_REMOTE=$(python3 -c "import yaml; print(yaml.safe_load(open('${LSTM_CONFIG}'))['experiment']['output_dir'])")
LSTM_OUT_DIR_LOCAL="${LSTM_LOCAL_DIR}${LSTM_OUT_DIR_REMOTE#/artifacts}"

# ── Consulta o volume Modal (fonte de verdade remota) ───────────────────────
# Uma única chamada, populando variáveis <PIPELINE>_<ARM>_REMOTE_DONE=0|1
# (e GAN_REMOTE_DONE) usadas por _maybe_sync_pipeline e pelo passo do GAN.
LSTM_REMOTE_SUBDIR="${LSTM_OUT_DIR_REMOTE#/artifacts/}"

if [ "$FORCE" != "1" ]; then
    MODAL_STATUS_OUTPUT="$(python3 scripts/check_modal_status.py \
        --gan-exp "$GAN_EXP" --lstm-remote-subdir "$LSTM_REMOTE_SUBDIR")" || true
    [ -n "$MODAL_STATUS_OUTPUT" ] && eval "$MODAL_STATUS_OUTPUT"
fi

# _run_step <label> <results_csv_check_path> <comando modal completo...>
# Pula o comando se o results.csv já existir localmente e FORCE != 1.
_run_step() {
    local label="$1"
    local check_path="$2"
    shift 2
    if [ "$FORCE" != "1" ] && [ -f "$check_path" ]; then
        echo -e "\n${label} — JÁ EXISTE (${check_path}), pulando. [FORCE=1 pra re-rodar]"
        return 0
    fi
    echo -e "\n${label}..."
    "$@"
}

# _maybe_sync_pipeline <label> <pipeline_local_dir> <pipeline_out_dir> <script_path> <VAR_PREFIX>
# Se FORCE!=1 e pelo menos um braço já está pronto no Modal mas ainda não
# existe localmente, baixa a árvore da pipeline (--only-download) ANTES do
# loop de _run_step, pra que o check de arquivo local de cada braço "acerte"
# e pule o treino corretamente.
_maybe_sync_pipeline() {
    local label="$1" local_dir="$2" out_dir="$3" script_path="$4" var_prefix="$5"

    [ "$FORCE" == "1" ] && return 0

    local arm need_sync=0 remote_var
    for arm in original synthetic newfeatures all basin all_basin; do
        remote_var="${var_prefix}_${arm^^}_REMOTE_DONE"
        if [ "${!remote_var:-0}" == "1" ] && [ ! -f "${out_dir}/${arm}/results.csv" ]; then
            need_sync=1
        fi
    done

    if [ "$need_sync" == "1" ]; then
        echo -e "\n${label} — resultado(s) já prontos no Modal, ainda não baixados; sincronizando (--only-download)..."
        modal run "$script_path" --only-download --local-dir "$local_dir"
    fi
}

echo "================================================================="
echo "   Matriz de Ablation Completa — lazy + mlp + lstm (Modal)"
echo "================================================================="
echo "   LSTM config: ${LSTM_CONFIG}"
echo "   FORCE=${FORCE} (1 = ignora o que já existe e re-roda tudo)"
echo "================================================================="

# ── -1. Cache do merge ERA5 (uma vez, reaproveitado por GAN+lazy+mlp+lstm) ──
# Sem isso, cada uma das ~20 invocações da matriz recalcula o merge
# INMET×ERA5(s) do zero (~15min-2h cada) — o cache elimina essa redundância
# e garante que todas as pipelines (inclusive o GAN) treinem sobre exatamente
# o mesmo dataset sincronizado.

CACHE_CHECK="$(modal volume ls irc-vendaval-dataset raw 2>/dev/null | grep -c era5_merged_cache.nc || true)"
if [ "$FORCE" != "1" ] && [ "$CACHE_CHECK" != "0" ]; then
    echo -e "\n[0/17] Cache do merge ERA5 já existe no volume, pulando. [FORCE=1 pra regenerar]"
else
    echo -e "\n[0/17] Gerando cache do merge ERA5 completo (INMET×ERA5-Stratified×18UTC×BT55×Basin)..."
    modal run src/modal/cluster_lazy.py --build-cache
fi

# ── 0. Dados sintéticos (uma vez, reaproveitados pelas 3 pipelines) ─────────

DEFAULT_SYNTH="${GAN_LOCAL_DIR}/gan_clusters/${GAN_EXP}/synthetic_augment.csv"

if [ -n "$EXISTING_SYNTH" ] && [ -f "$EXISTING_SYNTH" ]; then
    echo -e "\n[1/17] Reaproveitando CSV sintético existente: ${EXISTING_SYNTH}"
    SYNTHETIC_CSV="$EXISTING_SYNTH"
elif [ "$FORCE" != "1" ] && [ -f "$DEFAULT_SYNTH" ]; then
    echo -e "\n[1/17] JÁ EXISTE (${DEFAULT_SYNTH}), pulando geração. [FORCE=1 pra re-rodar]"
    SYNTHETIC_CSV="$DEFAULT_SYNTH"
elif [ "$FORCE" != "1" ] && [ "${GAN_REMOTE_DONE:-0}" == "1" ]; then
    echo -e "\n[1/17] Encontrado no Modal (gan_clusters/${GAN_EXP}/synthetic_augment.csv), ainda não local; sincronizando..."
    modal run src/modal/cluster_gan.py --only-download --local-dir "$GAN_LOCAL_DIR"
    SYNTHETIC_CSV="$DEFAULT_SYNTH"
else
    echo -e "\n[1/17] Gerando dados sintéticos (cluster_gan, exp=${GAN_EXP})..."
    modal run src/modal/cluster_gan.py --exp-name "$GAN_EXP" --epochs 300 --extreme-percentile 90 --n-per-cluster-ratio 0.3 --local-dir "$GAN_LOCAL_DIR" --force-dataset-upload
    SYNTHETIC_CSV="$DEFAULT_SYNTH"
fi

if [ ! -f "$SYNTHETIC_CSV" ]; then
    echo "ERRO: CSV sintético não encontrado em ${SYNTHETIC_CSV} — abortando."
    exit 1
fi
echo "CSV sintético: ${SYNTHETIC_CSV}"

# ── 1. cluster_lazy — 6 braços ──────────────────────────────────────────────

_maybe_sync_pipeline "[sync] LAZY" "$LAZY_LOCAL_DIR" "$LAZY_OUT_DIR" src/modal/cluster_lazy.py LAZY

_run_step "[2/23] LAZY — Braço: original" "${LAZY_OUT_DIR}/original/results.csv" \
    modal run src/modal/cluster_lazy.py --exp-name original --feature-groups original --ablation-group original --local-dir "$LAZY_LOCAL_DIR" --skip-spatial

_run_step "[3/23] LAZY — Braço: synthetic" "${LAZY_OUT_DIR}/synthetic/results.csv" \
    modal run src/modal/cluster_lazy.py --exp-name synthetic --feature-groups original --synthetic-csv "$SYNTHETIC_CSV" --ablation-group synthetic --local-dir "$LAZY_LOCAL_DIR" --skip-spatial

_run_step "[4/23] LAZY — Braço: newfeatures" "${LAZY_OUT_DIR}/newfeatures/results.csv" \
    modal run src/modal/cluster_lazy.py --exp-name newfeatures --feature-groups original,era5_18z,bt55 --restrict-coverage --ablation-group newfeatures --local-dir "$LAZY_LOCAL_DIR" --skip-spatial

_run_step "[5/23] LAZY — Braço: all" "${LAZY_OUT_DIR}/all/results.csv" \
    modal run src/modal/cluster_lazy.py --exp-name all --feature-groups original,era5_18z,bt55 --synthetic-csv "$SYNTHETIC_CSV" --ablation-group all --local-dir "$LAZY_LOCAL_DIR" --skip-spatial

_run_step "[6/23] LAZY — Braço: basin" "${LAZY_OUT_DIR}/basin/results.csv" \
    modal run src/modal/cluster_lazy.py --exp-name basin --feature-groups original,era5_basin --ablation-group basin --local-dir "$LAZY_LOCAL_DIR" --skip-spatial

_run_step "[7/23] LAZY — Braço: all_basin" "${LAZY_OUT_DIR}/all_basin/results.csv" \
    modal run src/modal/cluster_lazy.py --exp-name all_basin --feature-groups original,era5_18z,bt55,era5_basin --synthetic-csv "$SYNTHETIC_CSV" --ablation-group all_basin --local-dir "$LAZY_LOCAL_DIR" --skip-spatial

echo -e "\n[8/23] LAZY — Agregando resultados..."
python3 scripts/run_ablation_lazy.py --summary-only --output-dir "$LAZY_OUT_DIR"

echo -e "\n[8/23] LAZY — Sincronizando com o dashboard (Parquet)..."
python3 scripts/sync_ablation_to_dashboard.py --pipeline lazy --source-dir "$LAZY_OUT_DIR" || true

# ── 2. cluster_mlp — 6 braços ────────────────────────────────────────────────

_maybe_sync_pipeline "[sync] MLP" "$MLP_LOCAL_DIR" "$MLP_OUT_DIR" src/modal/cluster_mlp.py MLP

_run_step "[9/23] MLP — Braço: original" "${MLP_OUT_DIR}/original/results.csv" \
    modal run src/modal/cluster_mlp.py --exp-name original --feature-groups original --ablation-group original --local-dir "$MLP_LOCAL_DIR" --skip-spatial

_run_step "[10/23] MLP — Braço: synthetic" "${MLP_OUT_DIR}/synthetic/results.csv" \
    modal run src/modal/cluster_mlp.py --exp-name synthetic --feature-groups original --synthetic-csv "$SYNTHETIC_CSV" --ablation-group synthetic --local-dir "$MLP_LOCAL_DIR" --skip-spatial

_run_step "[11/23] MLP — Braço: newfeatures" "${MLP_OUT_DIR}/newfeatures/results.csv" \
    modal run src/modal/cluster_mlp.py --exp-name newfeatures --feature-groups original,era5_18z,bt55 --restrict-coverage --ablation-group newfeatures --local-dir "$MLP_LOCAL_DIR" --skip-spatial

_run_step "[12/23] MLP — Braço: all" "${MLP_OUT_DIR}/all/results.csv" \
    modal run src/modal/cluster_mlp.py --exp-name all --feature-groups original,era5_18z,bt55 --synthetic-csv "$SYNTHETIC_CSV" --ablation-group all --local-dir "$MLP_LOCAL_DIR" --skip-spatial

_run_step "[13/23] MLP — Braço: basin" "${MLP_OUT_DIR}/basin/results.csv" \
    modal run src/modal/cluster_mlp.py --exp-name basin --feature-groups original,era5_basin --ablation-group basin --local-dir "$MLP_LOCAL_DIR" --skip-spatial

_run_step "[14/23] MLP — Braço: all_basin" "${MLP_OUT_DIR}/all_basin/results.csv" \
    modal run src/modal/cluster_mlp.py --exp-name all_basin --feature-groups original,era5_18z,bt55,era5_basin --synthetic-csv "$SYNTHETIC_CSV" --ablation-group all_basin --local-dir "$MLP_LOCAL_DIR" --skip-spatial

echo -e "\n[15/23] MLP — Agregando resultados..."
python3 scripts/run_ablation.py --summary-only --output-dir "$MLP_OUT_DIR"

echo -e "\n[15/23] MLP — Sincronizando com o dashboard (Parquet)..."
python3 scripts/sync_ablation_to_dashboard.py --pipeline mlp --source-dir "$MLP_OUT_DIR" || true

# ── 3. cluster_lstm — 6 braços ───────────────────────────────────────────────

_maybe_sync_pipeline "[sync] LSTM" "$LSTM_LOCAL_DIR" "$LSTM_OUT_DIR_LOCAL" src/modal/cluster_lstm.py LSTM

_run_step "[16/23] LSTM — Braço: original" "${LSTM_OUT_DIR_LOCAL}/original/results.csv" \
    modal run src/modal/cluster_lstm.py --config "$LSTM_CONFIG_NAME" --exp-name original --feature-groups original --ablation-group original --local-dir "$LSTM_LOCAL_DIR" --skip-spatial

_run_step "[17/23] LSTM — Braço: synthetic" "${LSTM_OUT_DIR_LOCAL}/synthetic/results.csv" \
    modal run src/modal/cluster_lstm.py --config "$LSTM_CONFIG_NAME" --exp-name synthetic --feature-groups original --synthetic-csv "$SYNTHETIC_CSV" --ablation-group synthetic --local-dir "$LSTM_LOCAL_DIR" --skip-spatial

_run_step "[18/23] LSTM — Braço: newfeatures" "${LSTM_OUT_DIR_LOCAL}/newfeatures/results.csv" \
    modal run src/modal/cluster_lstm.py --config "$LSTM_CONFIG_NAME" --exp-name newfeatures --feature-groups original,era5_18z,bt55 --restrict-coverage --ablation-group newfeatures --local-dir "$LSTM_LOCAL_DIR" --skip-spatial

_run_step "[19/23] LSTM — Braço: all" "${LSTM_OUT_DIR_LOCAL}/all/results.csv" \
    modal run src/modal/cluster_lstm.py --config "$LSTM_CONFIG_NAME" --exp-name all --feature-groups original,era5_18z,bt55 --synthetic-csv "$SYNTHETIC_CSV" --ablation-group all --local-dir "$LSTM_LOCAL_DIR" --skip-spatial

_run_step "[20/23] LSTM — Braço: basin" "${LSTM_OUT_DIR_LOCAL}/basin/results.csv" \
    modal run src/modal/cluster_lstm.py --config "$LSTM_CONFIG_NAME" --exp-name basin --feature-groups original,era5_basin --ablation-group basin --local-dir "$LSTM_LOCAL_DIR" --skip-spatial

_run_step "[21/23] LSTM — Braço: all_basin" "${LSTM_OUT_DIR_LOCAL}/all_basin/results.csv" \
    modal run src/modal/cluster_lstm.py --config "$LSTM_CONFIG_NAME" --exp-name all_basin --feature-groups original,era5_18z,bt55,era5_basin --synthetic-csv "$SYNTHETIC_CSV" --ablation-group all_basin --local-dir "$LSTM_LOCAL_DIR" --skip-spatial

echo -e "\n[22/23] LSTM — Agregando resultados..."
python3 scripts/run_ablation_lstm.py --config "$LSTM_CONFIG" --summary-only --local-output-dir "$LSTM_OUT_DIR_LOCAL"

echo -e "\n[22/23] LSTM — Sincronizando com o dashboard (Parquet)..."
python3 scripts/sync_ablation_to_dashboard.py --pipeline lstm --source-dir "$LSTM_OUT_DIR_LOCAL" || true

# ── Comparativo cruzado: as 18 combinações (3 pipelines × 6 braços) juntas ──

echo -e "\n[23/23] Comparando as 3 pipelines × 6 braços juntas..."
ALL_TAG=$(date +%Y%m%d_%H%M%S)
python3 scripts/compare_ablation_all.py \
    --lazy-dir "$LAZY_OUT_DIR" --mlp-dir "$MLP_OUT_DIR" --lstm-dir "$LSTM_OUT_DIR_LOCAL" \
    --tag "$ALL_TAG"

# ── Resumo final ─────────────────────────────────────────────────────────

echo -e "\n================================================================="
echo "   Matriz de ablation completa concluída — lazy + mlp + lstm!"
echo "================================================================="
echo "   LAZY: ${LAZY_OUT_DIR}/{original,synthetic,newfeatures,all,basin,all_basin}"
echo "         ${LAZY_OUT_DIR}/comparison/"
echo "   MLP:  ${MLP_OUT_DIR}/{original,synthetic,newfeatures,all,basin,all_basin}"
echo "         ${MLP_OUT_DIR}/comparison/"
echo "   LSTM: ${LSTM_OUT_DIR_LOCAL}/{original,synthetic,newfeatures,all,basin,all_basin}"
echo "         ${LSTM_OUT_DIR_LOCAL}/comparison/"
echo "   TUDO: artifacts/ablation_comparison_all/${ALL_TAG}/  (as 18 combinações juntas)"
echo "================================================================="
