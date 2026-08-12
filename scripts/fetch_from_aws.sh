#!/bin/bash
# Copia um arquivo de uma máquina remota (ex: servidor interno da empresa
# acessível a partir desta instância AWS) para dataset/raw/ deste repositório,
# via scp interativo — login e senha são pedidos pelo próprio SSH na hora;
# este script nunca vê nem guarda a senha.
#
# Depois de copiado, o arquivo é enviado automaticamente pro volume Modal
# 'irc-vendaval-dataset' na próxima vez que qualquer pipeline
# (cluster_lazy/mlp/lstm/gan) rodar — desde que esteja listado em
# DATASET_SENTINELS (src/modal/cluster_*.py) e ainda não exista no volume.
# Funciona com qualquer conta Modal ativa (`modal profile current`), já que
# o upload é condicional à presença do arquivo no volume, não à conta.
#
# Uso:
#   ./scripts/fetch_from_aws.sh <user@host> <caminho/remoto/arquivo.nc> [nome_local.nc]
#   ./scripts/fetch_from_aws.sh                      # modo interativo, pede tudo
#
# Exemplo:
#   ./scripts/fetch_from_aws.sh ubuntu@10.0.1.23 /home/ubuntu/dados/ERA5_Features_Basin_2000_2026.nc

set -e

REMOTE_HOST="$1"
REMOTE_PATH="$2"
LOCAL_NAME="$3"

if [ -z "$REMOTE_HOST" ]; then
    read -rp "Usuário@host remoto (ex: ubuntu@10.0.1.23): " REMOTE_HOST
fi
if [ -z "$REMOTE_PATH" ]; then
    read -rp "Caminho completo do arquivo remoto: " REMOTE_PATH
fi
if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PATH" ]; then
    echo "ERRO: host e caminho remoto são obrigatórios." >&2
    exit 1
fi
if [ -z "$LOCAL_NAME" ]; then
    default_name="$(basename "$REMOTE_PATH")"
    read -rp "Nome local (Enter p/ manter '${default_name}'): " LOCAL_NAME
    LOCAL_NAME="${LOCAL_NAME:-$default_name}"
fi

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dataset/raw"
DEST_PATH="${DEST_DIR}/${LOCAL_NAME}"

mkdir -p "$DEST_DIR"

# Tamanho remoto ANTES de copiar — usado depois pra confirmar que a
# transferência não caiu no meio sem erro visível (scp/rsync às vezes
# retornam 0 mesmo com a conexão interrompida).
echo "Consultando tamanho do arquivo remoto (pode pedir a senha de novo)..."
REMOTE_SIZE="$(ssh "$REMOTE_HOST" "stat -c%s '$REMOTE_PATH' 2>/dev/null || stat -f%z '$REMOTE_PATH'")"
if [ -z "$REMOTE_SIZE" ]; then
    echo "AVISO: não consegui obter o tamanho remoto — seguindo sem verificação de integridade." >&2
fi

echo ""
echo "Copiando ${REMOTE_HOST}:${REMOTE_PATH}"
echo "       -> ${DEST_PATH}"
[ -n "$REMOTE_SIZE" ] && echo "       (tamanho remoto: $(numfmt --to=iec "$REMOTE_SIZE" 2>/dev/null || echo "${REMOTE_SIZE} bytes"))"
echo "(rsync vai pedir login/senha de ${REMOTE_HOST} agora, se autenticação por senha estiver habilitada)"
echo ""
# --partial + retomar: se cair no meio, rodar o script de novo continua de
# onde parou em vez de refazer tudo. -P mostra progresso (arquivo é grande).
rsync -aP --partial "${REMOTE_HOST}:${REMOTE_PATH}" "$DEST_PATH"

if [ ! -s "$DEST_PATH" ]; then
    echo "ERRO: arquivo copiado está vazio ou não existe em ${DEST_PATH}." >&2
    exit 1
fi

LOCAL_SIZE="$(stat -c%s "$DEST_PATH" 2>/dev/null || stat -f%z "$DEST_PATH")"
if [ -n "$REMOTE_SIZE" ] && [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
    echo "" >&2
    echo "ERRO: transferência incompleta — local ${LOCAL_SIZE} bytes != remoto ${REMOTE_SIZE} bytes." >&2
    echo "Rode o script de novo com os mesmos argumentos; rsync --partial retoma de onde parou." >&2
    exit 1
fi

SIZE=$(du -h "$DEST_PATH" | cut -f1)
echo ""
echo "OK — ${DEST_PATH} (${SIZE})$([ -n "$REMOTE_SIZE" ] && echo ", tamanho conferido com o remoto")"
echo ""
echo "Esse arquivo será enviado automaticamente pro volume Modal 'irc-vendaval-dataset'"
echo "na próxima vez que você rodar qualquer pipeline (cluster_lazy/mlp/lstm/gan),"
echo "desde que ainda não exista lá — funciona com qualquer conta Modal ativa."
echo "(Se quiser forçar o re-upload do dataset inteiro nessa próxima run, use --force-dataset-upload.)"
