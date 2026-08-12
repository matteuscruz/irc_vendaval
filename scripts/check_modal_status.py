#!/usr/bin/env python3
"""Checa no volume Modal (irc-vendaval-artifacts) quais experimentos da
matriz de ablation (+ GAN) já existem na nuvem — fonte de verdade remota,
independente do que já foi baixado para esta máquina.

Faz UMA chamada vol.listdir("/", recursive=True) e checa os caminhos
esperados (1 GAN + 4 braços x 3 pipelines) contra o manifesto resultante.

stdout: SOMENTE linhas "NOME_REMOTE_DONE=0|1", uma por variável, nada mais —
        pensado para `eval "$(python3 scripts/check_modal_status.py ...)"`.
stderr: avisos/erros (nunca vai para stdout, não contamina o eval).

Se a chamada ao Modal falhar (auth/rede/volume ainda não existe), todas as
flags saem como 0 (degrada para "nada pronto na nuvem" — mesmo
comportamento de hoje, sem quebrar o restante do script).

Uso:
    eval "$(python3 scripts/check_modal_status.py \\
        --gan-exp gan_v1 --lstm-remote-subdir modal/experiments \\
        --exp-prefix ablation)"
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._ablation_common import ABLATION_MATRIX, exp_name_for

ARM_NAMES = [c["name"] for c in ABLATION_MATRIX]
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"


def _fetch_manifest() -> set[str] | None:
    """Conjunto de todo path de arquivo (relativo, sem barra inicial) no
    volume, ou None se a chamada falhar por qualquer motivo.

    stdout é redirecionado para stderr durante a chamada ao SDK do Modal —
    ele costuma imprimir mensagens de inicialização/log que, se vazassem
    para stdout, quebrariam o `eval` do lado bash.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import modal
            from modal.volume import FileEntryType

            vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
            entries = vol.listdir("/", recursive=True)
            return {
                e.path.lstrip("/") for e in entries if e.type == FileEntryType.FILE
            }
    except Exception as exc:  # noqa: BLE001 — qualquer falha degrada, nunca propaga
        print(
            f"[check_modal_status] AVISO: falha ao consultar o volume "
            f"'{ARTIFACT_VOLUME_NAME}' ({exc.__class__.__name__}: {exc}). "
            f"Assumindo nada pronto na nuvem (equivalente a rodar do zero).",
            file=sys.stderr,
        )
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gan-exp", default="gan_v1")
    parser.add_argument(
        "--lstm-remote-subdir",
        default="modal/experiments",
        help="Prefixo relativo ao volume onde a LSTM salva resultados "
             "(output_dir do YAML sem o prefixo /artifacts/)",
    )
    parser.add_argument("--exp-prefix", default="")
    args = parser.parse_args()

    manifest = _fetch_manifest()  # None => tudo 0

    checks: dict[str, str] = {
        "GAN_REMOTE_DONE": f"gan_clusters/{args.gan_exp}/synthetic_augment.csv",
    }
    for arm in ARM_NAMES:
        arm_upper = arm.upper()
        exp_name = exp_name_for(args.exp_prefix, arm)
        checks[f"LAZY_{arm_upper}_REMOTE_DONE"] = f"lazy_clusters/{exp_name}/results.csv"
        checks[f"MLP_{arm_upper}_REMOTE_DONE"] = f"mlp_clusters/{exp_name}/results.csv"
        checks[f"LSTM_{arm_upper}_REMOTE_DONE"] = (
            f"{args.lstm_remote_subdir.strip('/')}/{exp_name}/results.csv"
        )

    for var_name, rel_path in checks.items():
        present = manifest is not None and rel_path in manifest
        print(f"{var_name}={1 if present else 0}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
