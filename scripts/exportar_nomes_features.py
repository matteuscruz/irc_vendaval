"""Exporta as colunas selecionadas com o NOME VERDADEIRO de cada variável (ERA5 + operação aplicada) e a unidade.

    uv run python scripts/exportar_nomes_features.py

Gera `config/features_selecionadas_58.txt` e `config/features_selecionadas_sem_anor_49.txt` a partir de
`config/features_maping.json`. As regras das operações são as de `src/feature_study/data/era5_features.py`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# variável base: (nome por extenso, variável ERA5 e paramId, unidade)
BASE = {
    "gust10fg": ("Rajada de vento a 10 m", "ERA5 10fg, paramId 49", "m/s"),
    "gust": ("Rajada de vento a 10 m", "ERA5 10fg, paramId 49", "m/s"),
    "v10": ("Componente meridional (norte-sul) do vento a 10 m", "ERA5 10v, paramId 166", "m/s"),
    "v100": ("Componente meridional (norte-sul) do vento a 100 m", "ERA5 100v, paramId 228247", "m/s"),
    "w10": ("Velocidade do vento a 10 m, raiz(u10² + v10²)", "ERA5 10u/10v, paramId 165/166", "m/s"),
    "w100": ("Velocidade do vento a 100 m, raiz(u100² + v100²)", "ERA5 100u/100v, paramId 228246/228247", "m/s"),
    "cape": ("Energia potencial convectiva disponível (CAPE)", "ERA5 cape, paramId 59", "J/kg"),
    "mcpr": ("Taxa média de precipitação convectiva", "ERA5 mcpr, paramId 235030", "kg m⁻² s⁻¹"),
    "grad_mslp": ("Módulo do gradiente horizontal da pressão ao nível do mar", "ERA5 msl, paramId 151", "hPa/100 km"),
    "anor": ("Ângulo da orografia subgrade (orientação do relevo)", "ERA5 anor, paramId 162, estático", "rad"),
    "hora_solar_sin": ("Seno da hora solar média, sen(2π·h/24), h = (hora UTC + lon/15) mod 24",
                       "calculada (hora + longitude)", "adimensional"),
}
ESTAT = {"median": "mediana", "max": "máximo", "min": "mínimo", "std": "desvio-padrão", "mean": "média",
         "fracpos": "fração das células com valor > 0"}


def descrever(col: str) -> tuple[str, str, str]:
    if col == "grad_mslp_hpa_100km":
        n, f, u = BASE["grad_mslp"]
        return f"{n}, na célula", f, u
    for base in sorted(BASE, key=len, reverse=True):
        if col == base:
            n, f, u = BASE[base]
            return (f"{n}, na célula, na hora do pico" if base not in ("hora_solar_sin",) else n), f, u
        if col.startswith(base + "_"):
            n, f, u = BASE[base]
            op = col[len(base) + 1:]
            break
    else:
        raise KeyError(col)
    if op == "ponto":
        return f"{n}, valor na célula", f, u
    if op == "lag1h":
        return f"{n}, 1 hora antes do pico (t−1)", f, u
    if op == "max_prev3h":
        return f"{n}, máximo nas 3 horas antes do pico (t−3 a t−1)", f, u
    if op == "std_prev3h":
        return f"{n}, desvio-padrão nas 3 horas antes do pico (t−3 a t−1)", f, u
    if op == "delta3h":
        return f"{n}, variação em 3 horas: valor no pico menos valor em t−3", f, u
    if op == "c75_250":
        return f"{n}, contraste: mediana em 75 km menos mediana em 250 km", f, u
    if op == "l75_250":
        return f"{n}, contraste: máximo em 75 km menos mediana em 250 km", f, u
    m = re.fullmatch(r"r(75|250)_(\w+)", op)
    if m:
        raio, est = m.groups()
        un = "fração (0–1)" if est == "fracpos" else u
        return f"{n}, {ESTAT[est]} no raio de {raio} km em torno da célula", f, un
    raise KeyError(col)


def escrever(cols, path: Path, titulo: str) -> None:
    linhas = [f"# {titulo}",
              "# Uma linha por (célula/estação, dia), na hora de maior rajada ERA5 do dia (UTC).",
              "# Raios de 75/250 km: células da grade ERA5 (0,25°) a até essa distância (haversine).",
              "# código\tnome da variável\tunidade\tfonte", ""]
    grupos = json.loads((RAIZ / "config/features_maping.json").read_text())["colunas_por_familia"]
    grupo_de = {c: g for g, fams in grupos.items() for cs in fams.values() for c in cs}
    atual = None
    for c in cols:
        if grupo_de[c] != atual:
            atual = grupo_de[c]
            linhas.append(f"## {atual.replace('grupo', 'Grupo ')}")
        n, f, u = descrever(c)
        linhas.append(f"{c}\t{n}\t{u}\t{f}")
    path.write_text("\n".join(linhas) + "\n")


def main() -> None:
    m = json.loads((RAIZ / "config/features_maping.json").read_text())
    cols = [c for g in m["colunas_por_familia"].values() for f in g.values() for c in f]
    sa = [c for c in cols if not c.startswith("anor")]
    escrever(cols, RAIZ / "config/features_selecionadas_58.txt", "58 colunas selecionadas (arm sel__val12)")
    escrever(sa, RAIZ / "config/features_selecionadas_sem_anor_49.txt",
             "49 colunas sem anor (arm sel__val12_sem_anor) — as do Corrigido ML")
    print(len(cols), len(sa))


if __name__ == "__main__":
    main()
