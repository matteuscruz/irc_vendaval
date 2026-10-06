"""Fonte de features por GRUPO — a estrutura nova (2026-09-30).

`dataset/raw/feature_study_cluster_3/` traz quatro parquets, um por grupo do
spec (`features_grupos_1_e_4.md` e as Tabelas 2 e 3 do mesmo PDF):

  grupo 1  vento local, camada limite, tempo          43 colunas horárias
  grupo 2  convecção (cape, cin, mcpr, cisalhamento…)  30 colunas horárias
  grupo 3  sinótico (mslp, w850, vorticidade…)         30 colunas horárias
  grupo 4  geografia e orografia                       1 linha por estação

Os três primeiros vêm em formato LONGO — uma linha por (estação, hora) — já com
os operadores do spec: vizinhança (`r75`/`r250`) e defasagens (`lag1h`,
`max_prev3h`, `std_prev3h`, `delta3h`). Nada disso precisa ser recalculado.

O ALVO continua diário (uma rajada máxima por estação e dia, do INMET), então
algo precisa ligar ~100 features horárias a uma linha por dia. O desenho
adotado é **um ponto por dia, na hora do pico da rajada do ERA5**:

  - a hora é escolhida por `gust10fg` (ERA5) — nunca pelo INMET, então não há
    vazamento do alvo;
  - é também o que um produto de correção faz em produção, já que o ERA5 do dia
    inteiro está disponível;
  - nessa hora `gust10fg` é, por construção, o máximo diário da rajada do ERA5,
    ou seja, a linha de base honesta contra a qual o modelo é medido.

Regras do projeto, mantidas:
  - nenhuma imputação: dia com QUALQUER hora de rajada faltando é descartado
    inteiro; coluna com mais de 1 % de nulos sai (com aviso) em vez de ser
    preenchida — é o caso de `cin` (76 % nulo nas 9 estações);
  - leitura por grupo de linhas do parquet, nunca o arquivo inteiro: a máquina
    local tem ~4 GB livres e os parquets somam 2,3 GB.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xarray as xr

SUBDIR = "feature_study_cluster_3"
GROUP_FILES = {
    "grupo1": "features_grupo1_cluster3.parquet",
    "grupo2": "features_grupo2_cluster3.parquet",
    "grupo3": "features_grupo3_cluster3.parquet",
    "grupo4": "features_grupo4_cluster3.parquet",
}
# O grupo 4 já se chamou `features_grupo4_estatico_cluster3.parquet`; o usuário o
# renomeou (2026-10-02) sem aviso e o estudo passou a falhar com "arquivo não
# encontrado". Os nomes antigos continuam aceitos, na ordem, para que a próxima
# renomeação não derrube uma rodada paga no meio.
LEGACY_NAMES = {
    "grupo4": ("features_grupo4_estatico_cluster3.parquet",),
}
HOURLY_GROUPS = ("grupo1", "grupo2", "grupo3")
STATIC_GROUP = "grupo4"

STATION_SRC, TIME_SRC = "codigo_estacao", "time"
STATION, TIME = "estacao", "time"
PEAK_COLUMN = "gust10fg"
COORDS = ("latitude", "longitude")

# Colunas do grupo 4 que identificam ou auditam a célula, não descrevem relevo.
# `*_n` é a contagem de células no raio: constante por estação e sem significado
# físico, então só serviria de identificador de estação.
STATIC_IDENTIFIERS = ("latitude", "longitude", "grid_lat", "grid_lon", "dist_celula_km")
STATIC_DROP_SUFFIX = ("_n",)

NULL_LIMIT = 0.01
REFERENCE_COLUMN = "era5_gust_max"      # rajada do ERA5: a linha de base honesta
PEAK_HOUR_COLUMN = "hora_pico_utc"
STUDY_END = pd.Timestamp("2024-12-31")  # mesma janela (2000–2024) do estudo anterior
TARGET_FILE = "INMET_Stratified.nc"
TARGET_VAR = "daily_wind_gust_max"
CACHE_NAME = "_daily_peak_cache.parquet"


def group_dir(raw_dir) -> Path:
    return Path(raw_dir) / SUBDIR


def group_path(raw_dir, group: str) -> Path:
    """Caminho do parquet do grupo: o nome atual, ou um legado se for o que existe."""
    base = group_dir(raw_dir)
    for nome in (GROUP_FILES[group], *LEGACY_NAMES.get(group, ())):
        if (base / nome).exists():
            return base / nome
    return base / GROUP_FILES[group]       # inexistente: o erro cita o nome esperado


def is_available(raw_dir) -> bool:
    return all(group_path(raw_dir, g).exists() for g in GROUP_FILES)


def _names(path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def hourly_columns(raw_dir, group: str) -> list[str]:
    """Features de um grupo horário, sem as chaves."""
    return [c for c in _names(group_path(raw_dir, group)) if c not in (STATION_SRC, TIME_SRC)]


def static_columns(raw_dir) -> list[str]:
    """Features de relevo do grupo 4: o ponto e as estatísticas de vizinhança,
    sem identificadores de célula nem contagens."""
    return [
        c for c in _names(group_path(raw_dir, STATIC_GROUP))
        if c != STATION_SRC and c not in STATIC_IDENTIFIERS
        and not c.endswith(STATIC_DROP_SUFFIX)
    ]


def available_stations(raw_dir) -> list[str]:
    t = pq.read_table(group_path(raw_dir, STATIC_GROUP), columns=[STATION_SRC]).to_pandas()
    return sorted(t[STATION_SRC].astype(str))


BATCH_ROWS = 131_072


def _iter_row_groups(path, columns, stations):
    """Lotes pequenos do parquet, já filtrados por estação.

    Duas decisões de memória, ambas medidas. O filtro roda no Arrow, ANTES de
    virar pandas; e a leitura é por lotes de ~131 mil linhas em vez de por grupo
    de linhas (~1 milhão). A primeira versão convertia o grupo de linhas inteiro
    e passava de 2,4 GB de pico — inviável numa máquina com ~4 GB livres.
    """
    pf = pq.ParquetFile(path)
    # o tipo de string vem da própria coluna: `string` ou `large_string`
    # conforme quem escreveu o parquet, e `is_in` exige o mesmo tipo
    tipo = pf.schema_arrow.field(STATION_SRC).type
    valores = pa.array(sorted(set(map(str, stations))), type=tipo)
    for lote in pf.iter_batches(batch_size=BATCH_ROWS, columns=columns):
        tabela = pa.Table.from_batches([lote])
        tabela = tabela.filter(pc.is_in(tabela[STATION_SRC], value_set=valores))
        if tabela.num_rows == 0:
            continue
        df = tabela.to_pandas()
        df[TIME_SRC] = df[TIME_SRC].astype("datetime64[ns]")
        yield df


def peak_keys(raw_dir, stations) -> pd.DataFrame:
    """(estação, hora do pico, dia) — uma linha por estação e dia UTC completo.

    Dia com qualquer uma das 24 horas sem `gust10fg` é descartado inteiro: a
    hora do pico não é identificável, e escolher entre as horas restantes seria
    imputar a informação que falta.
    """
    cols = [STATION_SRC, TIME_SRC, PEAK_COLUMN]
    partes = list(_iter_row_groups(group_path(raw_dir, "grupo1"), cols, stations))
    if not partes:
        raise ValueError(f"nenhuma linha do grupo 1 para as estações {list(stations)}")
    g = pd.concat(partes, ignore_index=True)
    if g.duplicated([STATION_SRC, TIME_SRC]).any():
        raise ValueError("grupo 1 tem (estação, hora) duplicados — a escolha da hora do pico seria ambígua")

    g["dia"] = g[TIME_SRC].dt.floor("D")
    completo = g.groupby([STATION_SRC, "dia"])[PEAK_COLUMN].transform(lambda s: s.notna().sum()) == 24
    g = g[completo]
    idx = g.groupby([STATION_SRC, "dia"])[PEAK_COLUMN].idxmax()
    return g.loc[idx, [STATION_SRC, TIME_SRC, "dia"]].reset_index(drop=True)


def _rows_at(path, columns, keys) -> pd.DataFrame:
    partes = []
    for d in _iter_row_groups(path, columns, set(keys[STATION_SRC])):
        partes.append(d.merge(keys[[STATION_SRC, TIME_SRC]], on=[STATION_SRC, TIME_SRC], how="inner"))
    if not partes:
        raise ValueError(f"{Path(path).name}: nenhuma linha casou com as horas de pico")
    out = pd.concat(partes, ignore_index=True)
    for c in out.columns:
        if c not in (STATION_SRC, TIME_SRC):
            out[c] = out[c].astype("float32")
    return out


def _fingerprint(raw_dir, stations) -> str:
    h = hashlib.sha1()
    for g in GROUP_FILES:
        st = group_path(raw_dir, g).stat()
        h.update(f"{g}|{st.st_size}|{st.st_mtime_ns}".encode())
    h.update(f"{sorted(map(str, stations))}|{PEAK_COLUMN}|{NULL_LIMIT}".encode())
    return h.hexdigest()


def build_daily_table(raw_dir, stations, *, cache_dir=None, use_cache: bool = True):
    """Uma linha por (estação, dia), com as features da hora do pico de rajada.

    Devolve `(tabela, info)`; `info["group_of"]` diz de que grupo veio cada
    coluna, que é o que `arms.py` usa para montar os arms por grupo.
    """
    stations = sorted(map(str, stations))
    fp = _fingerprint(raw_dir, stations)
    cache = Path(cache_dir) / CACHE_NAME if cache_dir else None
    meta = cache.with_suffix(".json") if cache else None
    if use_cache and cache is not None and cache.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info.get("fingerprint") == fp:
            print(f"[groups_source] usando cache {cache.name}", flush=True)
            return pd.read_parquet(cache), info
        print("[groups_source] arquivos ou estações mudaram — refazendo o cache", flush=True)

    keys = peak_keys(raw_dir, stations)
    print(f"[groups_source] {len(keys)} dias completos em {keys[STATION_SRC].nunique()} estações", flush=True)

    daily = keys.copy()
    group_of: dict[str, str] = {}
    for g in HOURLY_GROUPS:
        cols = hourly_columns(raw_dir, g)
        t = _rows_at(group_path(raw_dir, g), [STATION_SRC, TIME_SRC, *cols], keys)
        if len(t) != len(keys):
            raise ValueError(
                f"{g}: {len(t)} linhas nas horas de pico, esperado {len(keys)} — o grupo não "
                "cobre as mesmas horas que o grupo 1"
            )
        daily = daily.merge(t, on=[STATION_SRC, TIME_SRC], how="inner", validate="one_to_one")
        group_of.update({c: g for c in cols})
        del t

    daily = daily.assign(**{
        PEAK_HOUR_COLUMN: daily[TIME_SRC].dt.hour.astype("int8"),
        REFERENCE_COLUMN: daily[PEAK_COLUMN],
    }).drop(columns=[TIME_SRC]).rename(columns={STATION_SRC: STATION, "dia": TIME})

    static_cols = static_columns(raw_dir)
    est = pq.read_table(group_path(raw_dir, STATIC_GROUP),
                        columns=[STATION_SRC, *COORDS, *static_cols]).to_pandas()
    est = est.rename(columns={STATION_SRC: STATION})
    for c in (*COORDS, *static_cols):
        est[c] = est[c].astype("float32")
    daily = daily.merge(est, on=STATION, how="left", validate="many_to_one")
    group_of.update({c: STATIC_GROUP for c in static_cols})
    # o spec põe lat/lon no grupo 1: entram na base junto dele
    group_of.update({c: "grupo1" for c in COORDS})

    nulos = daily[list(group_of)].isna().mean()
    sai = nulos[nulos > NULL_LIMIT]
    if len(sai):
        print("[groups_source] AVISO: colunas com mais de "
              f"{NULL_LIMIT:.0%} de nulos saem (sem imputação): "
              + ", ".join(f"{c} ({v:.0%})" for c, v in sai.items()), flush=True)
    daily = daily.drop(columns=list(sai.index))
    group_of = {c: g for c, g in group_of.items() if c not in sai.index}

    info = {
        "fingerprint": fp, "stations": stations, "n_days": int(len(daily)),
        "group_of": group_of,
        "dropped_null_columns": {c: float(v) for c, v in sai.items()},
        "row_design": "peak_hour", "peak_column": PEAK_COLUMN,
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(cache, index=False)
        meta.write_text(json.dumps(info))
    return daily, info


def load_target(raw_dir, end=STUDY_END):
    """Rajada máxima diária do INMET (longa) e o eixo de dias usado no split.

    O eixo vai só até `end` (2024-12-31): é com ele que
    `default_month_block_split` sorteia os blocos de validação, e com os dias de
    2025 o sorteio muda (verificado: 15 dos 32 blocos trocam). Truncar aqui faz
    o split ser IDÊNTICO ao do estudo anterior.
    """
    with xr.open_dataset(Path(raw_dir) / TARGET_FILE) as ds:
        eixo = pd.DatetimeIndex(ds.time.values)
        s = ds[TARGET_VAR].to_series().rename(TARGET_VAR).reset_index()
    s = s.rename(columns={"estacao": STATION}).dropna(subset=[TARGET_VAR])
    s[STATION] = s[STATION].astype(str)
    s = s[s[TIME] <= end]
    return s, eixo[eixo <= end]
