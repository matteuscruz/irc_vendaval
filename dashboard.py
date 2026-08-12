"""
IRC Vendaval — Dashboard interativo (Plotly Dash)
=================================================
Aba 1 — Explorador MLP:
  mapa de clusters + estações clicáveis, série temporal,
  métricas de qualidade e importância de features.

Aba 2 — Screening LazyPredict:
  ranking de 43 modelos por cluster, justificando a escolha
  do MLPRegressor como modelo principal.

Uso
---
python dashboard.py
# Abrir http://localhost:8050
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback, dcc, html, dash_table

# ── Caminhos ─────────────────────────────────────────────────────────────────

ARTIFACTS = Path(
    os.getenv("ARTIFACTS_DIR", "artifacts/mlp_modal/mlp_clusters")
)
LAZY_ARTIFACTS = Path(
    os.getenv("LAZY_ARTIFACTS_DIR", "artifacts/lazy_modal/lazy_clusters")
)
SHP_PATH = Path(
    os.getenv("SHP_PATH", "dataset/shp/shp_vento.shp")
)

# ── Carregar dados — MLP (multi-experimento) ─────────────────────────────────

gdf_clusters = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
geojson_clusters = json.loads(gdf_clusters.to_json())

METRICS = ["RMSE", "RMSE_P90", "Bias_P90"]
YAXIS_WIND = "Rajada máxima (m/s)"
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]
TOP_N = 15

_MLP_PREDS_COLS = [
    "estacao", "time", "latitude", "longitude", "cluster_id",
    "y_true", "y_pred", "era5_wind_mag_max", "ratio_pred", "ratio_true",
]
_MLP_TABLE_LABELS = {
    "cluster_id": "Cluster", "n_stations": "Est.",
    "n_train": "N treino", "n_val": "N val",
    "MLP_R2": "MLP R²", "MLP_RMSE": "MLP RMSE",
    "MLP_Bias_P90": "MLP Bias@P90",
    "ERA5_R2": "ERA5 R²", "ERA5_RMSE": "ERA5 RMSE",
    "ERA5_Bias_P90": "ERA5 Bias@P90",
}

# Estado MLP — preenchido por _load_mlp_experiment()
_IMP_COLS = ["feature", "importance", "std", "cluster_id"]
_STATION_COLS = ["estacao", "latitude", "longitude", "cluster_id"]

results_df = pd.DataFrame()
importance_df = pd.DataFrame(columns=_IMP_COLS)
preds_df = pd.DataFrame(columns=_MLP_PREDS_COLS)
stations_df = pd.DataFrame(columns=_STATION_COLS)
CLUSTER_IDS: list = []
CLUSTER_COLORS: dict = {}
CLUSTER_MEMBERS: dict = {}
POLY_TO_CLUSTER: dict = {}
mlp_table_data: list = []
mlp_table_cols: list = []


def _cluster_members(cid) -> list[int]:
    """Polígonos base de um cluster_id ('1-2-3' -> [1,2,3]; 4 -> [4])."""
    s = str(cid)
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip().isdigit()]
    try:
        return [int(float(s))]
    except ValueError:
        return []


def _build_mlp_table() -> None:
    """(Re)constrói data/columns da tabela MLP a partir de results_df."""
    global mlp_table_data, mlp_table_cols
    if results_df.empty:
        mlp_table_data, mlp_table_cols = [], []
        return
    has_counts = "n_train" in results_df.columns
    cols = [
        "cluster_id", "n_stations",
        *(["n_train", "n_val"] if has_counts else []),
        "MLP_R2", "MLP_RMSE", "MLP_Bias_P90",
        "ERA5_R2", "ERA5_RMSE", "ERA5_Bias_P90",
    ]
    cols = [c for c in cols if c in results_df.columns]
    round_cols = [
        c for c in cols
        if c not in ("cluster_id", "n_stations", "n_train", "n_val")
    ]
    mlp_table_data = (
        results_df[cols].copy()
        .assign(**{c: results_df[c].round(3) for c in round_cols})
        .to_dict("records")
    )
    mlp_table_cols = [
        {"name": _MLP_TABLE_LABELS.get(c, c), "id": c} for c in cols
    ]


def _load_mlp_experiment(exp_dir: Path) -> None:
    """Lê os CSVs de um experimento MLP e popula o estado global."""
    global results_df, importance_df, preds_df, stations_df
    global CLUSTER_IDS, CLUSTER_COLORS, CLUSTER_MEMBERS, POLY_TO_CLUSTER

    # cluster_mlp.py grava esses CSVs em subpastas via ArtifactManager, não
    # na raiz do experimento (_partial/csv/ para results/importance/stations,
    # predictions/ para as previsões).
    res_path = exp_dir / "_partial" / "csv" / "mlp_cluster_results.csv"
    results_df = (
        pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    )

    imp_path = exp_dir / "_partial" / "csv" / "feature_importance.csv"
    importance_df = (
        pd.read_csv(imp_path) if imp_path.exists()
        else pd.DataFrame(columns=_IMP_COLS)
    )

    preds_path = exp_dir / "predictions" / "predictions_by_station.csv"
    preds_df = (
        pd.read_csv(preds_path, parse_dates=["time"])
        if preds_path.exists() else pd.DataFrame(columns=_MLP_PREDS_COLS)
    )

    st_path = exp_dir / "_partial" / "csv" / "stations_metadata.csv"
    stations_df = (
        pd.read_csv(st_path) if st_path.exists()
        else pd.DataFrame(columns=_STATION_COLS)
    )

    CLUSTER_IDS = (
        sorted(results_df["cluster_id"].tolist())
        if not results_df.empty else []
    )
    CLUSTER_COLORS = {
        cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(CLUSTER_IDS)
    }
    CLUSTER_MEMBERS = {cid: _cluster_members(cid) for cid in CLUSTER_IDS}
    POLY_TO_CLUSTER = {
        m: cid for cid, members in CLUSTER_MEMBERS.items() for m in members
    }
    _build_mlp_table()


def _discover_mlp_experiments() -> list[dict]:
    """Varre ARTIFACTS/* (fallback: ARTIFACTS plano como '(raiz)').

    Não filtra por "exp*" — experimentos de ablation (ablation_original,
    ablation_synthetic, ...) usam outro padrão de nome e ficariam
    invisíveis com um glob mais restrito.
    """
    exps: list[dict] = []
    for d in ARTIFACTS.glob("*"):
        if not d.is_dir() or not (d / "_partial" / "csv" / "mlp_cluster_results.csv").exists():
            continue
        merge = None
        ablation_group = None
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                meta_data = json.loads(meta.read_text())
                merge = meta_data.get("cluster_merge")
                ablation_group = meta_data.get("ablation_group")
            except (json.JSONDecodeError, OSError):
                merge = None
        suffix = f" — merge {merge}" if merge else " — baseline"
        exps.append({
            "id": d.name, "dir": str(d), "label": d.name + suffix,
            "ablation_group": ablation_group,
        })

    def _exp_num(e: dict) -> int:
        m = re.search(r"\d+", e["id"])
        return int(m.group()) if m else 0

    exps.sort(key=lambda e: (_exp_num(e), e["id"]))

    if not exps and (ARTIFACTS / "_partial" / "csv" / "mlp_cluster_results.csv").exists():
        exps.append({
            "id": "(raiz)", "dir": str(ARTIFACTS), "label": "(raiz)",
            "ablation_group": None,
        })
    return exps


MLP_EXPERIMENTS = _discover_mlp_experiments()
_mlp_exp_by_id = {e["id"]: e for e in MLP_EXPERIMENTS}
_mlp_default_exp = MLP_EXPERIMENTS[0]["id"] if MLP_EXPERIMENTS else None
_load_mlp_experiment(
    Path(_mlp_exp_by_id[_mlp_default_exp]["dir"])
    if _mlp_default_exp else ARTIFACTS
)

# ── Carregar dados — LazyPredict (multi-experimento) ─────────────────────────

_LAZY_EMPTY_COLS = [
    "cluster_id", "n_stations", "Model", "R2", "Adj_R2", "RMSE", "Time",
]


def _load_lazy_df(results_csv: Path) -> pd.DataFrame:
    """Lê e normaliza um lazy_cluster_results.csv de um experimento."""
    if not results_csv.exists():
        return pd.DataFrame(columns=_LAZY_EMPTY_COLS)
    df = pd.read_csv(results_csv).rename(columns={
        "R-Squared": "R2",
        "Adjusted R-Squared": "Adj_R2",
        "Time Taken": "Time",
    })
    for c in ("R2", "Adj_R2", "RMSE"):
        if c in df.columns:
            df[c] = df[c].round(4)
    if "Time" in df.columns:
        df["Time"] = df["Time"].round(3)
    return df


def _discover_lazy_experiments() -> list[dict]:
    """
    Varre LAZY_ARTIFACTS/* e monta a lista de experimentos disponíveis.

    Cada item: {id, path, label, merge, ablation_group}. Ordenado por número
    do exp. Não filtra por "exp*" — experimentos de ablation (ablation_original,
    ...) usam outro padrão de nome. Fallback: se não houver subpastas com
    resultado, usa o CSV plano na raiz.
    """
    exps: list[dict] = []
    for d in LAZY_ARTIFACTS.glob("*"):
        csv = d / "lazy_cluster_results.csv"
        if not d.is_dir() or not csv.exists():
            continue
        merge = None
        ablation_group = None
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                meta_data = json.loads(meta.read_text())
                merge = meta_data.get("cluster_merge")
                ablation_group = meta_data.get("ablation_group")
            except (json.JSONDecodeError, OSError):
                merge = None
        suffix = f" — merge {merge}" if merge else " — baseline"
        exps.append({
            "id": d.name, "path": str(csv),
            "label": d.name + suffix, "merge": merge,
            "ablation_group": ablation_group,
        })

    def _exp_num(e: dict) -> int:
        m = re.search(r"\d+", e["id"])
        return int(m.group()) if m else 0

    exps.sort(key=lambda e: (_exp_num(e), e["id"]))

    if not exps:
        root = LAZY_ARTIFACTS / "lazy_cluster_results.csv"
        if root.exists():
            exps.append({
                "id": "(raiz)", "path": str(root),
                "label": "(raiz)", "merge": None, "ablation_group": None,
            })
    return exps


def _cluster_ids(df: pd.DataFrame) -> list:
    return sorted(df["cluster_id"].unique().tolist())


LAZY_EXPERIMENTS = _discover_lazy_experiments()
_lazy_exp_by_id = {e["id"]: e for e in LAZY_EXPERIMENTS}
_lazy_default_exp = LAZY_EXPERIMENTS[0]["id"] if LAZY_EXPERIMENTS else None

lazy_df = (
    _load_lazy_df(Path(_lazy_exp_by_id[_lazy_default_exp]["path"]))
    if _lazy_default_exp
    else pd.DataFrame(columns=_LAZY_EMPTY_COLS)
)

LAZY_CLUSTER_IDS = _cluster_ids(lazy_df)

# ── Constantes ───────────────────────────────────────────────────────────────

# ── Helpers — MLP ────────────────────────────────────────────────────────────

def _cluster_color(cid) -> str:
    return CLUSTER_COLORS.get(cid, PALETTE[0])


def _build_map(selected_station: str | None = None) -> go.Figure:
    fig = go.Figure()

    # Expande cada cluster (possivelmente agregado) nos polígonos base
    # para casar com featureidkey "properties.cluster" ("01".."14").
    locations: list[str] = []
    zvals: list[int] = []
    texts: list[str] = []
    for i, r in enumerate(results_df.itertuples()):
        htext = (
            f"Cluster {r.cluster_id}<br>Estações: {r.n_stations}"
            f"<br>R²: {r.MLP_R2:.3f}<br>RMSE: {r.MLP_RMSE:.3f}"
        )
        for m in CLUSTER_MEMBERS.get(r.cluster_id, []):
            locations.append(f"{m:02d}")
            zvals.append(i)
            texts.append(htext)

    if locations:
        fig.add_trace(go.Choroplethmap(
            geojson=geojson_clusters,
            featureidkey="properties.cluster",
            locations=locations,
            z=zvals,
            colorscale="Viridis",
            zmin=0,
            zmax=max(len(CLUSTER_IDS) - 1, 1),
            marker_opacity=0.30,
            marker_line_width=1.2,
            marker_line_color="white",
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            showscale=False,
            name="Clusters",
        ))

    if not stations_df.empty:
        colors = [
            _cluster_color(cid) for cid in stations_df["cluster_id"]
        ]
        sizes = [
            16 if selected_station == r.estacao else 10
            for r in stations_df.itertuples()
        ]
        symbols = [
            "star" if selected_station == r.estacao else "circle"
            for r in stations_df.itertuples()
        ]
        fig.add_trace(go.Scattermap(
            lat=stations_df["latitude"],
            lon=stations_df["longitude"],
            mode="markers",
            marker={"size": sizes, "color": colors, "symbol": symbols},
            customdata=stations_df[["estacao", "cluster_id"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Lat: %{lat:.2f}°  Lon: %{lon:.2f}°"
                "<extra></extra>"
            ),
            name="Estações",
        ))

    fig.update_layout(
        map={
            "style": "carto-positron",
            "center": {"lat": -28.5, "lon": -52.5},
            "zoom": 5,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        uirevision="map",
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
    )
    return fig


def _build_timeseries(estacao: str | None) -> go.Figure:
    if estacao is None or preds_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Clique numa estação no mapa",
            xaxis_title="Data",
            yaxis_title=YAXIS_WIND,
            template="plotly_white",
        )
        return fig

    df_st = preds_df[preds_df["estacao"] == estacao].sort_values("time")
    cid = df_st["cluster_id"].iloc[0] if len(df_st) else "?"
    r2 = df_st["y_true"].corr(df_st["y_pred"]) ** 2
    rmse = ((df_st["y_true"] - df_st["y_pred"]) ** 2).mean() ** 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["y_true"],
        name="INMET observado",
        line={"color": "royalblue", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["y_pred"],
        name="MLP predito",
        line={"color": "darkorange", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["era5_wind_mag_max"],
        name="ERA5 raw",
        line={"color": "tomato", "width": 1, "dash": "dot"},
    ))
    fig.update_layout(
        title=(
            f"Estação {estacao} — Cluster {cid}"
            f"   |   R²={r2:.3f}  RMSE={rmse:.2f} m/s"
        ),
        xaxis_title="Data",
        yaxis_title=YAXIS_WIND,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
        margin={"t": 50, "b": 60},
    )
    return fig


def _build_metrics_bar(metric: str) -> go.Figure:
    clusters = [f"C{c}" for c in results_df["cluster_id"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MLP",
        x=clusters,
        y=results_df[f"MLP_{metric}"],
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        name="ERA5",
        x=clusters,
        y=results_df[f"ERA5_{metric}"],
        marker_color="tomato",
    ))
    fig.update_layout(
        barmode="group",
        title=f"MLP vs ERA5 — {metric}",
        yaxis_title=metric,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 50, "b": 60},
    )
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    return fig


def _build_importance(cluster_id: int | None) -> go.Figure:
    if cluster_id is None:
        cluster_id = CLUSTER_IDS[0]
    df_imp = (
        importance_df[importance_df["cluster_id"] == cluster_id]
        .sort_values("importance", ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=df_imp["importance"],
        y=df_imp["feature"],
        orientation="h",
        error_x={"type": "data", "array": df_imp["std"], "visible": True},
        marker_color="steelblue",
    ))
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    fig.update_layout(
        title=f"Importância por Permutação — Cluster {cluster_id}",
        xaxis_title="Queda em R²",
        template="plotly_white",
        margin={"t": 50, "l": 160},
    )
    return fig


# ── Helpers — Distribuição ───────────────────────────────────────────────────

def _build_distribution(selected_cluster: int | None = None) -> go.Figure:
    if preds_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Rode a pipeline para gerar predictions_by_station.csv",
            template="plotly_white",
        )
        return fig

    fig = go.Figure()
    for cid in CLUSTER_IDS:
        df_c = preds_df[preds_df["cluster_id"] == cid]
        opacity = 1.0 if (
            selected_cluster is None or selected_cluster == cid
        ) else 0.20
        label = f"C{cid}"

        fig.add_trace(go.Violin(
            x=[label] * len(df_c),
            y=df_c["y_true"],
            name="INMET" if cid == CLUSTER_IDS[0] else None,
            legendgroup="INMET",
            showlegend=cid == CLUSTER_IDS[0],
            side="negative",
            line_color="royalblue",
            fillcolor="rgba(65,105,225,0.35)",
            opacity=opacity,
            meanline_visible=True,
            hovertemplate=(
                f"Cluster {cid} — INMET<br>"
                "Valor: %{y:.2f} m/s<extra></extra>"
            ),
        ))
        fig.add_trace(go.Violin(
            x=[label] * len(df_c),
            y=df_c["y_pred"],
            name="MLP" if cid == CLUSTER_IDS[0] else None,
            legendgroup="MLP",
            showlegend=cid == CLUSTER_IDS[0],
            side="positive",
            line_color="darkorange",
            fillcolor="rgba(255,140,0,0.35)",
            opacity=opacity,
            meanline_visible=True,
            hovertemplate=(
                f"Cluster {cid} — MLP<br>"
                "Valor: %{y:.2f} m/s<extra></extra>"
            ),
        ))

    p90_global = preds_df["y_true"].quantile(0.9)
    fig.add_hline(
        y=p90_global,
        line_dash="dot",
        line_color="gray",
        annotation_text="P90 global",
        annotation_position="top right",
    )
    fig.update_layout(
        title=(
            "Distribuição por Cluster — INMET observado (esq.) "
            "vs MLP predito (dir.)"
        ),
        yaxis_title=YAXIS_WIND,
        xaxis_title="Cluster",
        violingap=0.1,
        violinmode="overlay",
        template="plotly_white",
        legend={
            "orientation": "h",
            "y": -0.15,
            "title": {"text": "Série:"},
        },
        margin={"t": 55, "b": 70},
    )
    return fig


def _build_scatter_regression(
    selected_cluster: int | None = None,
) -> go.Figure:
    if preds_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Rode a pipeline para gerar predictions_by_station.csv",
            template="plotly_white",
        )
        return fig

    fig = go.Figure()

    # Limites globais para linhas de referência
    global_min = preds_df[["y_true", "y_pred"]].min().min()
    global_max = preds_df[["y_true", "y_pred"]].max().max()
    ref = [global_min, global_max]

    # Linha 1:1 (perfeição)
    fig.add_trace(go.Scatter(
        x=ref, y=ref,
        mode="lines",
        name="1:1",
        line={"color": "black", "dash": "dash", "width": 1.2},
        showlegend=True,
    ))

    for cid in CLUSTER_IDS:
        df_c = preds_df[preds_df["cluster_id"] == cid].dropna(
            subset=["y_true", "y_pred"]
        )
        if df_c.empty:
            continue

        color = _cluster_color(cid)
        opacity = 1.0 if (
            selected_cluster is None or selected_cluster == cid
        ) else 0.12
        label = f"C{cid}"

        # Pontos
        fig.add_trace(go.Scatter(
            x=df_c["y_true"],
            y=df_c["y_pred"],
            mode="markers",
            name=label,
            legendgroup=label,
            marker={
                "color": color,
                "size": 5,
                "opacity": opacity,
            },
            hovertemplate=(
                f"Cluster {cid}<br>"
                "Obs: %{x:.2f} m/s<br>"
                "Pred: %{y:.2f} m/s"
                "<extra></extra>"
            ),
        ))

        # Regressão OLS (numpy polyfit grau 1)
        coeffs = np.polyfit(df_c["y_true"], df_c["y_pred"], 1)
        x_line = np.array([df_c["y_true"].min(), df_c["y_true"].max()])
        y_line = np.polyval(coeffs, x_line)
        r2_c = float(results_df.loc[
            results_df["cluster_id"] == cid, "MLP_R2"
        ].iloc[0])
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"{label} reg (R²={r2_c:.2f})",
            legendgroup=label,
            opacity=opacity,
            line={"color": color, "width": 2},
            hovertemplate=(
                f"Cluster {cid} — regressão<br>"
                f"a={coeffs[0]:.2f}  b={coeffs[1]:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Observado × Predito por Cluster (linha = regressão OLS)",
        xaxis_title=f"Observado — {YAXIS_WIND}",
        yaxis_title=f"Predito — {YAXIS_WIND}",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.20, "font": {"size": 11}},
        margin={"t": 55, "b": 80},
    )
    return fig


# ── Helpers — LazyPredict ────────────────────────────────────────────────────

def _build_lazy_bar(df: pd.DataFrame, cluster_id) -> go.Figure:
    df_c = (
        df[df["cluster_id"] == cluster_id]
        .sort_values("R2", ascending=False)
        .head(TOP_N)
        .sort_values("R2", ascending=True)   # barh: menor embaixo
    )
    is_mlp = df_c["Model"] == "MLPRegressor"
    colors = [
        "#ff7f0e" if m else "#1f77b4"
        for m in is_mlp
    ]
    fig = go.Figure(go.Bar(
        x=df_c["R2"],
        y=df_c["Model"],
        orientation="h",
        marker_color=colors,
        text=df_c["R2"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=(
            f"Top {TOP_N} modelos — Cluster {cluster_id}"
            " (laranja = MLPRegressor)"
        ),
        xaxis_title="R²",
        template="plotly_white",
        margin={"t": 55, "l": 220, "r": 60},
        xaxis={"range": [
            max(0, df_c["R2"].min() - 0.05),
            min(1, df_c["R2"].max() + 0.10),
        ]},
    )
    return fig


def _lazy_table_data(df: pd.DataFrame, cluster_id) -> tuple[list, list]:
    df_c = (
        df[df["cluster_id"] == cluster_id]
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )
    df_c.insert(0, "Rank", range(1, len(df_c) + 1))
    cols = ["Rank", "Model", "R2", "Adj_R2", "RMSE", "Time"]
    labels = {
        "Rank": "#",
        "Model": "Modelo",
        "R2": "R²",
        "Adj_R2": "R² Aj.",
        "RMSE": "RMSE",
        "Time": "Tempo (s)",
    }
    columns = [{"name": labels[c], "id": c} for c in cols]
    data = df_c[cols].to_dict("records")

    # Highlight MLPRegressor
    cond = [{
        "if": {"filter_query": '{Model} = "MLPRegressor"'},
        "backgroundColor": "#fff3cd",
        "fontWeight": "bold",
    }]
    return data, columns, cond


# ── Banner de split temporal (reativo ao experimento) ────────────────────────

TRAIN_SLICE = ("2008-01-01", "2018-12-31")
VAL_SLICE = ("2019-01-01", "2019-12-31")


def _split_banner() -> dbc.Alert:
    if "n_train" in results_df.columns and not results_df.empty:
        total_train = int(results_df["n_train"].sum())
        total_val = int(results_df["n_val"].sum())
        count_txt = (
            f"Treino: {total_train:,} amostras   |   "
            f"Validação: {total_val:,} amostras"
        )
    else:
        count_txt = "Execute a pipeline para ver contagem de amostras."

    return dbc.Alert(
        [
            html.Strong("Split temporal:  "),
            f"Treino {TRAIN_SLICE[0]} → {TRAIN_SLICE[1]}   |   "
            f"Validação {VAL_SLICE[0]} → {VAL_SLICE[1]}",
            html.Span("   |   ", className="text-muted"),
            count_txt,
        ],
        color="light",
        className="py-2 px-3 mb-0 border",
        style={"fontSize": "13px"},
    )

# ── App ──────────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="IRC Vendaval",
)

_CARD = {"height": "100%", "padding": "8px"}
_CELL = {"textAlign": "center", "fontSize": "12px", "padding": "4px"}
_CELL_LEFT = {
    "textAlign": "left", "fontSize": "12px", "padding": "4px",
}

# ── Conteúdo das abas ────────────────────────────────────────────────────────

_tab_mlp = dbc.Container(fluid=True, children=[
    dcc.Store(id="selected-station"),
    dcc.Store(id="selected-cluster"),
    dcc.Store(id="mlp-exp"),

    # Seletor de grupo de ablation + experimento
    dbc.Row([
        dbc.Col(
            html.Span("Grupo de ablation:", className="text-muted me-2"),
            width="auto", className="d-flex align-items-center",
        ),
        dbc.Col(
            dcc.Dropdown(
                id="mlp-ablation-group-dropdown",
                options=[{"label": "Todos", "value": "__all__"}] + [
                    {"label": g, "value": g}
                    for g in sorted({
                        e["ablation_group"] for e in MLP_EXPERIMENTS
                        if e.get("ablation_group")
                    })
                ],
                value="__all__",
                clearable=False,
                style={"width": "220px"},
            ),
            width="auto",
        ),
        dbc.Col(
            html.Span("Experimento:", className="text-muted me-2"),
            width="auto", className="d-flex align-items-center",
        ),
        dbc.Col(
            dcc.Dropdown(
                id="mlp-exp-dropdown",
                options=[
                    {"label": e["label"], "value": e["id"]}
                    for e in MLP_EXPERIMENTS
                ],
                value=_mlp_default_exp,
                clearable=False,
                style={"width": "360px"},
            ),
            width="auto",
        ),
    ], className="mt-2 mb-1"),

    # Banner de split temporal
    dbc.Row(
        dbc.Col(html.Div(_split_banner(), id="mlp-split-banner")),
        className="mb-1",
    ),

    # Linha 1: Mapa + Série temporal
    dbc.Row([
        dbc.Col(dbc.Card(
            dcc.Graph(
                id="map-graph",
                figure=_build_map(),
                style={"height": "420px"},
                config={"scrollZoom": True},
            ),
            style=_CARD,
        ), width=6),
        dbc.Col(dbc.Card(
            dcc.Graph(
                id="timeseries-graph",
                figure=_build_timeseries(None),
                style={"height": "420px"},
            ),
            style=_CARD,
        ), width=6),
    ], className="mb-2 mt-2"),

    # Linha 2: Distribuição + Scatter regressão
    dbc.Row([
        dbc.Col(dbc.Card(
            dcc.Graph(
                id="distribution-graph",
                figure=_build_distribution(),
                style={"height": "420px"},
            ),
            style=_CARD,
        ), width=6),
        dbc.Col(dbc.Card(
            dcc.Graph(
                id="scatter-graph",
                figure=_build_scatter_regression(),
                style={"height": "420px"},
            ),
            style=_CARD,
        ), width=6),
    ], className="mb-2"),

    # Linha 3: Métricas + Importância
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.Row([dbc.Col(
                dcc.Dropdown(
                    id="metric-dropdown",
                    options=[{"label": m, "value": m} for m in METRICS],
                    value="RMSE",
                    clearable=False,
                    style={"width": "180px"},
                ),
                width="auto",
            )], className="mb-1 mt-1 ms-1"),
            dash_table.DataTable(
                id="metrics-table",
                columns=mlp_table_cols,
                data=mlp_table_data,
                style_table={"overflowX": "auto"},
                style_header={"fontWeight": "bold", "fontSize": "12px"},
                style_cell=_CELL,
                style_data_conditional=[],
                row_selectable="single",
                selected_rows=[],
            ),
            dcc.Graph(
                id="metrics-bar",
                figure=_build_metrics_bar("RMSE"),
                style={"height": "260px"},
            ),
        ], style=_CARD), width=6),
        dbc.Col(dbc.Card(
            dcc.Graph(
                id="importance-graph",
                figure=_build_importance(
                    CLUSTER_IDS[0] if CLUSTER_IDS else None
                ),
                style={"height": "480px"},
            ),
            style=_CARD,
        ), width=6),
    ], className="mb-2"),
], className="px-0")

_lazy_init_cid = LAZY_CLUSTER_IDS[0] if LAZY_CLUSTER_IDS else 1
_lazy_init_data, _lazy_init_cols, _lazy_init_cond = _lazy_table_data(
    lazy_df, _lazy_init_cid
)

_tab_lazy = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.P(
        "Screening de 43 modelos via LazyPredict por cluster espacial "
        "(treino 2000–2022 / validação 2023). "
        "O MLPRegressor (destacado) lidera em 3 de 6 clusters e aparece "
        "no top-3 em outros 2, justificando a escolha do MLP.",
        className="text-muted mt-2 mb-1",
    ))),
    dbc.Row([
        dbc.Col(
            html.Span("Grupo de ablation:", className="text-muted me-2"),
            width="auto", className="d-flex align-items-center",
        ),
        dbc.Col(
            dcc.Dropdown(
                id="lazy-ablation-group-dropdown",
                options=[{"label": "Todos", "value": "__all__"}] + [
                    {"label": g, "value": g}
                    for g in sorted({
                        e["ablation_group"] for e in LAZY_EXPERIMENTS
                        if e.get("ablation_group")
                    })
                ],
                value="__all__",
                clearable=False,
                style={"width": "220px"},
            ),
            width="auto",
        ),
        dbc.Col(
            html.Span("Experimento:", className="text-muted me-2"),
            width="auto", className="d-flex align-items-center",
        ),
        dbc.Col(
            dcc.Dropdown(
                id="lazy-exp-dropdown",
                options=[
                    {"label": e["label"], "value": e["id"]}
                    for e in LAZY_EXPERIMENTS
                ],
                value=_lazy_default_exp,
                clearable=False,
                style={"width": "360px"},
            ),
            width="auto",
        ),
        dbc.Col(
            dcc.Dropdown(
                id="lazy-cluster-dropdown",
                options=[
                    {"label": f"Cluster {c}", "value": c}
                    for c in LAZY_CLUSTER_IDS
                ],
                value=_lazy_init_cid,
                clearable=False,
                style={"width": "200px"},
            ),
            width="auto",
        ),
    ], className="mb-2 mt-1"),
    dbc.Row([
        dbc.Col(dbc.Card(
            dcc.Graph(
                id="lazy-bar",
                figure=_build_lazy_bar(lazy_df, _lazy_init_cid),
                style={"height": "520px"},
            ),
            style=_CARD,
        ), width=6),
        dbc.Col(dbc.Card([
            dash_table.DataTable(
                id="lazy-table",
                columns=_lazy_init_cols,
                data=_lazy_init_data,
                style_table={"overflowY": "auto", "maxHeight": "500px"},
                style_header={
                    "fontWeight": "bold", "fontSize": "12px",
                },
                style_cell=_CELL,
                style_cell_conditional=[{
                    "if": {"column_id": "Model"},
                    **_CELL_LEFT,
                }],
                style_data_conditional=_lazy_init_cond,
                page_action="none",
                sort_action="native",
            ),
        ], style=_CARD), width=6),
    ], className="mb-2"),
], className="px-0")

# ── Layout principal ─────────────────────────────────────────────────────────

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H4(
        "IRC Vendaval — Correção de Viés de Rajadas de Vento Extremo",
        className="text-center my-2",
    ))),
    dbc.Tabs([
        dbc.Tab(_tab_mlp, label="Explorador MLP", tab_id="tab-mlp"),
        dbc.Tab(_tab_lazy, label="Screening LazyPredict", tab_id="tab-lazy"),
    ], id="main-tabs", active_tab="tab-mlp"),
])

# ── Callbacks — MLP ──────────────────────────────────────────────────────────


@callback(
    Output("distribution-graph", "figure"),
    Input("selected-cluster", "data"),
    Input("mlp-exp", "data"),
)
def _update_distribution(cluster_id, _exp):
    return _build_distribution(cluster_id)


@callback(
    Output("scatter-graph", "figure"),
    Input("selected-cluster", "data"),
    Input("mlp-exp", "data"),
)
def _update_scatter(cluster_id, _exp):
    return _build_scatter_regression(cluster_id)


@callback(
    Output("selected-station", "data"),
    Output("selected-cluster", "data"),
    Input("map-graph", "clickData"),
    Input("metrics-table", "selected_rows"),
)
def _update_selection(click_data, selected_rows):
    station = None
    cluster = CLUSTER_IDS[0] if CLUSTER_IDS else None

    if click_data:
        pt = click_data["points"][0]
        cd = pt.get("customdata")
        if cd is not None:
            station = cd[0]
            cluster = cd[1]   # rótulo do cluster (pode ser "1-2-3")
        else:
            loc = pt.get("location")
            if loc is not None:
                cluster = POLY_TO_CLUSTER.get(int(loc), cluster)

    if selected_rows:
        cluster = results_df.iloc[selected_rows[0]]["cluster_id"]

    return station, cluster


@callback(
    Output("map-graph", "figure"),
    Input("selected-station", "data"),
    Input("mlp-exp", "data"),
)
def _update_map(station, _exp):
    return _build_map(station)


@callback(
    Output("timeseries-graph", "figure"),
    Input("selected-station", "data"),
    Input("mlp-exp", "data"),
)
def _update_timeseries(station, _exp):
    return _build_timeseries(station)


@callback(
    Output("metrics-bar", "figure"),
    Input("metric-dropdown", "value"),
    Input("mlp-exp", "data"),
)
def _update_metrics_bar(metric, _exp):
    return _build_metrics_bar(metric)


@callback(
    Output("importance-graph", "figure"),
    Input("selected-cluster", "data"),
    Input("mlp-exp", "data"),
)
def _update_importance(cluster_id, _exp):
    return _build_importance(cluster_id)


@callback(
    Output("metrics-table", "style_data_conditional"),
    Input("selected-cluster", "data"),
)
def _highlight_mlp_row(cluster_id):
    if cluster_id is None or results_df.empty:
        return []
    ids = results_df["cluster_id"].tolist()
    if cluster_id not in ids:
        return []
    return [{
        "if": {"row_index": ids.index(cluster_id)},
        "backgroundColor": "#d0e8ff",
        "fontWeight": "bold",
    }]


@callback(
    Output("mlp-exp-dropdown", "options"),
    Output("mlp-exp-dropdown", "value"),
    Input("mlp-ablation-group-dropdown", "value"),
    prevent_initial_call=True,
)
def _update_mlp_experiments(ablation_group):
    filtered = MLP_EXPERIMENTS if ablation_group in (None, "__all__") else [
        e for e in MLP_EXPERIMENTS if e.get("ablation_group") == ablation_group
    ]
    options = [{"label": e["label"], "value": e["id"]} for e in filtered]
    value = filtered[0]["id"] if filtered else None
    return options, value


@callback(
    Output("mlp-exp", "data"),
    Output("selected-station", "data", allow_duplicate=True),
    Output("selected-cluster", "data", allow_duplicate=True),
    Output("metrics-table", "data"),
    Output("metrics-table", "columns"),
    Output("metrics-table", "selected_rows"),
    Output("mlp-split-banner", "children"),
    Input("mlp-exp-dropdown", "value"),
    prevent_initial_call=True,
)
def _switch_mlp_experiment(exp_id):
    if exp_id in _mlp_exp_by_id:
        _load_mlp_experiment(Path(_mlp_exp_by_id[exp_id]["dir"]))
    cluster0 = CLUSTER_IDS[0] if CLUSTER_IDS else None
    return (
        exp_id, None, cluster0,
        mlp_table_data, mlp_table_cols, [],
        _split_banner(),
    )


# ── Callbacks — LazyPredict ──────────────────────────────────────────────────


def _lazy_df_for(exp_id) -> pd.DataFrame:
    """DataFrame do experimento selecionado (recai no default se inválido)."""
    if exp_id in _lazy_exp_by_id:
        return _load_lazy_df(Path(_lazy_exp_by_id[exp_id]["path"]))
    return lazy_df


@callback(
    Output("lazy-exp-dropdown", "options"),
    Output("lazy-exp-dropdown", "value"),
    Input("lazy-ablation-group-dropdown", "value"),
    prevent_initial_call=True,
)
def _update_lazy_experiments(ablation_group):
    filtered = LAZY_EXPERIMENTS if ablation_group in (None, "__all__") else [
        e for e in LAZY_EXPERIMENTS if e.get("ablation_group") == ablation_group
    ]
    options = [{"label": e["label"], "value": e["id"]} for e in filtered]
    value = filtered[0]["id"] if filtered else None
    return options, value


@callback(
    Output("lazy-cluster-dropdown", "options"),
    Output("lazy-cluster-dropdown", "value"),
    Input("lazy-exp-dropdown", "value"),
)
def _update_lazy_clusters(exp_id):
    ids = _cluster_ids(_lazy_df_for(exp_id))
    options = [{"label": f"Cluster {c}", "value": c} for c in ids]
    value = ids[0] if ids else None
    return options, value


@callback(
    Output("lazy-bar", "figure"),
    Output("lazy-table", "data"),
    Output("lazy-table", "columns"),
    Output("lazy-table", "style_data_conditional"),
    Input("lazy-exp-dropdown", "value"),
    Input("lazy-cluster-dropdown", "value"),
)
def _update_lazy(exp_id, cluster_id):
    df = _lazy_df_for(exp_id)
    ids = _cluster_ids(df)
    # Ao trocar de experimento, o cluster antigo pode não existir no novo
    if cluster_id not in ids:
        cluster_id = ids[0] if ids else None
    data, cols, cond = _lazy_table_data(df, cluster_id)
    return _build_lazy_bar(df, cluster_id), data, cols, cond


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    app.run(debug=debug, port=port)
