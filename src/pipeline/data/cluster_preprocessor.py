from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.data.climatology import get_climatology
from src.data.cluster_assigner import assign_station_clusters
from src.data.netcdf_loader import NetCDFLoader
from src.data.static_features import StationStaticFeatures


SEASONS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

from src.pipelines.common import (
    ERA5_GUST_PROXY, TEST_SLICE, TRAIN_SLICE, VAL_SLICE, build_flat_dataframe,
    resolve_feature_groups, restrict_to_feature_coverage,
)


@dataclass
class ClusterDataBatch:
    # {season: (N, lookback, n_features)}
    x_train: dict[str, np.ndarray] = field(default_factory=dict)
    x_val: dict[str, np.ndarray] = field(default_factory=dict)
    x_test: dict[str, np.ndarray] = field(default_factory=dict)
    # {season: (N, n_static)} — features estáticas por amostra (TRWindBC-style)
    x_static_train: dict[str, np.ndarray] = field(default_factory=dict)
    x_static_val: dict[str, np.ndarray] = field(default_factory=dict)
    x_static_test: dict[str, np.ndarray] = field(default_factory=dict)
    # {season: (N, 1)} — razao INMET/ERA5 escalonada
    y_train: dict[str, np.ndarray] = field(default_factory=dict)
    y_val: dict[str, np.ndarray] = field(default_factory=dict)
    y_test: dict[str, np.ndarray] = field(default_factory=dict)
    # {season: (N,)} — valores brutos de ERA5 (wind_mag_max), ancora de
    # reconstrucao (y_pred = ratio_pred * era5)
    era5_train: dict[str, np.ndarray] = field(default_factory=dict)
    era5_val: dict[str, np.ndarray] = field(default_factory=dict)
    era5_test: dict[str, np.ndarray] = field(default_factory=dict)
    # {season: {"estacao"/"latitude"/"longitude"/"time": (N,)}} — identidade
    # de cada janela, alinhada posicionalmente com x_*/y_*/era5_* do mesmo
    # split/season. Permite reconstruir predictions_by_station.csv (mesma
    # granularidade que cluster_mlp.py já produz) em vez de só o agregado
    # por cluster_id.
    meta_train: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    meta_val: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    meta_test: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    scaler_x: RobustScaler = field(default_factory=RobustScaler)
    scaler_y: RobustScaler = field(default_factory=RobustScaler)
    imputer_x: SimpleImputer = field(
        default_factory=lambda: SimpleImputer(strategy="mean", keep_empty_features=True)
    )
    feature_names: list[str] = field(default_factory=list)
    static_feature_names: list[str] = field(default_factory=list)
    cluster_ids: list[int] = field(default_factory=list)
    station_clusters_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class ClusterPreprocessor:
    """
    Orquestra o pre-processamento para o pipeline de clusters:
      1. Carrega INMET + ERA5 (NetCDF)
      2. Atribui clusters via spatial join
      3. Split espacial (80/20 estacoes treino/teste)
      4. Calcula climatologia so no periodo de treino
      5. Features + anomalias + one-hot de cluster
      6. RobustScaler ajustado so no treino
      7. Janelas deslizantes
      8. Segregacao por estacao climatica (DJF/MAM/JJA/SON)
    """

    def __init__(
        self,
        raw_dir: str,
        shp_dir: str,
        target_var: str = "daily_wind_gust_max",
        train_slice: tuple[str, str] = TRAIN_SLICE,
        val_slice: tuple[str, str] = VAL_SLICE,
        test_slice: tuple[str, str] = TEST_SLICE,
        test_station_fraction: float = 0.2,
        lookback: int = 7,
        seed: int = 42,
        feature_groups: str | None = None,
        restrict_coverage: bool = False,
    ) -> None:
        self.raw_dir = raw_dir
        self.shp_dir = shp_dir
        self.target_var = target_var
        self.train_slice = slice(*train_slice)
        self.val_slice = slice(*val_slice)
        self.test_slice = slice(*test_slice)
        self.test_station_fraction = test_station_fraction
        self.lookback = lookback
        self.seed = seed
        self.feature_groups = feature_groups
        self.restrict_coverage = restrict_coverage

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(self) -> ClusterDataBatch:
        print("[preprocessor] Carregando datasets NetCDF...")
        ds_inmet, ds_era5 = NetCDFLoader(self.raw_dir).load_extended()

        print("[preprocessor] Atribuindo clusters...")
        station_clusters = assign_station_clusters(ds_inmet, self.shp_dir)
        cluster_ids = sorted(station_clusters["cluster_id"].unique().tolist())

        cluster_cols = [f"cluster_{c}" for c in cluster_ids]
        # feature_names será definido após construir o DataFrame (só features disponíveis)

        print("[preprocessor] Split apenas temporal — todas as estacoes em train/val/test...")
        # Alinhado com cluster_lazy.py/cluster_mlp.py: mesmas estações em
        # treino/val/teste, diferenciadas só pelo corte de tempo (TRAIN_SLICE/
        # VAL_SLICE/TEST_SLICE) — não por holdout espacial de estações nunca
        # vistas no treino. Isso sacrifica a validação de generalização
        # espacial em troca de fidelidade estrita de comparação entre as 3
        # pipelines no braço "original". test_station_fraction (parâmetro do
        # construtor, YAML, spatial_correction_dl.py) fica mantido por
        # compatibilidade de assinatura, mas não é mais usado aqui.
        all_stations = ds_inmet.estacao.values
        train_stations = all_stations
        test_stations = all_stations

        # Climatologia ERA5 necessária para a feature era5_clim_wind (entrada,
        # não mais o alvo — ver _build_dataframe: alvo agora é razão INMET/ERA5,
        # igual cluster_mlp.py/cluster_lazy.py, não anomalia vs. climatologia).
        print("[preprocessor] Calculando climatologia ERA5 (treino)...")
        ds_clim_era5 = get_climatology(ds_era5, ERA5_GUST_PROXY, self.train_slice)

        print("[preprocessor] Construindo DataFrame de features...")
        df = self._build_dataframe(
            ds_inmet, ds_era5, station_clusters, ds_clim_era5, cluster_ids
        )
        if self.restrict_coverage:
            df = restrict_to_feature_coverage(df, self.feature_groups)

        df_tr = df[df["estacao"].isin(train_stations)]
        df_te = df[df["estacao"].isin(test_stations)]

        print("[preprocessor] Ajustando scalers no treino...")
        df_train = df_tr[
            (df_tr["time"] >= self.train_slice.start)
            & (df_tr["time"] <= self.train_slice.stop)
        ]
        # Filtra apenas features disponíveis (ERA5-18UTC/BT55 podem estar ausentes)
        _candidate_features = resolve_feature_groups(self.feature_groups)
        avail_features = [f for f in _candidate_features if f in df.columns and df[f].notna().any()]
        missing_feats = [f for f in _candidate_features if f not in df.columns]
        if missing_feats:
            print(
                f"[preprocessor] AVISO: {len(missing_feats)} features ausentes "
                f"(p.ex. _18z/bt55): {missing_feats[:5]}{'...' if len(missing_feats) > 5 else ''}"
            )
        feature_names = avail_features + cluster_cols

        # Imputação (média, fit só no treino) ANTES do scaler — sem isso,
        # qualquer estação sem cobertura de era5_18z/bt55 (só 47/271 e
        # 58/271 respectivamente) fica com NaN nessas features e
        # _make_windows() descarta a janela inteira (ver comentário lá).
        # Mesma técnica de preprocess_df() em common.py, usada por
        # lazy/MLP — sem isso o LSTM ficava restrito a ~45/271 estações
        # (só as com cobertura simultânea de era5_18z E bt55), bem menos
        # que as outras duas pipelines (~235/271).
        imputer_x = SimpleImputer(strategy="mean", keep_empty_features=True)
        imputer_x.fit(df_train[avail_features])

        scaler_x = RobustScaler()
        scaler_y = RobustScaler()
        scaler_x.fit(imputer_x.transform(df_train[avail_features]))
        scaler_y.fit(df_train[["ratio"]])

        print("[preprocessor] Computando features estaticas por estacao...")
        static_builder = StationStaticFeatures()
        df_static = static_builder.compute(
            ds_inmet, self.target_var, self.train_slice, train_stations
        )

        print("[preprocessor] Gerando janelas por estacao climatica...")
        batch = ClusterDataBatch(
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            imputer_x=imputer_x,
            feature_names=feature_names,
            static_feature_names=StationStaticFeatures.FEATURE_NAMES,
            cluster_ids=cluster_ids,
            station_clusters_df=station_clusters,
        )

        splits = {
            "train": (df_tr, self.train_slice),
            "val": (df_tr, self.val_slice),
            "test": (df_te, self.test_slice),
        }

        for split_name, (df_split, time_sl) in splits.items():
            mask = (df_split["time"] >= time_sl.start) & (
                df_split["time"] <= time_sl.stop
            )
            df_period = df_split[mask].copy()
            x_s, xs_s, y_s, era5_s, meta_s = self._make_windows(
                df_period, avail_features, cluster_cols, imputer_x, scaler_x,
                scaler_y, df_static,
            )
            setattr(batch, f"x_{split_name}", x_s)
            setattr(batch, f"x_static_{split_name}", xs_s)
            setattr(batch, f"y_{split_name}", y_s)
            setattr(batch, f"era5_{split_name}", era5_s)
            setattr(batch, f"meta_{split_name}", meta_s)

        return batch

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _build_dataframe(
        self,
        ds_inmet,
        ds_era5,
        station_clusters,
        ds_clim_era5,
        cluster_ids,
    ):
        """Constroi DataFrame flat com features, razao ERA5 e cluster.

        Delega ao build_flat_dataframe canonical de common.py (que adiciona
        month_sin/cos, era5_clim_wind via ds_clim_era5, lags autoregressivos
        INMET, etc.) e depois acrescenta o alvo (razao INMET/ERA5) e one-hot
        de cluster.
        """
        # build_flat_dataframe requer climatologia ERA5 para era5_clim_wind
        df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim_era5)

        # Alvo = razao INMET/ERA5 (igual cluster_mlp.py/cluster_lazy.py) em vez
        # de anomalia vs. climatologia INMET — ancorado no ERA5 (que ja
        # correlaciona fortemente com a verdade), muito mais facil de aprender
        # que a variancia residual pura da anomalia climatologica.
        era5_safe = df[ERA5_GUST_PROXY].clip(lower=0.1)
        df["ratio"] = df[self.target_var] / era5_safe
        df = df.dropna(subset=["ratio", ERA5_GUST_PROXY])

        # One-hot de cluster (cluster_id já está no df via build_flat_dataframe)
        for c in cluster_ids:
            df[f"cluster_{c}"] = (df["cluster_id"] == c).astype(float)

        month_to_season: dict[int, str] = {}
        for s, months in SEASONS.items():
            for m in months:
                month_to_season[m] = s
        df["season"] = df["time"].dt.month.map(month_to_season)

        return df.sort_values(["estacao", "time"]).reset_index(drop=True)

    def _make_windows(
        self,
        df: pd.DataFrame,
        base_features: list[str],
        cluster_cols: list[str],
        imputer_x: SimpleImputer,
        scaler_x: RobustScaler,
        scaler_y: RobustScaler,
        df_static: pd.DataFrame,
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Janelas deslizantes segregadas por estacao climatica.

        Retorna (x_dynamic, x_static, y, era5, meta) — cada um mapeado por
        season. x_static[season] tem shape (N, n_static): o vetor estático
        da estação é replicado para cada janela gerada por aquela estação.
        era5[season] guarda o valor bruto de wind_mag_max por janela — ancora
        usada pra reconstruir a predição absoluta (y_pred = ratio_pred *
        era5), igual ao padrão de cluster_mlp.py. meta[season] é um dict com
        "estacao"/"latitude"/"longitude"/"time" por janela — permite ao
        pipeline salvar predictions_by_station.csv com granularidade real de
        estação, em vez de só o agregado por cluster_id.
        """
        x_seas: dict[str, list] = {s: [] for s in SEASONS}
        xs_seas: dict[str, list] = {s: [] for s in SEASONS}
        y_seas: dict[str, list] = {s: [] for s in SEASONS}
        era5_seas: dict[str, list] = {s: [] for s in SEASONS}
        station_seas: dict[str, list] = {s: [] for s in SEASONS}
        lat_seas: dict[str, list] = {s: [] for s in SEASONS}
        lon_seas: dict[str, list] = {s: [] for s in SEASONS}
        time_seas: dict[str, list] = {s: [] for s in SEASONS}

        static_lookup = df_static.set_index("estacao")

        for station, df_st in df.groupby("estacao"):
            df_st = df_st.sort_values("time")
            if len(df_st) < self.lookback:
                continue

            # Imputar (media do treino) antes de escalonar — evita descartar
            # a estacao inteira so por faltar era5_18z/bt55 (ver run()).
            # Cluster cols sao binarias (sem escala/imputacao).
            x_base = scaler_x.transform(imputer_x.transform(df_st[base_features]))
            x_cluster = df_st[cluster_cols].values
            x_all = np.hstack([x_base, x_cluster])

            y_scaled = scaler_y.transform(df_st[["ratio"]])
            era5_vals = df_st[ERA5_GUST_PROXY].values
            seasons = df_st["season"].values
            lat_val = df_st["latitude"].values
            lon_val = df_st["longitude"].values
            time_val = df_st["time"].values

            static_vec = static_lookup.loc[station].values.astype("float32")

            for i in range(self.lookback, len(df_st)):
                window = x_all[i - self.lookback: i]
                if not np.isfinite(window).all() or not np.isfinite(y_scaled[i]).all():
                    # janela contaminada por NaN (ex.: estacao fora da cobertura
                    # ERA5-18Z/BT55) — pular em vez de deixar o NaN se propagar
                    # silenciosamente pros pesos compartilhados da LSTM.
                    continue
                s = seasons[i]
                x_seas[s].append(window)
                xs_seas[s].append(static_vec)
                y_seas[s].append(y_scaled[i])
                era5_seas[s].append(era5_vals[i])
                station_seas[s].append(station)
                lat_seas[s].append(lat_val[i])
                lon_seas[s].append(lon_val[i])
                time_seas[s].append(time_val[i])

        # season-first (igual x_seas/y_seas/era5_seas) — meta[season]["estacao"],
        # não meta["estacao"][season].
        meta = {
            s: {
                "estacao": np.array(station_seas[s]),
                "latitude": np.array(lat_seas[s]),
                "longitude": np.array(lon_seas[s]),
                "time": np.array(time_seas[s]),
            }
            for s in SEASONS
        }

        return (
            {s: np.array(v) for s, v in x_seas.items()},
            {s: np.array(v) for s, v in xs_seas.items()},
            {s: np.array(v) for s, v in y_seas.items()},
            {s: np.array(v) for s, v in era5_seas.items()},
            meta,
        )
