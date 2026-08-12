"""
analise_comparativa_suavizacao.py
==================================
Análise espacial comparativa dos três métodos de interpolação IDW para o
projeto Vendaval V2. Focada em um "domo" de 200km de uma estação referencial.
Gera painéis de máximos históricos, p95 e p99, e mapas de diferença.
Adicionalmente, plota um mapa do Brasil com a distribuição espacial das estações e da grade.
 
USO:
    ./.venv/bin/python 05_analysis/analise_comparativa_suavizacao.py
 
SAÍDA:
    05_analysis/plots/dome_200km_max.png
    05_analysis/plots/dome_200km_p95.png
    05_analysis/plots/dome_200km_p99.png
    05_analysis/plots/mapa_brasil_scatter.png
"""
 
import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.geodesic as cgeo
from shapely.geometry import Polygon
 
warnings.filterwarnings('ignore')
 
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_GLOB   = os.path.join(BASE_DIR, "output_corrigido_v2_1", "*.nc")
EPS_GLOB   = os.path.join(BASE_DIR, "output_corrigido_v2_epsilon", "*.nc")
GAUS_GLOB  = os.path.join(BASE_DIR, "output_corrigido_v2_gaussian", "*.nc")
CSV_PATH   = os.path.join(BASE_DIR, "data", "in", "Training_Dataset_INMET_ERA5_Paired.csv")
OUT_DIR    = os.path.join(BASE_DIR, "05_analysis", "plots")
 
os.makedirs(OUT_DIR, exist_ok=True)
 
# Estação de Referência: A879 (Cambará do Sul, RS) - Estação com rajadas muito fortes
REF_STATION_CODE = "A879"
 
# ═══════════════════════════════════════════════════════════════════════════════
# PALETAS DE CORES
# ═══════════════════════════════════════════════════════════════════════════════
WIND_COLORS = [
    "#d4e9f7", "#95c8e8", "#4da6d6", "#1a7fc4", "#0d5a9e", "#1a8a3c",
    "#6ab84d", "#c8d84a", "#f5d020", "#f5980a", "#e85a0a", "#c41a0a",
    "#8c0a1a", "#4a0a28"
]
cmap_wind = LinearSegmentedColormap.from_list("wind_nws", WIND_COLORS, N=512)
cmap_diff = plt.get_cmap("RdBu_r")
 
# ═══════════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DE DADOS: LOCAL E REFERÊNCIA
# ═══════════════════════════════════════════════════════════════════════════════
print("Carregando estações...")
df = pd.read_csv(CSV_PATH)
stations = df[["codigo_estacao", "latitude", "longitude"]].drop_duplicates("codigo_estacao").reset_index(drop=True)
 
ref_st = stations[stations["codigo_estacao"] == REF_STATION_CODE].iloc[0]
lat_c, lon_c = ref_st["latitude"], ref_st["longitude"]
 
# Raio de ~200km corresponde a ~2.0 graus
RADIUS_DEG = 2.0
lat_min, lat_max = lat_c - RADIUS_DEG, lat_c + RADIUS_DEG
lon_min, lon_max = lon_c - RADIUS_DEG, lon_c + RADIUS_DEG
 
print(f"Estação de Referência: {REF_STATION_CODE} (Lat: {lat_c:.2f}, Lon: {lon_c:.2f})")
print(f"Recorte espacial (domo 200km): Lat {lat_min:.2f} a {lat_max:.2f}, Lon {lon_min:.2f} a {lon_max:.2f}")
 
def load_sliced_metric(pattern, lat_min, lat_max, lon_min, lon_max, metric="max", q=0.95):
    """Carrega dados já recortados no domínio local, evitando estourar memória com quantis."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado: {pattern}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    # Lats decrescentes no ERA5, então o slice deve ser de max para min
    sliced = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    
    if metric == "max":
        return sliced["rajada_max_corrigida"].max(dim="valid_time").load()
    elif metric == "quantile":
        return sliced["rajada_max_corrigida"].quantile(q, dim="valid_time").load()
 
# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS DO DOMO (200km)
# ═══════════════════════════════════════════════════════════════════════════════
metrics = [("max", None), ("quantile", 0.95), ("quantile", 0.99)]
metric_names = ["max", "p95", "p99"]
PROJ = ccrs.PlateCarree()
 
for (m_type, q), m_name in zip(metrics, metric_names):
    print(f"\nCalculando métrica: {m_name.upper()}...")
    da_ref  = load_sliced_metric(REF_GLOB, lat_min, lat_max, lon_min, lon_max, metric=m_type, q=q)
    da_eps  = load_sliced_metric(EPS_GLOB, lat_min, lat_max, lon_min, lon_max, metric=m_type, q=q)
    da_gaus = load_sliced_metric(GAUS_GLOB, lat_min, lat_max, lon_min, lon_max, metric=m_type, q=q)
    
    # Diferenças
    diff_eps  = da_eps - da_ref
    diff_gaus = da_gaus - da_ref
    
    fig = plt.figure(figsize=(25, 6), facecolor="#07111c")
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.15)
    
    panels = [
        (da_ref, "1. Original (Sem Suavização)", cmap_wind, True),
        (da_eps, "2. IDW + Epsilon", cmap_wind, True),
        (da_gaus, "3. Gaussian Kernel", cmap_wind, True),
        (diff_eps, "4. Diff: IDW - Original", cmap_diff, False),
        (diff_gaus, "5. Diff: Gauss - Original", cmap_diff, False)
    ]
    
    vmin_abs = float(da_ref.min())
    vmax_abs = float(da_ref.max())
    # Para a divergência, queremos que o 0 fique centralizado
    vmax_diff = max(abs(float(diff_eps.min())), abs(float(diff_eps.max())),
                    abs(float(diff_gaus.min())), abs(float(diff_gaus.max())))
    
    axes = []
    for i, (da, title, cmap, is_abs) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], projection=PROJ)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=PROJ)
        
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#0f1e2c")
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#07111c")
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#4a7090")
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#243040", linewidth=0.5)
        
        # Borda do painel
        for sp in ax.spines.values():
            sp.set_edgecolor('#1a2f40')
            sp.set_linewidth(1.5)
        
        # Desenha o domo de 200km
        geom_coords = cgeo.Geodesic().circle(lon=lon_c, lat=lat_c, radius=200000)
        geom = Polygon(geom_coords)
        ax.add_geometries((geom,), crs=PROJ, facecolor='none', edgecolor='#ffffff', linewidth=1.5, linestyle='--', alpha=0.5)
        
        # Estação
        ax.scatter(lon_c, lat_c, color='#f5d020', marker='*', s=200, zorder=10, transform=PROJ, edgecolors='black')
        
        # Plot do campo de vento/diferença
        if is_abs:
            norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
            levels = np.linspace(vmin_abs, vmax_abs, 40)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
            levels = np.linspace(-vmax_diff, vmax_diff, 40)
            
        cf = ax.contourf(da.longitude, da.latitude, da.values, levels=levels, cmap=cmap, norm=norm, transform=PROJ, extend='both', alpha=0.92)
        
        ax.set_title(title, color="#d0e4f0", fontsize=11, fontweight="bold", pad=12)
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, color='#1a2f40', alpha=0.5, linestyle=':')
        gl.top_labels = False; gl.right_labels = False
        if i > 0: gl.left_labels = False
        gl.xlabel_style = {'size': 8, 'color': '#6090a8'}
        gl.ylabel_style = {'size': 8, 'color': '#6090a8'}
        
        axes.append(ax)
        
        # Colorbar individual (ou pares)
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08, aspect=25, shrink=0.85)
        cbar.ax.tick_params(colors='#6090a8', labelsize=8)
        cbar.set_label("Diferença (m/s)" if not is_abs else f"{m_name.upper()} (m/s)", color="#a0c4d8", size=9)
        cbar.outline.set_edgecolor("#243040")
 
    fig.suptitle(f"Diagnóstico Geográfico (Domo 200km) - Estação {REF_STATION_CODE} | Métrica: {m_name.upper()}",
                 fontsize=14, color="#d0e8f4", fontweight="bold", y=1.05)
    
    out_path = os.path.join(OUT_DIR, f"dome_200km_{m_name}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Salvo: {out_path}")
 
# ═══════════════════════════════════════════════════════════════════════════════
# PLOT DO MAPA DO BRASIL: SCATTER DE ESTAÇÕES E GRADE (5 PAINÉIS)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGerando Mapa Nacional em 5 painéis (Scatter da Média Histórica)...")
st_mean = df.groupby(["codigo_estacao", "latitude", "longitude"])["rajada_max_inmet_ms"].mean().reset_index()
 
def load_national_mean(pattern):
    files = sorted(glob.glob(pattern))
    ds = xr.open_mfdataset(files, combine="by_coords")
    # Para o mapa nacional em scatter, vamos usar stride=2 para visualização
    sliced = ds.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
    return sliced["rajada_max_corrigida"].mean(dim="valid_time").load()
 
print("Calculando médias históricas da grade para o Brasil todo...")
da_ref_nat  = load_national_mean(REF_GLOB)
da_eps_nat  = load_national_mean(EPS_GLOB)
da_gaus_nat = load_national_mean(GAUS_GLOB)
 
diff_eps_nat = da_eps_nat - da_ref_nat
diff_gaus_nat = da_gaus_nat - da_ref_nat
 
LONS_NAT, LATS_NAT = np.meshgrid(da_ref_nat.longitude.values, da_ref_nat.latitude.values)
sc_lons = LONS_NAT.flatten()
sc_lats = LATS_NAT.flatten()
 
# Filter inside Brazil Bbox
valid_idx = (sc_lons >= -74.5) & (sc_lons <= -34.0) & (sc_lats >= -34.5) & (sc_lats <= 6.0)
sc_lons = sc_lons[valid_idx]
sc_lats = sc_lats[valid_idx]
 
fig_nat = plt.figure(figsize=(25, 6), facecolor="#07111c")
gs_nat = fig_nat.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.15)
 
panels_nat = [
    (da_ref_nat, "1. Original (Média)", cmap_wind, True),
    (da_eps_nat, "2. IDW + Eps (Média)", cmap_wind, True),
    (da_gaus_nat, "3. Gaussian (Média)", cmap_wind, True),
    (diff_eps_nat, "4. Diff: IDW - Orig", cmap_diff, False),
    (diff_gaus_nat, "5. Diff: Gauss - Orig", cmap_diff, False)
]
 
# Recalculate color limits for mean
vmin_nat = float(da_ref_nat.min())
vmax_nat = float(da_ref_nat.max())
vmax_diff_nat = max(abs(float(diff_eps_nat.min())), abs(float(diff_eps_nat.max())),
                abs(float(diff_gaus_nat.min())), abs(float(diff_gaus_nat.max())))
 
axes_nat = []
for i, (da, title, cmap, is_abs) in enumerate(panels_nat):
    ax = fig_nat.add_subplot(gs_nat[0, i], projection=PROJ)
    ax.set_extent([-74.5, -34.0, -34.5, 6.0], crs=PROJ)
    
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#0f1e2c")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#07111c")
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#243040", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#3a5568", linewidth=1.0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#4a7090")
    
    for sp in ax.spines.values():
        sp.set_edgecolor('#1a2f40')
        sp.set_linewidth(1.5)
        
    vals = da.values.flatten()[valid_idx]
    
    if is_abs:
        norm = mcolors.Normalize(vmin=vmin_nat, vmax=vmax_nat)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=-vmax_diff_nat, vcenter=0, vmax=vmax_diff_nat)
        
    # Scatter the grid
    sc = ax.scatter(
        sc_lons, sc_lats, c=vals,
        s=1.5, cmap=cmap, norm=norm, alpha=0.8, transform=PROJ,
        marker='s', linewidths=0
    )
    
    # Scatter stations
    if is_abs:
        # Plot stations with their true historical mean using the same color scale
        ax.scatter(
            st_mean.longitude, st_mean.latitude, c=st_mean.rajada_max_inmet_ms,
            s=60, cmap=cmap, norm=norm, marker="*", transform=PROJ,
            edgecolors='black', linewidths=0.5
        )
    else:
        # For diff panels, just plot stations as white dots for reference
        ax.scatter(
            st_mean.longitude, st_mean.latitude, c='white',
            s=25, marker="*", transform=PROJ, alpha=0.4, linewidths=0
        )
        
    ax.set_title(title, color="#d0e4f0", fontsize=11, fontweight="bold", pad=12)
    
    gl = ax.gridlines(draw_labels=True, color='#1a2f40', alpha=0.6, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    if i > 0: gl.left_labels = False
    gl.xlabel_style = {'size': 8, 'color': '#6090a8'}
    gl.ylabel_style = {'size': 8, 'color': '#6090a8'}
    
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.08, aspect=25, shrink=0.85)
    cbar.ax.tick_params(colors='#6090a8', labelsize=8)
    cbar.set_label("Média Histórica (m/s)" if is_abs else "Diferença da Média (m/s)", color="#a0c4d8", size=9)
    cbar.outline.set_edgecolor("#243040")
 
fig_nat.suptitle("Análise Comparativa em Escala Nacional: Scatter da Média Histórica (Grade vs Estações)",
                 fontsize=14, color="#d0e8f4", fontweight="bold", y=1.05)
 
out_nat = os.path.join(OUT_DIR, "mapa_brasil_scatter_comparativo.png")
plt.savefig(out_nat, dpi=180, bbox_inches="tight", facecolor=fig_nat.get_facecolor())
plt.close()
print(f"✓ Salvo: {out_nat}")
 
print("\nProcessamento concluído com sucesso!")