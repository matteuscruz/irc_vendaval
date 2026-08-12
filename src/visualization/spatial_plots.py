"""Plots espaciais comparativos entre pipelines (estilo plot.py / SpatialCorrector).

Gera um painel nacional com N mapas absolutos (um por pipeline) + diffs
pareados contra uma pipeline de referência, na mesma paleta/estilo dark
usado em src.inference.spatial_correction._plot_map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

WIND_COLORS = [
    "#d4e9f7", "#95c8e8", "#4da6d6", "#1a7fc4", "#0d5a9e", "#1a8a3c",
    "#6ab84d", "#c8d84a", "#f5d020", "#f5980a", "#e85a0a", "#c41a0a",
    "#8c0a1a", "#4a0a28",
]


def _prep_axis(ax, cfeature):
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#0f1e2c")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#07111c")
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#243040", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#3a5568", linewidth=1.0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#4a7090")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2f40")
        sp.set_linewidth(1.5)


def plot_pipeline_comparison(
    named_max: dict[str, "xr.DataArray"],
    out_path: str | Path,
    reference: str | None = None,
    title: str = "Comparação Espacial entre Pipelines — Rajada Máxima (Período)",
    extent: tuple[float, float, float, float] = (-74.5, -34.0, -34.5, 6.0),
) -> Path:
    """Painel nacional: 1 mapa absoluto por pipeline + diffs vs `reference`.

    Parameters
    ----------
    named_max : {nome_da_pipeline: DataArray 2D (latitude, longitude)}
        Já reduzido (ex: .max(dim="time")), coordenadas monotonicamente
        crescentes (sortby latitude/longitude antes de chamar).
    reference : nome da pipeline usada como base dos painéis de diferença.
        Default: a primeira chave de `named_max`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    names = list(named_max.keys())
    if len(names) < 1:
        raise ValueError("Nenhuma pipeline fornecida para comparação espacial.")
    ref = reference or names[0]
    if ref not in named_max:
        raise ValueError(f"Pipeline de referência '{ref}' não está em {names}")

    cmap_wind = LinearSegmentedColormap.from_list("wind_nws", WIND_COLORS, N=512)
    cmap_diff = plt.get_cmap("RdBu_r")
    proj = ccrs.PlateCarree()

    diffs = {n: named_max[n] - named_max[ref] for n in names if n != ref}

    panels = [(n, named_max[n], f"{n} (absoluto)", cmap_wind, True) for n in names]
    panels += [(n, d, f"diff: {n} − {ref}", cmap_diff, False) for n, d in diffs.items()]

    vmin_abs = min(float(da.min()) for da in named_max.values())
    vmax_abs = max(float(da.max()) for da in named_max.values())
    vmax_diff = max((float(np.abs(d).max()) for d in diffs.values()), default=1.0)
    if vmax_diff == 0:
        vmax_diff = 1.0

    n_panels = len(panels)
    fig = plt.figure(figsize=(6.2 * n_panels, 6.5), facecolor="#07111c")
    gs = fig.add_gridspec(1, n_panels, wspace=0.15)

    lon_min, lon_max, lat_min, lat_max = extent

    for i, (name, da, subtitle, cmap, is_abs) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        _prep_axis(ax, cfeature)

        if is_abs:
            norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
            levels = np.linspace(vmin_abs, vmax_abs, 40)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
            levels = np.linspace(-vmax_diff, vmax_diff, 40)

        cf = ax.contourf(
            da.longitude, da.latitude, da.values,
            levels=levels, cmap=cmap, norm=norm, transform=proj, extend="both", alpha=0.92,
        )
        ax.set_title(subtitle, color="#d0e4f0", fontsize=11, fontweight="bold", pad=12)

        gl = ax.gridlines(draw_labels=True, color="#1a2f40", alpha=0.5, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        if i > 0:
            gl.left_labels = False
        gl.xlabel_style = {"size": 8, "color": "#6090a8"}
        gl.ylabel_style = {"size": 8, "color": "#6090a8"}

        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08, aspect=25, shrink=0.85)
        cbar.ax.tick_params(colors="#6090a8", labelsize=8)
        cbar.set_label("Rajada Máxima (m/s)" if is_abs else "Diferença (m/s)", color="#a0c4d8", size=9)
        cbar.outline.set_edgecolor("#243040")

    fig.suptitle(title, fontsize=14, color="#d0e8f4", fontweight="bold", y=1.05)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[spatial_plots] Salvo: {out_path}")
    return out_path
