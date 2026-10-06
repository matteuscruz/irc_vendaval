"""Extração de campo em grade para pontos (estações): nearest ou bilinear.

Bilinear = interpolação nos 4 nós da célula que contém o ponto (a metodologia
adotada interpola os preditores "de forma bilinear para as coordenadas
geográficas exatas das estações"). Onde algum dos 4 nós é NaN — típico na
borda da máscara da bacia do ERA5-Basin, que é NaN fora da bacia — o bilinear
viraria NaN e uma estação coberta pelo nearest perderia o dado; nesses pontos
usa-se o nó de maior peso (o mais próximo), sem leitura extra.

Coordenadas podem ser crescentes ou decrescentes (a latitude do ERA5-Basin vai
de -13.75 a -35.0).

O nearest dos loaders continua sendo o `.sel(method="nearest")` original; o
modo "nearest" daqui existe para os testes e para quem precisar da mesma
interface nos dois métodos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

METHODS = ("nearest", "bilinear")


@dataclass(frozen=True)
class PointIndexer:
    lat_idx: np.ndarray    # (K, N) índice na dim de latitude, por nó
    lon_idx: np.ndarray    # (K, N) índice na dim de longitude, por nó
    weights: np.ndarray    # (K, N) pesos; somam 1 por ponto
    in_bounds: np.ndarray  # (N,) ponto dentro do retângulo da grade
    method: str

    def subset(self, sl) -> "PointIndexer":
        """Mesmo indexador restrito a um subconjunto de pontos (lote)."""
        return PointIndexer(
            self.lat_idx[:, sl], self.lon_idx[:, sl], self.weights[:, sl],
            self.in_bounds[sl], self.method,
        )


def _bracket(coord: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Para cada x, (i0, i1, w) tal que valor ≈ (1-w)·v[i0] + w·v[i1]."""
    coord = np.asarray(coord, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(coord)
    if n < 2:
        raise ValueError("interpolação bilinear precisa de ao menos 2 nós por eixo")
    descending = coord[0] > coord[-1]
    c = coord[::-1] if descending else coord
    j1 = np.clip(np.searchsorted(c, x, side="right"), 1, n - 1)
    j0 = j1 - 1
    w = np.clip((x - c[j0]) / (c[j1] - c[j0]), 0.0, 1.0)
    if descending:
        j0, j1 = n - 1 - j0, n - 1 - j1
    return j0, j1, w


def _nearest_index(coord: np.ndarray, x: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=float)
    return np.abs(coord[None, :] - np.asarray(x, dtype=float)[:, None]).argmin(axis=1)


def build_point_indexer(grid_lat, grid_lon, lats, lons, method: str = "bilinear") -> PointIndexer:
    if method not in METHODS:
        raise ValueError(f"method inválido: {method!r} ({' | '.join(METHODS)})")
    grid_lat = np.asarray(grid_lat, dtype=float)
    grid_lon = np.asarray(grid_lon, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    in_bounds = (
        (lats >= grid_lat.min()) & (lats <= grid_lat.max())
        & (lons >= grid_lon.min()) & (lons <= grid_lon.max())
    )
    if method == "nearest":
        return PointIndexer(
            _nearest_index(grid_lat, lats)[None, :],
            _nearest_index(grid_lon, lons)[None, :],
            np.ones((1, len(lats))),
            in_bounds, method,
        )
    a0, a1, wa = _bracket(grid_lat, lats)
    b0, b1, wb = _bracket(grid_lon, lons)
    return PointIndexer(
        lat_idx=np.stack([a0, a0, a1, a1]),
        lon_idx=np.stack([b0, b1, b0, b1]),
        weights=np.stack([(1 - wa) * (1 - wb), (1 - wa) * wb, wa * (1 - wb), wa * wb]),
        in_bounds=in_bounds,
        method=method,
    )


def extract_points(
    ds: xr.Dataset,
    indexer: PointIndexer,
    point_dim: str,
    point_coords,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Valores de `ds` nos pontos do indexador, com dimensão `point_dim`.

    Em Dataset lazy sem dask, a aritmética do bilinear força a leitura — chame
    com `indexer.subset(...)` por lote para manter o custo de memória limitado.
    Pontos fora do retângulo da grade ficam NaN.
    """
    corners = []
    for k in range(indexer.lat_idx.shape[0]):
        sel = ds.isel({
            lat_name: xr.DataArray(indexer.lat_idx[k], dims=point_dim),
            lon_name: xr.DataArray(indexer.lon_idx[k], dims=point_dim),
        })
        corners.append(sel.drop_vars([c for c in (lat_name, lon_name) if c in sel.coords]))

    if len(corners) == 1:
        out = corners[0]
    else:
        weighted = sum(
            corner * xr.DataArray(indexer.weights[k], dims=point_dim)
            for k, corner in enumerate(corners)
        )
        all_valid = corners[0].notnull()
        for corner in corners[1:]:
            all_valid = all_valid & corner.notnull()
        nearest_k = np.argmax(indexer.weights, axis=0)
        fallback = corners[-1]
        for k in range(len(corners) - 2, -1, -1):
            fallback = corners[k].where(xr.DataArray(nearest_k == k, dims=point_dim), fallback)
        out = weighted.where(all_valid, fallback)

    point_coords = np.asarray(point_coords)
    out = out.assign_coords({point_dim: point_coords})
    if not indexer.in_bounds.all():
        out = out.where(xr.DataArray(
            indexer.in_bounds, dims=point_dim, coords={point_dim: point_coords},
        ))
    return out
