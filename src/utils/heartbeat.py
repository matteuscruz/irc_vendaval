"""Heartbeat para operações longas e silenciosas (ex: .load()/.to_netcdf() em
arquivos grandes) — imprime progresso periódico pra distinguir "ainda
trabalhando" de "travou", já que blocos como esses não têm callback de
progresso nativo no xarray/netCDF4.
"""
from __future__ import annotations

import threading
import time


class Heartbeat:
    """Context manager que imprime `label` a cada `interval` segundos."""

    def __init__(self, label: str, interval: float = 30.0) -> None:
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            print(f"[heartbeat] {self.label}: ainda rodando "
                  f"({time.time() - self._t0:.0f}s decorridos)...")

    def __enter__(self) -> "Heartbeat":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
