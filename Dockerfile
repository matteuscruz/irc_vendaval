# syntax=docker/dockerfile:1
# ── Base: official TF image with CUDA already wired ───────────────────────────
FROM tensorflow/tensorflow:2.16.1-gpu

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python env ────────────────────────────────────────────────────────────────
WORKDIR /app

# Install only what the LSTM pipeline needs (torch not required here).
# Pin exact versions from uv.lock for reproducibility.
RUN pip install --no-cache-dir \
    "numpy>=1.24.0" \
    "pydantic>=2.13.3" \
    "pyyaml>=6.0.3" \
    "scikit-learn>=1.7.0" \
    "matplotlib>=3.10.0"

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/      ./src/
COPY config/   ./config/
COPY main.py   ./main.py

# Non-interactive matplotlib backend (no display needed)
ENV MPLBACKEND=Agg

# ── Dataset and artifacts come from volume mounts at runtime ──────────────────
# dataset/ → mounted at /app/dataset
# artifacts output dir → mounted at /artifacts

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "config/experiment_lstm.yaml"]
