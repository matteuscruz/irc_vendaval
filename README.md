# IRC Vendaval

Deep learning ensemble system for probabilistic wind speed forecasting using EMOS (Ensemble Model Output Statistics) with Neural Networks.

## Overview

IRC Vendaval implements probabilistic forecasting of extreme wind speed events combining spatial and temporal deep learning architectures:

- **NNModel** — dense network over 1D meteorological features (Keras/TensorFlow)
- **NNConvModel** — CNN over spatial wind speed grids (Keras/TensorFlow)
- **CNNModel** — 3D CNN autoencoder for spatio-temporal encoding (PyTorch)
- **CGRU / CLSTM** — convolutional GRU and LSTM cells for sequential data (PyTorch)

Outputs are parametric probability distributions optimized with the Threshold-weighted Continuous Ranked Probability Score (twCRPS) for improved performance on extreme events.

## References

- *Improving Probabilistic Forecasting in the Netherlands*
- *Adapting a deep convolutional RNN model with imbalanced regression loss for improved spatio-temporal forecasting of extreme wind speed events in the short to medium range*

## Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environment and install dependencies

```bash
uv sync
```

Include development dependencies (pytest, etc.):

```bash
uv sync --extra dev
```

### Activate the environment

```bash
source .venv/bin/activate
```

Or use `uv run` to execute commands without activating:

```bash
uv run python main.py
```

## Running Tests

```bash
uv run pytest
```

With coverage report:

```bash
uv run pytest --cov=src
```

## Project Structure

```
irc_vendaval/
├── src/
│   ├── models/
│   │   ├── nn_model.py          # Keras/TF dense and CNN models
│   │   ├── convcnn_model.py     # PyTorch 3D CNN autoencoder
│   │   └── convrnn_model.py     # PyTorch ConvGRU and ConvLSTM cells
│   ├── pipeline/
│   │   └── training/
│   │       └── train_bagging.py # Bagging ensemble training script
│   ├── config/                  # Configuration files
│   ├── data/                    # Data loading and processing
│   ├── utils/                   # Shared utilities
│   └── visualization/           # Plotting and analysis tools
├── test/                        # Test suite (pytest)
├── artifacts/
│   ├── logs/                    # Training logs
│   └── models/                  # Saved model weights
├── dataset/                     # Input datasets
├── research/
│   └── papers/                  # Reference papers and notes
└── main.py                      # Entry point
```

## Ensemble Training Configuration

| Parameter | Value |
|---|---|
| Ensemble size | 10 models |
| Epochs | 74 |
| Batch size | 64 |
| Optimizer | Adam (lr=0.000105) |
| Distribution | Truncated Normal |
| Loss | twCRPS (sigmoid chain, threshold=12 m/s) |
| L2 regularization | 0.031658 |
| Hidden units | [170, 170] |
| CV folds | 3 |

## Input Features

| Feature | Timesteps |
|---|---|
| wind_speed | 15 |
| press | 1 |
| kinetic | 1 |
| humid | 1 |
| geopot | 1 |
