"""Undercomplete autoencoder trained on normal transactions only.

The model learns to reconstruct ordinary behaviour; the per-row reconstruction
error is then the anomaly score. Unlike Isolation Forest, the per-feature error
decomposition also says *which* fields drove the score, which is what turns a
flagged record into an actionable issue.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class AEConfig:
    """Hyperparameters for the autoencoder."""

    input_dim: int
    hidden: tuple[int, ...] = (24, 16)
    latent: int = 8
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    max_epochs: int = 150
    patience: int = 12
    min_delta: float = 1e-5
    val_fraction: float = 0.1
    seed: int = 42
    num_threads: int = 4


class TabularAutoencoder(nn.Module):
    """Symmetric encoder/decoder with a linear output head.

    The output layer is deliberately linear. A sigmoid or tanh head saturates on
    standardized inputs and clips exactly the extreme values that carry the
    anomaly signal.
    """

    def __init__(self, config: AEConfig) -> None:
        super().__init__()
        dims = [config.input_dim, *config.hidden, config.latent]

        encoder_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0:
                encoder_layers.append(nn.Dropout(config.dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        reversed_dims = dims[::-1]
        decoder_layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(reversed_dims[:-1], reversed_dims[1:])):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if index < len(reversed_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """Trains :class:`TabularAutoencoder` and scores rows by reconstruction error."""

    name = "autoencoder"

    def __init__(self, config: AEConfig) -> None:
        self.config = config
        # Seed before the weights are drawn, not in fit(): initialisation
        # happens here, so seeding later would leave it at the mercy of
        # whatever global RNG state the caller happened to have.
        torch.manual_seed(config.seed)
        self.model = TabularAutoencoder(config)
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch_: int = 0

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        config = self.config
        torch.set_num_threads(config.num_threads)

        data = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        generator = torch.Generator().manual_seed(config.seed)
        shuffled = data[torch.randperm(len(data), generator=generator)]
        n_val = max(1, int(len(shuffled) * config.val_fraction))
        val_data, train_data = shuffled[:n_val], shuffled[n_val:]

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-5
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state: dict | None = None
        epochs_without_improvement = 0

        for epoch in range(1, config.max_epochs + 1):
            self.model.train()
            permutation = torch.randperm(len(train_data), generator=generator)
            epoch_loss = 0.0
            # Manual slicing rather than a DataLoader: for an in-memory tensor
            # this is several times faster, and it keeps shuffling deterministic
            # under a single seeded generator.
            for start in range(0, len(train_data), config.batch_size):
                batch = train_data[permutation[start:start + config.batch_size]]
                optimizer.zero_grad()
                loss = criterion(self.model(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            train_loss = epoch_loss / len(train_data)

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(val_data), val_data).item()
            scheduler.step(val_loss)

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_loss - config.min_delta:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _reconstruct(self, X: np.ndarray) -> torch.Tensor:
        self.model.eval()
        data = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        with torch.no_grad():
            return self.model(data)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-row mean squared reconstruction error."""
        data = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        reconstructed = self._reconstruct(X)
        return ((reconstructed - data) ** 2).mean(dim=1).numpy().astype("float64")

    def per_feature_errors(self, X: np.ndarray) -> np.ndarray:
        """Squared error per cell, used to explain why a record was flagged."""
        data = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        reconstructed = self._reconstruct(X)
        return ((reconstructed - data) ** 2).numpy().astype("float64")

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self._reconstruct(X).numpy().astype("float64")

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        config_path = path.with_suffix(".config.json")
        payload = asdict(self.config)
        payload["hidden"] = list(payload["hidden"])
        payload["best_epoch"] = self.best_epoch_
        payload["history"] = self.history_
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "AutoencoderDetector":
        path = Path(path)
        payload = json.loads(path.with_suffix(".config.json").read_text(encoding="utf-8"))
        history = payload.pop("history", {"train_loss": [], "val_loss": []})
        best_epoch = payload.pop("best_epoch", 0)
        payload["hidden"] = tuple(payload["hidden"])
        detector = cls(AEConfig(**payload))
        # weights_only keeps loading a checkpoint from being code execution.
        state = torch.load(path, map_location="cpu", weights_only=True)
        detector.model.load_state_dict(state)
        detector.model.eval()
        detector.history_ = history
        detector.best_epoch_ = best_epoch
        return detector
