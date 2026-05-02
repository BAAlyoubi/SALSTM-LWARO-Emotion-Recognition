from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    sequence: torch.Tensor
    label: torch.Tensor


def _as_sequence(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[:, None, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected 2D or 3D feature array, got shape {array.shape}")


def _pad_modalities(modalities: list[np.ndarray]) -> np.ndarray:
    max_time = max(modality.shape[1] for modality in modalities)
    padded = []
    for modality in modalities:
        if modality.shape[1] == max_time:
            padded.append(modality)
            continue
        pad_width = ((0, 0), (0, max_time - modality.shape[1]), (0, 0))
        padded.append(np.pad(modality, pad_width=pad_width, mode="constant"))
    return np.concatenate(padded, axis=-1).astype(np.float32)


class MultimodalNPZDataset(Dataset[Sample]):
    """Loads text, video, audio, and labels from a compressed NumPy file."""

    def __init__(self, path: str | Path) -> None:
        archive = np.load(Path(path), allow_pickle=False)
        required = {"text", "video", "audio", "labels"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing required arrays in {path}: {sorted(missing)}")

        text = _as_sequence(archive["text"])
        video = _as_sequence(archive["video"])
        audio = _as_sequence(archive["audio"])
        labels = archive["labels"].astype(np.int64)

        sample_count = labels.shape[0]
        for name, modality in {"text": text, "video": video, "audio": audio}.items():
            if modality.shape[0] != sample_count:
                raise ValueError(f"{name} sample count does not match labels")

        self.features = torch.from_numpy(_pad_modalities([text, video, audio]))
        self.labels = torch.from_numpy(labels)

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[-1])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Sample:
        return Sample(sequence=self.features[index], label=self.labels[index])


def collate_samples(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [sample.sequence for sample in samples]
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack([sample.label for sample in samples])
    return padded, lengths, labels

