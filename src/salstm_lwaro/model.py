from __future__ import annotations

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int | None = None) -> None:
        super().__init__()
        attention_dim = attention_dim or input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.projection(sequence).squeeze(-1)
        max_time = sequence.shape[1]
        mask = torch.arange(max_time, device=sequence.device)[None, :] >= lengths[:, None]
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return pooled, weights


class SALSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.25,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = SelfAttention(output_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        pooled, attention_weights = self.attention(outputs, lengths)
        logits = self.classifier(pooled)
        return logits, attention_weights

