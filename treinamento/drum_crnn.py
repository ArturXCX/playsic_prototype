"""
drum_crnn — modelo CRNN para predição do chart de drums.

Define:
    - Constantes específicas de drums (LANES, LANE_NAMES, N_LANES, GAMEPLAY_MAX_MIDI)
    - Hiperparâmetros padrão da arquitetura
    - Classe `DrumCRNN`

Quando os modelos de guitar/bass/vocals existirem, ficarão como módulos
irmãos (`guitar_crnn.py`, `bass_crnn.py`, `vocals_crnn.py`) seguindo a
mesma convenção.

Entrada da rede: mel-espectrograma normalizado, shape `[B, N_MELS, T]`
Saída:           logits, shape `[B, T, N_LANES]`

Para inferência, aplicar `torch.sigmoid` à saída e thresholdar por lane.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─── Mapeamento MIDI → lane do Rock Band ────────────────────────────────────
LANES = {
    24: 0,   # Kick    (C1)
    26: 1,   # Snare   (D1)
    27: 2,   # Yellow  (D#1) — hi-hat
    30: 3,   # Blue    (F#1) — tom blue
    31: 4,   # Green   (G1)  — crash
}
N_LANES    = 5
LANE_NAMES = ["Kick", "Snare", "Yellow", "Blue", "Green"]

# MIDI < 36 = lanes jogáveis. MIDI ≥ 60 = mirror visual (ignorar no target).
GAMEPLAY_MAX_MIDI = 36

# ─── Hiperparâmetros padrão da arquitetura ──────────────────────────────────
DEFAULT_N_MELS        = 128
DEFAULT_CONV_CHANNELS = (32, 64, 128)
DEFAULT_LSTM_HIDDEN   = 256
DEFAULT_LSTM_LAYERS   = 2
DEFAULT_DROPOUT       = 0.3


class DrumCRNN(nn.Module):
    """CRNN para predição multi-label de eventos de drums por step do grid.

    Arquitetura:
        - 3 blocos CNN (32 → 64 → 128 filtros), pool 2 só no eixo de frequência
        - BiLSTM 2 camadas, hidden=256
        - MLP head: Linear(2H, H) → ReLU → Dropout → Linear(H, N_LANES)
    """

    def __init__(self,
                 n_mels: int = DEFAULT_N_MELS,
                 n_lanes: int = N_LANES,
                 conv_channels=DEFAULT_CONV_CHANNELS,
                 lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
                 lstm_layers: int = DEFAULT_LSTM_LAYERS,
                 dropout: float = DEFAULT_DROPOUT):
        super().__init__()

        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),     # pool só em freq
                nn.Dropout2d(dropout * 0.5),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        freq_after  = n_mels // (2 ** len(conv_channels))
        cnn_out_dim = conv_channels[-1] * freq_after

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_lanes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_MELS, T]
        x = x.unsqueeze(1)         # [B, 1, N_MELS, T]
        x = self.cnn(x)            # [B, C, F', T]
        b, c, fp, tl = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, tl, c * fp)  # [B, T, C*F']
        x, _ = self.lstm(x)        # [B, T, 2H]
        return self.head(x)        # [B, T, N_LANES]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
