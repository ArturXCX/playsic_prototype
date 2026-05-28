"""
bass_crnn — modelo CRNN para predição de bass/rhythm guitar (5 frets).

Mesma arquitetura do DrumCRNN, adaptada para as 5 casas do baixo:
    Green=96, Red=97, Yellow=98, Blue=99, Orange=100  (notas Expert Clone Hero)

Nomenclatura do pipeline:
    INSTRUMENT = "rhythm"     →  aba no notes.xlsx / chave em SHEET_TO_TRACK
    TRACK_NAME = "PART BASS"  →  track do MIDI Clone Hero
    AUDIO_STEM = "rhythm.ogg" →  stem gerado pelo Demucs

Entrada da rede: mel-espectrograma normalizado, shape [B, N_MELS, T]
Saída:           logits, shape [B, T, N_FRETS]

Para inferência, aplicar `torch.sigmoid` à saída e thresholdar por fret.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─── Identificação do instrumento no pipeline ────────────────────────────────
INSTRUMENT = "rhythm"       # nome da aba no xlsx e chave em excel_to_midi.SHEET_TO_TRACK
TRACK_NAME = "PART BASS"    # nome da track MIDI Clone Hero
AUDIO_STEM = "rhythm.ogg"   # stem gerado por separa_audio.py

# ─── Mapeamento MIDI Expert → fret index ────────────────────────────────────
# Notas Expert de guitar/bass no formato Clone Hero (Rock Band):
#   Easy 60-64 | Medium 72-76 | Hard 84-88 | Expert 96-100
# O modelo é treinado e faz inferência sobre as notas Expert.
# A expansão para as 4 dificuldades ocorre em excel_to_midi.py.
FRETS = {
    96: 0,   # Green
    97: 1,   # Red
    98: 2,   # Yellow
    99: 3,   # Blue
    100: 4,  # Orange
}
N_FRETS    = 5
FRET_NAMES = ["Green", "Red", "Yellow", "Blue", "Orange"]

# ─── Hiperparâmetros padrão da arquitetura ──────────────────────────────────
DEFAULT_N_MELS        = 128
DEFAULT_CONV_CHANNELS = (32, 64, 128)
DEFAULT_LSTM_HIDDEN   = 256
DEFAULT_LSTM_LAYERS   = 2
DEFAULT_DROPOUT       = 0.3


class BassRhythmCRNN(nn.Module):
    """CRNN para predição multi-label de eventos de bass/rhythm guitar por step do grid.

    Arquitetura idêntica ao DrumCRNN:
        - 3 blocos CNN (32 → 64 → 128 filtros), MaxPool só no eixo de frequência
        - BiLSTM 2 camadas, hidden=256
        - MLP head: Linear(2H, H) → ReLU → Dropout → Linear(H, N_FRETS)
    """

    def __init__(self,
                 n_mels: int = DEFAULT_N_MELS,
                 n_frets: int = N_FRETS,
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
            nn.Linear(lstm_hidden, n_frets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_MELS, T]
        x = x.unsqueeze(1)         # [B, 1, N_MELS, T]
        x = self.cnn(x)            # [B, C, F', T]
        b, c, fp, tl = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, tl, c * fp)  # [B, T, C*F']
        x, _ = self.lstm(x)        # [B, T, 2H]
        return self.head(x)        # [B, T, N_FRETS]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
