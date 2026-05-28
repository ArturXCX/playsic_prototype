"""
vocals_crnn — modelo CRNN para predição de PART VOCALS (v1: onset detector).

Versão 1 simplificada: o modelo prediz uma única lane binária por step do
grid — "tem vocal ativo aqui?". Para o chart, cada onset detectado vira uma
nota MIDI 60 (C4) na track PART VOCALS, gerando um chart de karaokê monotone
mas jogável.

Razão da simplificação:
    - PART VOCALS no formato RB é pitched (uma nota afinada por sílaba) + lyric
      events. Modelar isso direito exige pitch tracking + alinhamento de
      lyrics, que extrapola o escopo de um CRNN multi-label.
    - 87% das notas vocais do dataset ficam em MIDI 36-84 e 10% são phrase
      markers (>84). Para v1, ignoramos os markers e tratamos qualquer pitch
      em 36-84 como "vocal ativo".
    - Esta v1 prova que o stem vocals.ogg → onsets funciona; uma v2 com
      24+1 classes de pitch pode ser construída em cima da mesma infra.

Nomenclatura do pipeline:
    INSTRUMENT = "vocals"      →  aba no notes.xlsx / chave em SHEET_TO_TRACK
    TRACK_NAME = "PART VOCALS" →  track do MIDI Clone Hero
    AUDIO_STEM = "vocals.ogg"  →  stem gerado pelo Demucs

Entrada da rede: mel-espectrograma normalizado, shape [B, N_MELS, T]
Saída:           logits, shape [B, T, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─── Identificação do instrumento no pipeline ────────────────────────────────
INSTRUMENT = "vocals"
TRACK_NAME = "PART VOCALS"
AUDIO_STEM = "vocals.ogg"

# ─── Faixa de pitches considerada "vocal ativo" ─────────────────────────────
# Pitches abaixo/acima são ignorados (markers, talk-overs, etc).
VOCAL_MIDI_MIN = 36   # C2
VOCAL_MIDI_MAX = 84   # C6

# ─── Lanes ──────────────────────────────────────────────────────────────────
# v1: uma única lane "vocal ativo". Mantemos a interface multi-label/lane
# pra reutilizar todo o resto do pipeline (compute_pos_weight, F1 por lane,
# section 7.5, etc).
LANES      = {-1: 0}              # placeholder — não usado diretamente (ver parse_vocals_events)
N_LANES    = 1
LANE_NAMES = ["VocalActive"]

# Nota MIDI que será escrita no chart pra cada onset detectado.
# Clone Hero precisa de uma nota afinada pra desenhar a barrinha; MIDI 60 = C4.
CHART_OUTPUT_MIDI = 60

# ─── Hiperparâmetros padrão da arquitetura ──────────────────────────────────
DEFAULT_N_MELS        = 128
DEFAULT_CONV_CHANNELS = (32, 64, 128)
DEFAULT_LSTM_HIDDEN   = 256
DEFAULT_LSTM_LAYERS   = 2
DEFAULT_DROPOUT       = 0.3


class VocalsCRNN(nn.Module):
    """CRNN binário para detecção de vocal activity por step.

    Arquitetura idêntica ao DrumCRNN / BassRhythmCRNN, com N_LANES=1.
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
