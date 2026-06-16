"""
onset_model — modelo de SELEÇÃO DE ONSET (pós-processamento) para o pipeline.

Treinado no experimento (experimentos/pos_processamento), supera a heurística de
densidade na escolha de quais onsets do basic-pitch viram nota (guitar/rhythm:
+9-11% de onset-F1 vs baseline). Aqui fica a versão de INFERÊNCIA usada pelo
`transcreve_basic_pitch`, com a MESMA construção de features do treino (fonte
única — o treino importa daqui).

Se o checkpoint não existir, o pipeline cai na heurística (`_thin_onsets`).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# Grid e janelas (devem casar com o treino)
GRID = 120          # 16avo (ticks) — = TICKS_PER_BEAT/4
WIN  = 512
HOP  = 256

STEM_ONEHOT = {"guitar": [1, 0, 0], "rhythm": [0, 1, 0], "vocals": [0, 0, 1]}
# thresholds default por stem (calibrados no val do treino completo, 986 músicas)
DEFAULT_THR = {"guitar": 0.60, "rhythm": 0.50, "vocals": 0.40}

_DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "treinamento" / "checkpoint" / "onset" / "onset_model.pt"
_CACHE: dict = {}


class OnsetBiLSTM(nn.Module):
    """BiLSTM de rotulagem de onset por step. (arquitetura igual à do treino)"""
    def __init__(self, n_feat: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.head(x).squeeze(-1)


def build_features(events: List[Tuple[int, int, int, int]],
                   stem: str, n_steps: int) -> np.ndarray:
    """Features por step a partir de eventos canônicos do pitch.

    events: lista de (note, start_tick, dur_ticks, velocity).
    Retorna feat[T, F] (T = n_steps). DEVE bater exatamente com o treino.
    """
    T = max(1, n_steps)
    onset = np.zeros(T); active = np.zeros(T); psum = np.zeros(T)
    pcnt = np.zeros(T); vmax = np.zeros(T)
    for note, st, dur, vel in events:
        s0 = st // GRID; s1 = (st + dur) // GRID
        if 0 <= s0 < T:
            onset[s0] += 1; vmax[s0] = max(vmax[s0], vel)
        for s in range(max(0, s0), min(s1 + 1, T)):
            active[s] += 1; psum[s] += note; pcnt[s] += 1
    meanp = np.where(pcnt > 0, psum / np.maximum(pcnt, 1), 60.0)
    dens = np.convolve(onset, np.ones(5), mode="same")
    steps = np.arange(T)
    onehot = STEM_ONEHOT[stem]
    feat = np.stack([
        np.minimum(onset, 4) / 4.0,
        np.minimum(active, 6) / 6.0,
        (meanp - 60.0) / 24.0,
        vmax / 127.0,
        np.minimum(dens, 10) / 10.0,
        (steps % 4) / 4.0,
        ((steps % 16) == 0).astype(float),
    ] + [np.full(T, v, dtype=float) for v in onehot], axis=1).astype(np.float32)
    return feat


def load_model(path: Optional[Path] = None) -> Optional[OnsetBiLSTM]:
    """Carrega o modelo (cacheado). Retorna None se o checkpoint não existir."""
    path = Path(path) if path else _DEFAULT_CKPT
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    if not path.exists():
        _CACHE[key] = None
        return None
    ckpt = torch.load(str(path), map_location="cpu")
    model = OnsetBiLSTM(int(ckpt["n_feat"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    _CACHE[key] = model
    return model


@torch.no_grad()
def predict_steps(model: OnsetBiLSTM,
                  events: List[Tuple[int, int, int, int]],
                  stem: str, n_steps: int,
                  thr: Optional[float] = None) -> List[int]:
    """Steps (índices de 16avo) selecionados como onsets pelo modelo."""
    if thr is None:
        thr = DEFAULT_THR.get(stem, 0.5)
    feat = build_features(events, stem, n_steps)
    T = feat.shape[0]
    prob = np.zeros(T); cov = np.zeros(T)
    for a in range(0, max(1, T - HOP), HOP):
        b = min(a + WIN, T)
        x = torch.from_numpy(feat[a:b]).unsqueeze(0)
        p = torch.sigmoid(model(x)).squeeze(0).numpy()[:b - a]
        prob[a:b] += p; cov[a:b] += 1
    prob = prob / np.maximum(cov, 1)
    return [int(s) for s in np.where(prob >= thr)[0]]
