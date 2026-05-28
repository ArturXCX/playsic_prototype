"""
audio_features — extração do mel-espectrograma alinhado ao grid musical.

A grade musical é definida por:
    - BPM (informado pelo usuário ou lido do notes.xlsx)
    - SUBDIV_PER_BEAT (default 4 = semicolcheia)

O hop do espectrograma é calculado DINAMICAMENTE por música, de modo que
cada frame de mel corresponda exatamente a 1 step do grid.

Este módulo é compartilhado entre o notebook de treinamento e o script
de inferência (`modelo_gera_excel.py`).
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


# ─── Áudio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 22050
N_MELS_DEFAULT  = 128
N_FFT           = 2048
WIN_LENGTH      = 1024
FMIN            = 30.0
FMAX            = SAMPLE_RATE / 2

# ─── Grade musical ──────────────────────────────────────────────────────────
SUBDIV_PER_BEAT = 4        # 4 = semicolcheia (1/16)
TICKS_PER_BEAT  = 480
TICKS_PER_STEP  = TICKS_PER_BEAT // SUBDIV_PER_BEAT


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo do grid
# ─────────────────────────────────────────────────────────────────────────────
def step_duration_seconds(bpm: float, subdiv_per_beat: int = SUBDIV_PER_BEAT) -> float:
    """Quanto dura 1 step do grid em segundos."""
    return 60.0 / (bpm * subdiv_per_beat)


def hop_for_bpm(bpm: float, sr: int = SAMPLE_RATE) -> int:
    """Hop (em samples) para que 1 frame de mel = 1 step do grid."""
    return int(round(step_duration_seconds(bpm) * sr))


def audio_duration_seconds(audio_path: Path | str) -> float:
    """Duração do áudio em segundos, sem carregar amostras."""
    info = sf.info(str(audio_path))
    return info.frames / info.samplerate


# ─────────────────────────────────────────────────────────────────────────────
# Mel-espectrograma
# ─────────────────────────────────────────────────────────────────────────────
def audio_to_grid_mel(audio_path: Path | str,
                      bpm: float,
                      n_steps: int,
                      n_mels: int = N_MELS_DEFAULT,
                      sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Carrega o áudio e devolve um mel-espectrograma `[N_MELS, n_steps]`.

    O hop é escolhido de modo que cada frame corresponda a 1 step do grid.
    O resultado é a amplitude em escala log (dB-like).
    """
    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate

    hop = hop_for_bpm(bpm, sr)

    mel_spec = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=hop,
        n_mels=n_mels,
        f_min=FMIN,
        f_max=FMAX,
        power=2.0,
    )(wav)
    mel = torch.log(mel_spec + 1e-6).squeeze(0)  # [N_MELS, T]

    t = mel.shape[1]
    if t >= n_steps:
        mel = mel[:, :n_steps]
    else:
        mel = F.pad(mel, (0, n_steps - t), value=mel.min().item())
    return mel


def normalize_mel(mel: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (mel - mean) / (std + 1e-6)
