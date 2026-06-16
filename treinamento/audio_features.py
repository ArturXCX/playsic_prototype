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

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
# Nota: torchaudio.load() e .save() passam por torchcodec no 2.11+ e falham
# no Windows sem FFmpeg shared DLLs.  Usamos soundfile para carregar áudio
# e mantemos torchaudio apenas para functional.resample e MelSpectrogram
# (computações PyTorch puras, sem FFmpeg).


# ─── Áudio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 22050
N_MELS_DEFAULT  = 128
N_FFT           = 2048
WIN_LENGTH      = 1024
FMIN            = 30.0
FMAX            = SAMPLE_RATE / 2
# Hop fino-alvo (samples) da extração mel sub-step. ~23ms @ 22050Hz, menor que
# WIN_LENGTH → janelas com overlap → cobertura total do áudio. Cada step do grid
# agrega vários desses sub-frames via max-pool (ver audio_to_grid_mel).
TARGET_FINE_HOP = 512

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

    Cada coluna corresponde a 1 step do grid, obtida agregando vários sub-frames
    STFT de hop fino via **max-pooling** — não mais 1 único frame por step.
    Isso corrige dois problemas do esquema antigo (1 frame/step):

      1. Cobertura: com hop = duração do step (~2756 samples @120BPM) e
         WIN_LENGTH=1024, ~60% do áudio ficava FORA de qualquer janela de
         análise; o transiente de onset caía nos buracos e sumia.
      2. Localização: 1 frame não localiza o ataque dentro do step.

    Com hop fino (~512 samples) as janelas têm overlap (cobertura total) e o max
    sobre os k sub-frames de cada step preserva o pico de energia do ataque onde
    quer que ele caia. Saída em log (dB-like), shape exatamente `[N_MELS, n_steps]`.
    """
    # Carrega via soundfile (sem torchcodec) e converte para tensor [C, T]
    data, sr = sf.read(str(audio_path), always_2d=True)   # [frames, channels]
    wav = torch.from_numpy(data.T.astype(np.float32))      # [channels, frames]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)  # puro PyTorch
        sr = sample_rate

    hop = hop_for_bpm(bpm, sr)
    # k = sub-frames por step (>=2); hop fino garante overlap entre janelas.
    k = max(2, int(round(hop / TARGET_FINE_HOP)))
    hop_fine = max(1, hop // k)

    mel_pow = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=hop_fine,
        n_mels=n_mels,
        f_min=FMIN,
        f_max=FMAX,
        power=2.0,
    )(wav).squeeze(0)        # [N_MELS, T_fine] em potência (linear)

    # Agrega cada bloco de k sub-frames num único step via max (preserva onsets).
    need   = n_steps * k
    t_fine = mel_pow.shape[1]
    if t_fine < need:
        mel_pow = F.pad(mel_pow, (0, need - t_fine), value=0.0)
    else:
        mel_pow = mel_pow[:, :need]
    mel_pow = mel_pow.reshape(n_mels, n_steps, k).amax(dim=2)  # [N_MELS, n_steps]

    return torch.log(mel_pow + 1e-6)


def normalize_mel(mel: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (mel - mean) / (std + 1e-6)
