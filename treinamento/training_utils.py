"""
training_utils — utilitários compartilhados pelo notebook de treino.

Contém o que NÃO precisa ser visível para inferência:
    - SongData (mel + target + metadata da música)
    - list_song_dirs / preprocess_song
    - compute_mel_stats
    - SpecAugment + DrumChartDataset

Importa de audio_features.py e notes_xlsx.py (sibling modules).
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# Sibling imports (sys.path local, evita exigir __init__.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_features import (  # noqa: E402
    SUBDIV_PER_BEAT, audio_duration_seconds, audio_to_grid_mel,
    normalize_mel, step_duration_seconds,
)
from notes_xlsx import (  # noqa: E402
    events_to_target_matrix, parse_drum_events, parse_info_sheet,
    parse_rhythm_events,
)


# ─── Defaults de treino ─────────────────────────────────────────────────────
CHUNK_STEPS         = 512        # ≈ 32 beats
CROPS_PER_SONG      = 6          # crops aleatórios por música/epoch (treino)
AUG_PROB            = 0.5
SPEC_AUG_FREQ_MASKS = 2
SPEC_AUG_FREQ_WIDTH = 8
SPEC_AUG_TIME_MASKS = 2
SPEC_AUG_TIME_WIDTH = 16


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura por música
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SongData:
    song_id: str
    folder:  Path
    mel:     torch.Tensor      # [N_MELS, n_steps]
    target:  torch.Tensor      # [n_steps, N_LANES]
    bpm:     float
    n_steps: int


# ─────────────────────────────────────────────────────────────────────────────
# Descoberta + pré-processamento
# ─────────────────────────────────────────────────────────────────────────────
def list_song_dirs(root: Path, audio_stem: str = "drums.ogg") -> List[Path]:
    """Cada sub-pasta com notes.xlsx + <audio_stem> é uma música."""
    return sorted([d for d in root.iterdir()
                   if d.is_dir()
                   and (d / "notes.xlsx").exists()
                   and (d / audio_stem).exists()])


def preprocess_song(song_dir: Path,
                    lanes_map: Dict[int, int],
                    n_lanes: int,
                    n_mels: int,
                    audio_stem: str = "drums.ogg",
                    parse_events_fn=None,
                    gameplay_max_midi: int = 36) -> Optional[SongData]:
    """Carrega uma música → SongData (mel + target).

    Args:
        song_dir:         pasta da música no dataset
        lanes_map:        dict MIDI_note → lane_index (ex: LANES para drums,
                          FRETS para bass)
        n_lanes:          número de lanes/frets do instrumento
        n_mels:           número de bins mel do espectrograma
        audio_stem:       arquivo de áudio a usar (ex: 'drums.ogg', 'rhythm.ogg')
        parse_events_fn:  callable(xlsx_path, lanes_map) → dict com 'events' e
                          'duration_ticks'. Se None, usa parse_drum_events
                          (compatibilidade retroativa).
        gameplay_max_midi: usado apenas quando parse_events_fn é None (drums).

    Retorna None se o processamento falhar.
    """
    if parse_events_fn is None:
        _fn = lambda xlsx, lm: parse_drum_events(xlsx, lm, gameplay_max_midi)
    else:
        _fn = parse_events_fn

    try:
        info           = parse_info_sheet(song_dir / "notes.xlsx")
        bpm            = info["bpm"]
        tpb            = info["ticks_per_beat"]
        ticks_per_step = tpb // SUBDIV_PER_BEAT

        data = _fn(song_dir / "notes.xlsx", lanes_map)

        n_steps_from_events = (data["duration_ticks"] // ticks_per_step) + 4
        audio_secs = audio_duration_seconds(song_dir / audio_stem)
        n_steps_from_audio  = int(audio_secs / step_duration_seconds(bpm)) + 1
        n_steps = min(n_steps_from_events, n_steps_from_audio)
        if n_steps <= 0:
            return None

        target = events_to_target_matrix(data["events"],
                                         n_steps, ticks_per_step, n_lanes)
        mel = audio_to_grid_mel(song_dir / audio_stem, bpm, n_steps, n_mels=n_mels)

        return SongData(
            song_id=song_dir.name,
            folder=song_dir,
            mel=mel,
            target=torch.from_numpy(target),
            bpm=bpm,
            n_steps=n_steps,
        )
    except Exception as e:
        print(f"[WARN] falha em {song_dir.name}: {e}")
        return None


def compute_mel_stats(songs: List[SongData]) -> Tuple[float, float]:
    """Mean/std globais dos mels (use só o train set para não vazar val)."""
    acc_sum, acc_sq, n = 0.0, 0.0, 0
    for s in songs:
        x = s.mel
        acc_sum += x.sum().item()
        acc_sq  += (x ** 2).sum().item()
        n       += x.numel()
    mean = acc_sum / n
    var  = acc_sq / n - mean ** 2
    return mean, math.sqrt(max(var, 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def spec_augment(mel: torch.Tensor,
                 freq_masks: int = SPEC_AUG_FREQ_MASKS,
                 freq_width: int = SPEC_AUG_FREQ_WIDTH,
                 time_masks: int = SPEC_AUG_TIME_MASKS,
                 time_width: int = SPEC_AUG_TIME_WIDTH) -> torch.Tensor:
    """Freq + time masking. Input shape: [N_MELS, T]."""
    mel = mel.clone()
    n_mels, t_len = mel.shape
    fill = mel.min().item()

    for _ in range(freq_masks):
        w = random.randint(0, freq_width)
        if w == 0:
            continue
        f0 = random.randint(0, n_mels - w)
        mel[f0:f0 + w, :] = fill

    for _ in range(time_masks):
        w = random.randint(0, time_width)
        if w == 0:
            continue
        t0 = random.randint(0, t_len - w)
        mel[:, t0:t0 + w] = fill

    return mel


# ─────────────────────────────────────────────────────────────────────────────
# Dataset PyTorch
# ─────────────────────────────────────────────────────────────────────────────
class DrumChartDataset(Dataset):
    """Crops aleatórios de `chunk_steps` por música. Sem vazamento entre splits.

    No treino (`augment=True`) cada música rende `crops_per_song` crops distintos
    por epoch — sem isso, um dataset de N músicas dava só N gradient steps/epoch
    (com chunk=512 ≈ 32 beats, a maior parte de cada música nunca era vista numa
    dada epoch). Na validação (`augment=False`) mantém-se 1 crop por música para
    métrica estável.
    """

    def __init__(self, songs: List[SongData],
                 mel_mean: float, mel_std: float,
                 chunk_steps: int = CHUNK_STEPS,
                 augment: bool = False,
                 aug_prob: float = AUG_PROB,
                 crops_per_song: int = CROPS_PER_SONG):
        self.songs          = songs
        self.mel_mean       = mel_mean
        self.mel_std        = mel_std
        self.chunk_steps    = chunk_steps
        self.augment        = augment
        self.aug_prob       = aug_prob
        self.crops_per_song = crops_per_song if augment else 1

    def __len__(self):
        return len(self.songs) * self.crops_per_song

    def __getitem__(self, idx):
        s = self.songs[idx % len(self.songs)]
        mel, target, n = s.mel, s.target, s.n_steps
        if n > self.chunk_steps:
            start = random.randint(0, n - self.chunk_steps)
            mel_c = mel[:, start:start + self.chunk_steps]
            tgt_c = target[start:start + self.chunk_steps]
        else:
            pad = self.chunk_steps - n
            mel_c = F.pad(mel, (0, pad), value=mel.min().item())
            tgt_c = F.pad(target, (0, 0, 0, pad), value=0.0)

        mel_c = normalize_mel(mel_c, self.mel_mean, self.mel_std)
        if self.augment and random.random() < self.aug_prob:
            mel_c = spec_augment(mel_c)
        return mel_c, tgt_c
