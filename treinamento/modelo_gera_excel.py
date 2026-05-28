"""
modelo_gera_excel — carrega um modelo treinado, roda inferência num .ogg de
instrumento e salva o .xlsx parcial (aba 'info' + aba do instrumento).

Esta é a ponte entre o modelo `.pt` e o restante do pipeline:
    drums.ogg + BPM + modelo  →  drums_partial.xlsx

Para gerar o chart Clone Hero completo, o usuário roda quatro vezes (uma por
instrumento) e depois consolida os xlsx em um único `notes.xlsx` antes de
chamar `excel_to_midi.py`.

Atualmente só o modelo de DRUMS está implementado. Os de guitar/bass/vocals
seguirão a mesma assinatura quando existirem.

Uso (CLI):
    python modelo_gera_excel.py \\
        --audio drums.ogg \\
        --bpm 140 \\
        --instrument drums \\
        --model treinamento/checkpoint/drums/drums_crnn_best.pt \\
        --meta  treinamento/checkpoint/drums/drums_crnn_meta.pt \\
        --out   drums_partial.xlsx

Uso (API):
    from treinamento.modelo_gera_excel import infer
    infer(audio_path="drums.ogg", bpm=140, instrument="drums",
          model_path="best.pt", meta_path="meta.pt", out_xlsx="out.xlsx")
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


# Sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_features import (  # noqa: E402
    SUBDIV_PER_BEAT, TICKS_PER_BEAT,
    audio_duration_seconds, audio_to_grid_mel,
    normalize_mel, step_duration_seconds,
)
from drum_crnn import (  # noqa: E402
    DrumCRNN, LANES as DRUMS_LANES, LANE_NAMES as DRUMS_LANE_NAMES,
    N_LANES as DRUMS_N_LANES,
)
from bass_crnn import (  # noqa: E402
    BassRhythmCRNN,
    FRETS as RHYTHM_FRETS, FRET_NAMES as RHYTHM_FRET_NAMES,
    N_FRETS as RHYTHM_N_FRETS,
)
from guitar_crnn import (  # noqa: E402
    GuitarCRNN,
    FRETS as GUITAR_FRETS, FRET_NAMES as GUITAR_FRET_NAMES,
    N_FRETS as GUITAR_N_FRETS,
)
from vocals_crnn import (  # noqa: E402
    VocalsCRNN,
    LANE_NAMES as VOCALS_LANE_NAMES,
    N_LANES as VOCALS_N_LANES,
    CHART_OUTPUT_MIDI as VOCALS_CHART_MIDI,
)
from notes_xlsx import predictions_to_xlsx  # noqa: E402


log = logging.getLogger(__name__)


SUPPORTED_INSTRUMENTS = {"drums", "rhythm", "guitar", "vocals"}


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do modelo
# ─────────────────────────────────────────────────────────────────────────────
def _load_drums_model(model_path: Path, meta_path: Path,
                      device: torch.device) -> tuple[DrumCRNN, Dict[str, Any]]:
    meta = torch.load(str(meta_path), map_location="cpu")
    n_mels = int(meta.get("n_mels", 128))

    model = DrumCRNN(n_mels=n_mels).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _infer_drums(audio_path: Path,
                 bpm: float,
                 out_xlsx: Path,
                 model_path: Path,
                 meta_path: Path,
                 thresholds: Optional[Union[float, Dict[str, float], list]] = None,
                 n_steps_override: Optional[int] = None,
                 device: Optional[torch.device] = None) -> int:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = _load_drums_model(model_path, meta_path, device)
    mel_mean = float(meta["mel_mean"])
    mel_std  = float(meta["mel_std"])
    n_mels   = int(meta.get("n_mels", 128))

    if thresholds is None:
        thresholds = meta.get("optimal_thresholds", 0.5)

    # array [N_LANES] de thresholds
    if isinstance(thresholds, dict):
        thr = np.array([thresholds[DRUMS_LANE_NAMES[i]] for i in range(DRUMS_N_LANES)])
    elif isinstance(thresholds, (int, float)):
        thr = np.full(DRUMS_N_LANES, float(thresholds))
    else:
        thr = np.asarray(thresholds, dtype=np.float32)

    # n_steps
    audio_secs = audio_duration_seconds(audio_path)
    n_steps = n_steps_override or (int(audio_secs / step_duration_seconds(bpm)) + 1)

    # mel + forward
    mel = audio_to_grid_mel(audio_path, bpm, n_steps, n_mels=n_mels)
    mel = normalize_mel(mel, mel_mean, mel_std).unsqueeze(0).to(device)
    logits = model(mel)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    preds  = (probs >= thr[None, :]).astype(np.float32)

    # escreve xlsx
    idx_to_midi = {v: k for k, v in DRUMS_LANES.items()}
    n_events = predictions_to_xlsx(
        preds, instrument="drums",
        idx_to_midi=idx_to_midi,
        bpm=bpm,
        ticks_per_beat=TICKS_PER_BEAT,
        out_path=out_xlsx,
        subdiv_per_beat=SUBDIV_PER_BEAT,
    )
    return n_events


# ─────────────────────────────────────────────────────────────────────────────
# Rhythm / Bass
# ─────────────────────────────────────────────────────────────────────────────
def _load_rhythm_model(model_path: Path, meta_path: Path,
                       device: torch.device) -> tuple[BassRhythmCRNN, Dict[str, Any]]:
    meta = torch.load(str(meta_path), map_location="cpu")
    n_mels = int(meta.get("n_mels", 128))

    model = BassRhythmCRNN(n_mels=n_mels).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


@torch.no_grad()
def _infer_rhythm(audio_path: Path,
                  bpm: float,
                  out_xlsx: Path,
                  model_path: Path,
                  meta_path: Path,
                  thresholds: Optional[Union[float, Dict[str, float], list]] = None,
                  n_steps_override: Optional[int] = None,
                  device: Optional[torch.device] = None) -> int:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = _load_rhythm_model(model_path, meta_path, device)
    mel_mean = float(meta["mel_mean"])
    mel_std  = float(meta["mel_std"])
    n_mels   = int(meta.get("n_mels", 128))

    if thresholds is None:
        thresholds = meta.get("optimal_thresholds", 0.5)

    if isinstance(thresholds, dict):
        thr = np.array([thresholds[RHYTHM_FRET_NAMES[i]] for i in range(RHYTHM_N_FRETS)])
    elif isinstance(thresholds, (int, float)):
        thr = np.full(RHYTHM_N_FRETS, float(thresholds))
    else:
        thr = np.asarray(thresholds, dtype=np.float32)

    audio_secs = audio_duration_seconds(audio_path)
    n_steps = n_steps_override or (int(audio_secs / step_duration_seconds(bpm)) + 1)

    mel = audio_to_grid_mel(audio_path, bpm, n_steps, n_mels=n_mels)
    mel = normalize_mel(mel, mel_mean, mel_std).unsqueeze(0).to(device)
    logits = model(mel)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    preds  = (probs >= thr[None, :]).astype(np.float32)

    idx_to_midi = {v: k for k, v in RHYTHM_FRETS.items()}
    n_events = predictions_to_xlsx(
        preds, instrument="rhythm",
        idx_to_midi=idx_to_midi,
        bpm=bpm,
        ticks_per_beat=TICKS_PER_BEAT,
        out_path=out_xlsx,
        subdiv_per_beat=SUBDIV_PER_BEAT,
    )
    return n_events


# ─────────────────────────────────────────────────────────────────────────────
# Guitar (lead)
# ─────────────────────────────────────────────────────────────────────────────
def _load_guitar_model(model_path: Path, meta_path: Path,
                       device: torch.device) -> tuple[GuitarCRNN, Dict[str, Any]]:
    meta = torch.load(str(meta_path), map_location="cpu")
    n_mels = int(meta.get("n_mels", 128))

    model = GuitarCRNN(n_mels=n_mels).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


@torch.no_grad()
def _infer_guitar(audio_path: Path,
                  bpm: float,
                  out_xlsx: Path,
                  model_path: Path,
                  meta_path: Path,
                  thresholds: Optional[Union[float, Dict[str, float], list]] = None,
                  n_steps_override: Optional[int] = None,
                  device: Optional[torch.device] = None) -> int:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = _load_guitar_model(model_path, meta_path, device)
    mel_mean = float(meta["mel_mean"])
    mel_std  = float(meta["mel_std"])
    n_mels   = int(meta.get("n_mels", 128))

    if thresholds is None:
        thresholds = meta.get("optimal_thresholds", 0.5)

    if isinstance(thresholds, dict):
        thr = np.array([thresholds[GUITAR_FRET_NAMES[i]] for i in range(GUITAR_N_FRETS)])
    elif isinstance(thresholds, (int, float)):
        thr = np.full(GUITAR_N_FRETS, float(thresholds))
    else:
        thr = np.asarray(thresholds, dtype=np.float32)

    audio_secs = audio_duration_seconds(audio_path)
    n_steps = n_steps_override or (int(audio_secs / step_duration_seconds(bpm)) + 1)

    mel = audio_to_grid_mel(audio_path, bpm, n_steps, n_mels=n_mels)
    mel = normalize_mel(mel, mel_mean, mel_std).unsqueeze(0).to(device)
    logits = model(mel)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    preds  = (probs >= thr[None, :]).astype(np.float32)

    idx_to_midi = {v: k for k, v in GUITAR_FRETS.items()}
    n_events = predictions_to_xlsx(
        preds, instrument="guitar",
        idx_to_midi=idx_to_midi,
        bpm=bpm,
        ticks_per_beat=TICKS_PER_BEAT,
        out_path=out_xlsx,
        subdiv_per_beat=SUBDIV_PER_BEAT,
    )
    return n_events


# ─────────────────────────────────────────────────────────────────────────────
# Vocals (v1: single-lane onset detector)
# ─────────────────────────────────────────────────────────────────────────────
def _load_vocals_model(model_path: Path, meta_path: Path,
                       device: torch.device) -> tuple[VocalsCRNN, Dict[str, Any]]:
    meta = torch.load(str(meta_path), map_location="cpu")
    n_mels = int(meta.get("n_mels", 128))

    model = VocalsCRNN(n_mels=n_mels).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


@torch.no_grad()
def _infer_vocals(audio_path: Path,
                  bpm: float,
                  out_xlsx: Path,
                  model_path: Path,
                  meta_path: Path,
                  thresholds: Optional[Union[float, Dict[str, float], list]] = None,
                  n_steps_override: Optional[int] = None,
                  device: Optional[torch.device] = None) -> int:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = _load_vocals_model(model_path, meta_path, device)
    mel_mean = float(meta["mel_mean"])
    mel_std  = float(meta["mel_std"])
    n_mels   = int(meta.get("n_mels", 128))

    if thresholds is None:
        thresholds = meta.get("optimal_thresholds", 0.5)

    if isinstance(thresholds, dict):
        thr = np.array([thresholds[VOCALS_LANE_NAMES[i]] for i in range(VOCALS_N_LANES)])
    elif isinstance(thresholds, (int, float)):
        thr = np.full(VOCALS_N_LANES, float(thresholds))
    else:
        thr = np.asarray(thresholds, dtype=np.float32)

    audio_secs = audio_duration_seconds(audio_path)
    n_steps = n_steps_override or (int(audio_secs / step_duration_seconds(bpm)) + 1)

    mel = audio_to_grid_mel(audio_path, bpm, n_steps, n_mels=n_mels)
    mel = normalize_mel(mel, mel_mean, mel_std).unsqueeze(0).to(device)
    logits = model(mel)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    preds  = (probs >= thr[None, :]).astype(np.float32)

    # v1: única lane → escrevemos sempre a mesma nota MIDI no chart.
    idx_to_midi = {0: VOCALS_CHART_MIDI}
    n_events = predictions_to_xlsx(
        preds, instrument="vocals",
        idx_to_midi=idx_to_midi,
        bpm=bpm,
        ticks_per_beat=TICKS_PER_BEAT,
        out_path=out_xlsx,
        subdiv_per_beat=SUBDIV_PER_BEAT,
    )
    return n_events


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────
def infer(audio_path: Path | str,
          bpm: float,
          instrument: str,
          model_path: Path | str,
          meta_path: Path | str,
          out_xlsx: Path | str,
          thresholds: Optional[Union[float, Dict[str, float], list]] = None,
          n_steps_override: Optional[int] = None,
          device: Optional[torch.device] = None) -> Path:
    """Roda inferência e salva o xlsx parcial.

    Args:
        audio_path:       caminho do .ogg do instrumento
        bpm:              BPM da música (informado pelo usuário)
        instrument:       'drums' (futuramente: 'guitar', 'bass', 'vocals')
        model_path:       checkpoint do modelo (.pt)
        meta_path:        meta do treino com mel_mean/mel_std/optimal_thresholds
        out_xlsx:         arquivo .xlsx a gerar
        thresholds:       sobrescreve os thresholds salvos no meta
                          (None = usa meta, float = único para todas as lanes,
                          dict = por lane_name, list/array = vetor)
        n_steps_override: força um número específico de steps
        device:           força um device (default: cuda se disponível)

    Returns:
        Path do .xlsx gerado.
    """
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise NotImplementedError(
            f"Modelo de {instrument!r} ainda não implementado. "
            f"Suportados: {sorted(SUPPORTED_INSTRUMENTS)}"
        )

    audio_path = Path(audio_path)
    model_path = Path(model_path)
    meta_path  = Path(meta_path)
    out_xlsx   = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    if instrument == "drums":
        n_events = _infer_drums(audio_path, bpm, out_xlsx,
                                model_path, meta_path,
                                thresholds=thresholds,
                                n_steps_override=n_steps_override,
                                device=device)
    elif instrument == "rhythm":
        n_events = _infer_rhythm(audio_path, bpm, out_xlsx,
                                 model_path, meta_path,
                                 thresholds=thresholds,
                                 n_steps_override=n_steps_override,
                                 device=device)
    elif instrument == "guitar":
        n_events = _infer_guitar(audio_path, bpm, out_xlsx,
                                 model_path, meta_path,
                                 thresholds=thresholds,
                                 n_steps_override=n_steps_override,
                                 device=device)
    else:  # vocals
        n_events = _infer_vocals(audio_path, bpm, out_xlsx,
                                 model_path, meta_path,
                                 thresholds=thresholds,
                                 n_steps_override=n_steps_override,
                                 device=device)
    log.info("%s: %d eventos → %s", instrument, n_events, out_xlsx)
    return out_xlsx


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inferência de um modelo treinado: áudio + BPM → xlsx parcial.",
    )
    p.add_argument("--audio",      required=True, type=Path)
    p.add_argument("--bpm",        required=True, type=float)
    p.add_argument("--instrument", required=True,
                   choices=sorted(SUPPORTED_INSTRUMENTS))
    p.add_argument("--model",      required=True, type=Path, dest="model_path")
    p.add_argument("--meta",       required=True, type=Path, dest="meta_path")
    p.add_argument("--out",        required=True, type=Path, dest="out_xlsx")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Threshold único (sobrescreve thresholds por lane do meta)")
    p.add_argument("--device",     type=str, default=None,
                   help="'cpu' ou 'cuda' (default: cuda se disponível)")
    p.add_argument("--quiet",      action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device) if args.device else None
    out = infer(
        audio_path=args.audio,
        bpm=args.bpm,
        instrument=args.instrument,
        model_path=args.model_path,
        meta_path=args.meta_path,
        out_xlsx=args.out_xlsx,
        thresholds=args.threshold,
        device=device,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
