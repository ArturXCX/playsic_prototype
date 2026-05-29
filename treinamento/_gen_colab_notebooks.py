"""
Gera os 4 notebooks de treinamento para Google Colab:
  colab_treino_drums.ipynb
  colab_treino_bass.ipynb
  colab_treino_guitar.ipynb
  colab_treino_vocals.ipynb

Execute:  python treinamento/_gen_colab_notebooks.py

Diferenças em relação aos notebooks locais:
  - Célula 0 de setup: pip install + git clone do repo + mount Drive
  - Paths apontam para o Google Drive (DRIVE_ROOT configurável)
  - Seção 9 (teste end-to-end chart/preview) removida
  - Célula final com files.download() do modelo treinado
  - BATCH_SIZE=32 (T4/A100 do Colab aguenta mais)
  - num_workers=2 (funciona bem no Colab)
"""
from __future__ import annotations

import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuração por instrumento
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "drums": {
        "title":        "drums",
        "crnn_import":  (
            "from drum_crnn import (\n"
            "    DrumCRNN, LANES, LANE_NAMES, N_LANES, GAMEPLAY_MAX_MIDI, count_params,\n"
            ")"
        ),
        "extra_imports": "",
        "notes_import": "from notes_xlsx import predictions_to_xlsx",
        "model_class":  "DrumCRNN",
        "n_lanes_var":  "N_LANES",
        "lane_names_var": "LANE_NAMES",
        "pos_weight_init": "torch.zeros(N_LANES)",
        "zip_names":    "LANE_NAMES",
        "ckpt_prefix":  "drums_crnn",
        "ckpt_subdir":  "drums",
        "audio_stem":   "drums.ogg",
        "list_song_dirs_call": 'list_song_dirs(DATASET_DIR)',
        "preprocess_call": (
            "s = preprocess_song(d, lanes_map=LANES, n_lanes=N_LANES,\n"
            "                        n_mels=N_MELS, gameplay_max_midi=GAMEPLAY_MAX_MIDI)"
        ),
        "skipped_label": "sem drums.ogg ou chart",
        "min_note_check": False,
        "epoch_log": (
            "f\"K={v['f1_Kick']:.3f} S={v['f1_Snare']:.3f} \"\n"
            "          f\"Y={v['f1_Yellow']:.3f} B={v['f1_Blue']:.3f} G={v['f1_Green']:.3f}\""
        ),
        "lane_loop_var": "lane",
        "lane_range_var": "N_LANES",
        "f1_key_fmt":   "f\"f1_{LANE_NAMES[lane]}\"",
        "piano_roll_yticks": "LANE_NAMES",
        "sec75_note": "",
    },
    "bass": {
        "title":        "bass",
        "crnn_import":  (
            "from bass_crnn import (\n"
            "    BassRhythmCRNN, FRETS, FRET_NAMES, N_FRETS, AUDIO_STEM, count_params,\n"
            ")"
        ),
        "extra_imports": "LANE_NAMES = FRET_NAMES\nN_LANES    = N_FRETS\n",
        "notes_import": "from notes_xlsx import parse_rhythm_events, predictions_to_xlsx",
        "model_class":  "BassRhythmCRNN",
        "n_lanes_var":  "N_FRETS",
        "lane_names_var": "FRET_NAMES",
        "pos_weight_init": "torch.zeros(N_FRETS)",
        "zip_names":    "FRET_NAMES",
        "ckpt_prefix":  "bass_crnn",
        "ckpt_subdir":  "bass",
        "audio_stem":   "rhythm.ogg",
        "list_song_dirs_call": 'list_song_dirs(DATASET_DIR, audio_stem=AUDIO_STEM)',
        "preprocess_call": (
            "s = preprocess_song(d, lanes_map=FRETS, n_lanes=N_FRETS,\n"
            "                        n_mels=N_MELS, audio_stem=AUDIO_STEM,\n"
            "                        parse_events_fn=parse_rhythm_events)"
        ),
        "skipped_label": "sem bass Expert",
        "min_note_check": True,
        "epoch_log": (
            "f\"G={v['f1_Green']:.3f} R={v['f1_Red']:.3f} \"\n"
            "          f\"Y={v['f1_Yellow']:.3f} B={v['f1_Blue']:.3f} O={v['f1_Orange']:.3f}\""
        ),
        "lane_loop_var": "fret",
        "lane_range_var": "N_FRETS",
        "f1_key_fmt":   "f\"f1_{FRET_NAMES[fret]}\"",
        "piano_roll_yticks": "FRET_NAMES",
        "sec75_note": "",
    },
    "guitar": {
        "title":        "guitar",
        "crnn_import":  (
            "from guitar_crnn import (\n"
            "    GuitarCRNN, FRETS, FRET_NAMES, N_FRETS, AUDIO_STEM, count_params,\n"
            ")"
        ),
        "extra_imports": "LANE_NAMES = FRET_NAMES\nN_LANES    = N_FRETS\n",
        "notes_import": "from notes_xlsx import parse_guitar_events, predictions_to_xlsx",
        "model_class":  "GuitarCRNN",
        "n_lanes_var":  "N_FRETS",
        "lane_names_var": "FRET_NAMES",
        "pos_weight_init": "torch.zeros(N_FRETS)",
        "zip_names":    "FRET_NAMES",
        "ckpt_prefix":  "guitar_crnn",
        "ckpt_subdir":  "guitar",
        "audio_stem":   "guitar.ogg",
        "list_song_dirs_call": 'list_song_dirs(DATASET_DIR, audio_stem=AUDIO_STEM)',
        "preprocess_call": (
            "s = preprocess_song(d, lanes_map=FRETS, n_lanes=N_FRETS,\n"
            "                        n_mels=N_MELS, audio_stem=AUDIO_STEM,\n"
            "                        parse_events_fn=parse_guitar_events)"
        ),
        "skipped_label": "sem guitar Expert",
        "min_note_check": True,
        "epoch_log": (
            "f\"G={v['f1_Green']:.3f} R={v['f1_Red']:.3f} \"\n"
            "          f\"Y={v['f1_Yellow']:.3f} B={v['f1_Blue']:.3f} O={v['f1_Orange']:.3f}\""
        ),
        "lane_loop_var": "fret",
        "lane_range_var": "N_FRETS",
        "f1_key_fmt":   "f\"f1_{FRET_NAMES[fret]}\"",
        "piano_roll_yticks": "FRET_NAMES",
        "sec75_note": "",
    },
    "vocals": {
        "title":        "vocals",
        "crnn_import":  (
            "from vocals_crnn import (\n"
            "    VocalsCRNN, LANES, LANE_NAMES, N_LANES, AUDIO_STEM,\n"
            "    VOCAL_MIDI_MIN, VOCAL_MIDI_MAX, count_params,\n"
            ")"
        ),
        "extra_imports": "",
        "notes_import": "from notes_xlsx import parse_vocals_events, predictions_to_xlsx",
        "model_class":  "VocalsCRNN",
        "n_lanes_var":  "N_LANES",
        "lane_names_var": "LANE_NAMES",
        "pos_weight_init": "torch.zeros(N_LANES)",
        "zip_names":    "LANE_NAMES",
        "ckpt_prefix":  "vocals_crnn",
        "ckpt_subdir":  "vocals",
        "audio_stem":   "vocals.ogg",
        "list_song_dirs_call": 'list_song_dirs(DATASET_DIR, audio_stem=AUDIO_STEM)',
        "preprocess_call": (
            "s = preprocess_song(d, lanes_map=LANES, n_lanes=N_LANES,\n"
            "                        n_mels=N_MELS, audio_stem=AUDIO_STEM,\n"
            "                        parse_events_fn=parse_vocals_events)"
        ),
        "skipped_label": "sem vocals no range",
        "min_note_check": True,
        "epoch_log": "f\"V={v['f1_VocalActive']:.3f}\"",
        "lane_loop_var": "lane",
        "lane_range_var": "N_LANES",
        "f1_key_fmt":   "f\"f1_{LANE_NAMES[lane]}\"",
        "piano_roll_yticks": "LANE_NAMES",
        "sec75_note": (
            "\n> **Nota vocals v1**: com `N_LANES=1` o modelo prevê apenas "
            '"tem vocal ativo?", logo `estrito ≡ onset`. '
            "A contagem ainda dá informação útil (densidade vocal)."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_nb():
    return {"cells": [], "metadata": {
        "colab": {"name": "", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    }, "nbformat": 4, "nbformat_minor": 5}


def md(nb, text):
    nb["cells"].append({
        "cell_type": "markdown", "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(nb, text):
    nb["cells"].append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Gerador principal
# ─────────────────────────────────────────────────────────────────────────────
def generate_notebook(cfg: dict) -> dict:
    nb  = make_nb()
    ins = cfg["title"]
    lv  = cfg["lane_loop_var"]
    lr  = cfg["lane_range_var"]
    fk  = cfg["f1_key_fmt"]
    zn  = cfg["zip_names"]

    # ── Título ────────────────────────────────────────────────────────────────
    md(nb, f"""# Treinamento — modelo de {ins} (Google Colab)

Notebook Colab: treina `{cfg['model_class']}` com dataset no Google Drive,
avalia no val set e oferece o download do modelo treinado.

**Não inclui** o teste end-to-end de chart/preview — rode `main.py`
localmente depois de baixar o `.pt`.
""")

    # ── Célula 0: Setup Colab ─────────────────────────────────────────────────
    md(nb, "## 0. Setup (execute apenas uma vez por sessão)")
    code(nb, """\
# Instala dependências
%pip install -q librosa mido openpyxl tqdm

# Clona o repositório Playsic (contém os módulos de treinamento)
import os, sys
if not os.path.exists('/content/playsic'):
    os.system('git clone https://github.com/ArturXCX/playsic_prototype.git /content/playsic')
    print("Repositório clonado.")
else:
    print("Repositório já presente.")

sys.path.insert(0, '/content/playsic')
sys.path.insert(0, '/content/playsic/treinamento')
sys.path.insert(0, '/content/playsic/processamento/midi_excel')

# Monta o Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("Drive montado.")
""")

    # ── Célula 1: GPU check ───────────────────────────────────────────────────
    md(nb, "## 1. Verificação de GPU")
    code(nb, """\
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM  : {props.total_memory / 1e9:.1f} GB")
else:
    print("\\n⚠  GPU não encontrada!")
    print("   Vá em: Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4")
""")

    # ── Célula 2: Paths + Hiperparâmetros ─────────────────────────────────────
    md(nb, "## 2. Paths e hiperparâmetros")
    code(nb, f"""\
from pathlib import Path
import random, numpy as np

# ── Ajuste se o nome da sua pasta no Drive for diferente ────────────────────
DRIVE_ROOT  = Path('/content/drive/MyDrive/playsic')  # ← pasta raiz no Drive
DATASET_DIR = DRIVE_ROOT / 'dataset'                   # ← onde está o dataset
CKPT_DIR    = DRIVE_ROOT / 'checkpoint' / '{ins}'      # ← onde salvar o modelo
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH   = CKPT_DIR / '{cfg["ckpt_prefix"]}_best.pt'
META_PATH   = CKPT_DIR / '{cfg["ckpt_prefix"]}_meta.pt'

assert DATASET_DIR.is_dir(), (
    f"Dataset não encontrado: {{DATASET_DIR}}\\n"
    "Ajuste DRIVE_ROOT acima para corresponder ao nome da sua pasta no Drive."
)

# ── Hiperparâmetros ──────────────────────────────────────────────────────────
SEED           = 42
N_MELS         = 128
BATCH_SIZE     = 32    # T4 16 GB aguenta; reduza para 16 se der OOM
EPOCHS         = 80
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
PATIENCE       = 15
POS_WEIGHT_CAP = 50.0
VAL_FRAC       = 0.15
MIN_NOTE_COUNT = 10

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Dataset : {{DATASET_DIR}}")
print(f"Ckpt    : {{CKPT_PATH}}")
print(f"Meta    : {{META_PATH}}")
""")

    # ── Célula 3: Imports dos módulos do projeto ───────────────────────────────
    md(nb, "## 3. Imports")
    extra = cfg["extra_imports"]
    vocals_extra = (
        "\nprint(f\"Vocal pitch range filtrado: MIDI {VOCAL_MIDI_MIN} – {VOCAL_MIDI_MAX}\")\n"
        "print(f\"N_LANES = {N_LANES}  ({LANE_NAMES})\")"
        if ins == "vocals" else ""
    )
    code(nb, f"""\
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

{cfg['crnn_import']}
{extra}from audio_features import (
    SAMPLE_RATE, SUBDIV_PER_BEAT, TICKS_PER_BEAT,
    audio_to_grid_mel, normalize_mel, step_duration_seconds,
)
{cfg['notes_import']}
from training_utils import (
    SongData, list_song_dirs, preprocess_song,
    compute_mel_stats, DrumChartDataset, CHUNK_STEPS,
){vocals_extra}
""")

    # ── Célula 4: Pré-processamento ────────────────────────────────────────────
    md(nb, "## 4. Pré-processamento do dataset")
    audio_stem_line = f'"{cfg["audio_stem"]}"'
    min_note_block = ""
    if cfg["min_note_check"]:
        min_note_block = (
            f"\n    if s.target.sum() < MIN_NOTE_COUNT:\n"
            f"        skipped_empty += 1   # {cfg['skipped_label']}\n"
            f"        continue"
        )
        songs_block = (
            "songs, skipped_none, skipped_empty = [], 0, 0\n"
            "for d in tqdm(song_dirs):\n"
            f"    {cfg['preprocess_call']}\n"
            "    if s is None:\n"
            "        skipped_none += 1\n"
            "        continue"
            f"{min_note_block}\n"
            "    songs.append(s)\n\n"
            "print(f\"OK: {len(songs)} músicas pré-processadas\")\n"
            "print(f\"   Puladas por erro:        {skipped_none}\")\n"
            f"print(f\"   Puladas ({cfg['skipped_label']}): {{skipped_empty}}\")"
        )
    else:
        songs_block = (
            "songs = []\n"
            "for d in tqdm(song_dirs):\n"
            f"    {cfg['preprocess_call']}\n"
            "    if s is not None:\n"
            "        songs.append(s)\n\n"
            "print(f\"OK: {len(songs)} músicas pré-processadas\")"
        )

    list_call = cfg["list_song_dirs_call"]
    code(nb, f"""\
print(f"Procurando músicas em: {{DATASET_DIR}}")
song_dirs = {list_call}
print(f"Encontradas {{len(song_dirs)}} músicas com {audio_stem_line} + notes.xlsx")
assert len(song_dirs) > 0, "Nenhuma música encontrada — verifique DATASET_DIR"

print("Carregando + extraindo mel-espectrograma...")
{songs_block}
assert len(songs) > 0, "Nenhuma música útil — verifique o dataset"
""")

    # ── Célula 5: Split + Loaders ─────────────────────────────────────────────
    md(nb, "## 5. Split + Dataset + Loaders")
    code(nb, """\
random.shuffle(songs)
n_val = max(1, int(len(songs) * VAL_FRAC))
val_songs   = songs[:n_val]
train_songs = songs[n_val:]
print(f"Train: {len(train_songs)}   Val: {len(val_songs)}")

mel_mean, mel_std = compute_mel_stats(train_songs)
print(f"Mel stats: mean={mel_mean:.3f}  std={mel_std:.3f}")

train_ds = DrumChartDataset(train_songs, mel_mean, mel_std, augment=True)
val_ds   = DrumChartDataset(val_songs,   mel_mean, mel_std, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f"Batches — train: {len(train_loader)}  val: {len(val_loader)}")
""")

    # ── Célula 6: Modelo + pos_weight ─────────────────────────────────────────
    md(nb, "## 6. Modelo + pos_weight")
    pw_init = cfg["pos_weight_init"]
    code(nb, f"""\
def compute_pos_weight(songs):
    pos = {pw_init}
    neg = {pw_init}
    for s in songs:
        pos += s.target.sum(dim=0)
        neg += (1 - s.target).sum(dim=0)
    pw = (neg / pos.clamp(min=1)).clamp(max=POS_WEIGHT_CAP)
    print({{n: round(v, 2) for n, v in zip({zn}, pw.tolist())}})
    return pw

model = {cfg['model_class']}(n_mels=N_MELS).to(DEVICE)
print(f"Parâmetros: {{count_params(model):,}}")

pos_weight = compute_pos_weight(train_songs)
""")

    # ── Célula 7: Treino ──────────────────────────────────────────────────────
    md(nb, "## 7. Treino com early stopping\n\nLoss: BCE + pos_weight. Mixed precision. Checkpoint pelo melhor F1 macro.")
    epoch_log = cfg["epoch_log"]
    code(nb, f"""\
import collections

@torch.no_grad()
def compute_metrics(probs, targets, threshold=0.5):
    preds = (probs >= threshold).float()
    out = {{}}
    f1s = []
    for {lv} in range({lr}):
        p = preds[..., {lv}].flatten()
        t = targets[..., {lv}].flatten()
        tp = ((p == 1) & (t == 1)).sum().item()
        fp = ((p == 1) & (t == 0)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[{fk}] = f1
        f1s.append(f1)
    out["f1_macro"] = float(np.mean(f1s))

    pred_cnt = preds.sum(dim=-1).flatten()
    true_cnt = targets.sum(dim=-1).flatten()
    pred_has = (pred_cnt >= 1).float()
    true_has = (true_cnt >= 1).float()
    tp = ((pred_has == 1) & (true_has == 1)).sum().item()
    fp = ((pred_has == 1) & (true_has == 0)).sum().item()
    fn = ((pred_has == 0) & (true_has == 1)).sum().item()
    po = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    out["onset_f1"]   = 2 * po * ro / (po + ro) if (po + ro) > 0 else 0.0
    out["onset_prec"] = po
    out["onset_rec"]  = ro
    out["count_acc"]  = (pred_cnt == true_cnt).float().mean().item()
    out["count_mae"]  = (pred_cnt - true_cnt).abs().float().mean().item()
    return out


def train_one_epoch(model, loader, optim, criterion, scaler):
    model.train()
    losses = []
    for mel, target in loader:
        mel    = mel.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        optim.zero_grad()
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            logits = model(mel)
            loss   = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, P, T = [], [], []
    for mel, target in loader:
        mel    = mel.to(DEVICE)
        target = target.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            logits = model(mel)
            loss   = criterion(logits, target)
        losses.append(loss.item())
        P.append(torch.sigmoid(logits).cpu())
        T.append(target.cpu())
    probs = torch.cat(P, dim=0)
    tgts  = torch.cat(T, dim=0)
    out   = compute_metrics(probs, tgts)
    out["loss"] = float(np.mean(losses))
    return out


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optim_    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=EPOCHS)
scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

best_f1, no_improve, history = -1.0, 0, []
for epoch in range(1, EPOCHS + 1):
    tl = train_one_epoch(model, train_loader, optim_, criterion, scaler)
    v  = evaluate(model, val_loader, criterion)
    sched.step()
    history.append({{"epoch": epoch, "train_loss": tl, **v}})

    print(f"[ep {{epoch:3d}}] tr={{tl:.4f}} vl={{v['loss']:.4f}} | "
          f"F1m={{v['f1_macro']:.3f}} ons={{v['onset_f1']:.3f}} cnt={{v['count_acc']:.3f}} | "
          {epoch_log})

    if v["f1_macro"] > best_f1:
        best_f1, no_improve = v["f1_macro"], 0
        torch.save({{"model": model.state_dict(),
                    "epoch": epoch,
                    "val_f1_macro": best_f1}}, CKPT_PATH)
        print(f"   → checkpoint salvo em Drive (F1m={{best_f1:.4f}})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping (sem melhora por {{PATIENCE}} epochs)")
            break

torch.save({{"mel_mean": mel_mean, "mel_std": mel_std,
            "history": history, "n_mels": N_MELS}}, META_PATH)
print(f"Treino concluído.  Best F1_macro = {{best_f1:.4f}}")
print(f"Checkpoint: {{CKPT_PATH}}")
print(f"Meta      : {{META_PATH}}")
""")

    # ── Célula 8: Threshold tuning ────────────────────────────────────────────
    md(nb, "## 8. Threshold tuning por lane\n\nEncontra o melhor threshold no val set e salva no meta.")
    ln_var = cfg["lane_names_var"]
    code(nb, f"""\
@torch.no_grad()
def collect_val_predictions(model, val_songs):
    model.eval()
    P, T = [], []
    for s in val_songs:
        mel   = normalize_mel(s.mel, mel_mean, mel_std).unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()
        P.append(probs)
        T.append(s.target.numpy())
    return np.concatenate(P, 0), np.concatenate(T, 0)


def f1_at(probs, targets, lane, t):
    p  = (probs[:, lane] >= t).astype(np.int32)
    g  = targets[:, lane].astype(np.int32)
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
print(f"Carregado: epoch {{ckpt['epoch']}}  F1_macro={{ckpt['val_f1_macro']:.4f}}")

probs, targets = collect_val_predictions(model, val_songs)
thresholds = np.arange(0.05, 0.95, 0.025)

optimal_thresholds = {{}}
print(f"\\n{{'Lane':<12s}} {{'thr':>6s}} {{'F1':>6s}} {{'P':>6s}} {{'R':>6s}}")
print("─" * 42)
for lane in range({lr}):
    best = max(((t, *f1_at(probs, targets, lane, t)) for t in thresholds),
               key=lambda x: x[1])
    t, f1, p, r = best
    optimal_thresholds[{ln_var}[lane]] = float(t)
    print(f"{{{ln_var}[lane]:<12s}} {{t:>6.3f}} {{f1:>6.3f}} {{p:>6.3f}} {{r:>6.3f}}")

meta = torch.load(META_PATH, map_location="cpu")
meta["optimal_thresholds"] = optimal_thresholds
torch.save(meta, META_PATH)
print(f"\\nThresholds salvos em {{META_PATH}}")
""")

    # ── Célula 9: Seção 7.5 ───────────────────────────────────────────────────
    sec75_note = cfg["sec75_note"]
    md(nb, f"""## 9. Métricas comparativas: estrito × onset × contagem{sec75_note}

- **estrito**  — F1 macro com thresholds otimizados. Objetivo principal.
- **onset**    — "previu nota onde tem nota?" — F1 binário sobre `sum > 0`.
- **contagem** — % de steps com número exato; MAE da diferença.
""")
    code(nb, f"""\
thr_arr   = np.array([optimal_thresholds[{ln_var}[i]] for i in range({lr})])
preds_opt = (probs >= thr_arr[None, :]).astype(np.float32)
gt        = targets.astype(np.float32)

# Estrito
f1s_strict = []
for lane in range({lr}):
    f1, _, _ = f1_at(probs, gt, lane, optimal_thresholds[{ln_var}[lane]])
    f1s_strict.append(f1)
f1_strict = float(np.mean(f1s_strict))

# Onset
pred_cnt = preds_opt.sum(axis=1)
true_cnt = gt.sum(axis=1)
pred_has = (pred_cnt >= 1).astype(np.float32)
true_has = (true_cnt >= 1).astype(np.float32)
tp = float(((pred_has == 1) & (true_has == 1)).sum())
fp = float(((pred_has == 1) & (true_has == 0)).sum())
fn = float(((pred_has == 0) & (true_has == 1)).sum())
prec_o   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_o    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_onset = 2 * prec_o * rec_o / (prec_o + rec_o) if (prec_o + rec_o) > 0 else 0.0

# Contagem
count_acc = float((pred_cnt == true_cnt).mean())
count_mae = float(np.abs(pred_cnt - true_cnt).mean())

diff = (pred_cnt - true_cnt).astype(int)
dist = collections.Counter(diff.tolist())
n_total = len(diff)

print(f"\\n{{'='*56}}")
print(f"  Métricas finais com thresholds otimizados")
print(f"{{'='*56}}")
print(f"  estrito   F1_macro = {{f1_strict:.4f}}")
print(f"  onset     F1       = {{f1_onset:.4f}}  (P={{prec_o:.3f}}  R={{rec_o:.3f}})")
print(f"  contagem  acurácia = {{count_acc:.4f}}  MAE = {{count_mae:.3f}}")

print(f"\\n  Distribuição (pred_count - true_count):")
for d in sorted(dist.keys()):
    n   = dist[d]
    pct = 100.0 * n / n_total
    bar = "█" * int(50 * n / n_total)
    print(f"    {{d:+3d}}: {{n:7d}} steps ({{pct:5.1f}}%) {{bar}}")

meta = torch.load(META_PATH, map_location="cpu")
meta["final_metrics"] = {{
    "f1_strict": f1_strict,   "f1_onset":   f1_onset,
    "onset_prec": prec_o,     "onset_rec":  rec_o,
    "count_acc":  count_acc,  "count_mae":  count_mae,
    "count_diff_distribution": {{int(k): int(v) for k, v in dist.items()}},
}}
torch.save(meta, META_PATH)
print(f"\\nMétricas salvas em {{META_PATH}}")
""")

    # ── Célula 10: Curvas de treino ───────────────────────────────────────────
    md(nb, "## 10. Curvas de treino + piano roll")
    pr_yticks = cfg["piano_roll_yticks"]
    code(nb, f"""\
def plot_training_curves(history):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    ax[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    ax[0].plot(epochs, [h["loss"]       for h in history], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Loss"); ax[0].grid(alpha=0.3)

    for lane_name in {ln_var}:
        ax[1].plot(epochs, [h[f"f1_{{lane_name}}"] for h in history],
                   label=lane_name, alpha=0.7)
    ax[1].plot(epochs, [h["f1_macro"] for h in history], "k--", lw=2, label="macro")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("F1"); ax[1].set_ylim(0, 1)
    ax[1].set_title("F1 estrito por lane"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    ax[2].plot(epochs, [h["f1_macro"]  for h in history], "k-", lw=2, label="estrito (macro)")
    ax[2].plot(epochs, [h["onset_f1"]  for h in history], "g-", lw=2, label="onset")
    ax[2].plot(epochs, [h["count_acc"] for h in history], "b-", lw=2, label="count_acc")
    ax[2].set_xlabel("epoch"); ax[2].set_ylim(0, 1)
    ax[2].set_title("Estrito vs onset vs contagem"); ax[2].legend(); ax[2].grid(alpha=0.3)

    plt.tight_layout(); plt.show()


@torch.no_grad()
def piano_roll(song: SongData, max_steps: int = 400):
    mel   = normalize_mel(song.mel, mel_mean, mel_std).unsqueeze(0).to(DEVICE)
    probs = torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()
    thr   = np.array([optimal_thresholds[{pr_yticks}[i]] for i in range({lr})])
    preds = (probs >= thr[None, :]).astype(np.float32)
    target = song.target.numpy()
    n = min(target.shape[0], probs.shape[0], max_steps)
    fig, ax = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    ax[0].imshow(target[:n].T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax[0].set_title(f"Alvo — {{song.song_id}}")
    ax[0].set_yticks(range({lr})); ax[0].set_yticklabels({pr_yticks})
    ax[1].imshow(probs[:n].T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[1].set_title("Probabilidades preditas")
    ax[1].set_yticks(range({lr})); ax[1].set_yticklabels({pr_yticks})
    ax[2].imshow(preds[:n].T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax[2].set_title("Predições (thresholds otimizados)")
    ax[2].set_yticks(range({lr})); ax[2].set_yticklabels({pr_yticks})
    ax[2].set_xlabel("step (semicolcheia)")
    plt.tight_layout(); plt.show()


plot_training_curves(history)
piano_roll(val_songs[0])
""")

    # ── Célula 11: Download modelo ────────────────────────────────────────────
    md(nb, f"""## 11. Download do modelo treinado

O modelo já foi salvo no Drive em `CKPT_DIR`. As células abaixo também
iniciam o download direto para o seu computador — útil se quiser copiar
os arquivos direto para `treinamento/checkpoint/{ins}/` sem abrir o Drive.
""")
    code(nb, """\
from google.colab import files

print(f"Baixando  {CKPT_PATH.name} ...")
files.download(str(CKPT_PATH))

print(f"Baixando  {META_PATH.name} ...")
files.download(str(META_PATH))

print()
print("Feito! Coloque os arquivos em:")
print(f"  treinamento/checkpoint/{CKPT_DIR.name}/")
print("e rode  python main.py --audio <musica.mp3>  para testar o pipeline.")
""")

    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Gera os 4 notebooks
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    for ins, cfg in INSTRUMENTS.items():
        nb   = generate_notebook(cfg)
        path = out_dir / f"colab_treino_{ins}.ipynb"
        nb["metadata"]["colab"]["name"] = path.name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        n_cells = len(nb["cells"])
        print(f"Gerado: {path.name}  ({n_cells} células)")
