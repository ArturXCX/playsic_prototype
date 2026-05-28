"""
Gera treinamento_modelo_vocals.ipynb a partir de uma lista de células.

v1: single-lane onset detector — modelo prediz "tem vocal ativo aqui?" por step,
e o chart escreve MIDI 60 (C4) em cada step previsto. Não é karaoke afinado;
serve como prova de conceito + base pra v2 com pitch tracking.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS = []


def md(text: str):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text: str):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# ─────────────────────────────────────────────────────────────────────────────
md("""# Treinamento e validação — modelo de vocals (v1: onset detector)

Notebook **enxuto**: treina o `VocalsCRNN`, avalia em validação, e roda um
teste end-to-end que gera o chart + web preview de uma música.

**Sobre a v1**: o modelo prevê uma única lane binária por step — "tem
vocal ativo aqui?". Para o chart, cada onset detectado vira uma nota
MIDI 60 (C4) na track PART VOCALS, gerando um chart de karaokê monotone
mas jogável. Pitch tracking real fica pra v2.

Toda a lógica reutilizável vive em arquivos `.py` irmãos:

| Arquivo | O que tem |
|---|---|
| `vocals_crnn.py`    | Classe `VocalsCRNN` + constantes (LANES=1) |
| `audio_features.py` | mel-espectrograma alinhado ao grid (BPM-dependente) |
| `notes_xlsx.py`     | `parse_vocals_events` (filtra pitches 36-84) |
| `training_utils.py` | `SongData`, `DrumChartDataset`, `preprocess_song`, etc. |
| `modelo_gera_excel.py` | API de inferência (áudio + BPM → xlsx parcial) |

Pipeline do teste final:

```
vocals.ogg + BPM
      │
      ▼  modelo_gera_excel.infer (instrument="vocals")
vocals_partial.xlsx
      │
      ▼  excel_to_midi.convert  (sem expansão de dificuldade)
notes.mid + .ogg copiados + song.ini gerado
      │
      ▼  onyx_web_preview.build_preview
resultados/dataset_validacao/previews/<nome>/index.html
```

⚠ A diferença para os outros notebooks:
- `N_LANES = 1` (única lane "VocalActive")
- A seção 7.5 (estrito vs onset vs contagem) fica degenerada — com 1 lane,
  `estrito ≡ onset`. A contagem ainda vira informação útil (densidade).
""")

md("## 1. Imports e paths")

code("""\
import sys, random, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ── REPO_ROOT (a partir do cwd ou do diretório atual do notebook)
THIS_DIR = Path.cwd()
if (THIS_DIR / "dados").is_dir():
    REPO_ROOT = THIS_DIR
elif THIS_DIR.name == "treinamento" and (THIS_DIR.parent / "dados").is_dir():
    REPO_ROOT = THIS_DIR.parent
else:
    REPO_ROOT = None
    for p in THIS_DIR.parents:
        if (p / "dados").is_dir():
            REPO_ROOT = p
            break
    assert REPO_ROOT is not None, f"raiz do repo não encontrada a partir de {THIS_DIR}"

# adiciona pastas relevantes ao sys.path para imports diretos
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "treinamento"))
sys.path.insert(0, str(REPO_ROOT / "processamento" / "midi_excel"))
sys.path.insert(0, str(REPO_ROOT / "onyx"))

# módulos do projeto
from vocals_crnn import (
    VocalsCRNN, LANES, LANE_NAMES, N_LANES, AUDIO_STEM,
    VOCAL_MIDI_MIN, VOCAL_MIDI_MAX, count_params,
)
from audio_features import (
    SAMPLE_RATE, SUBDIV_PER_BEAT, TICKS_PER_BEAT,
    audio_to_grid_mel, normalize_mel, step_duration_seconds,
)
from notes_xlsx import parse_vocals_events, predictions_to_xlsx
from training_utils import (
    SongData, list_song_dirs, preprocess_song,
    compute_mel_stats, DrumChartDataset, CHUNK_STEPS,
)
import modelo_gera_excel
import excel_to_midi
import onyx_web_preview

import song_ini  # raiz do repo

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"REPO_ROOT = {REPO_ROOT}")
print(f"Device    = {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU       = {torch.cuda.get_device_name(0)}")
""")

md("## 2. Configurações de treino")

code("""\
# ── pastas ──────────────────────────────────────────────────
DATASET_ROOT   = REPO_ROOT / "dados" / "dataset"
CHECKPOINT_DIR = REPO_ROOT / "treinamento" / "checkpoint" / "vocals"
VALIDATION_DIR = REPO_ROOT / "treinamento" / "validação" / "vocals"
CHARTS_DIR     = REPO_ROOT / "resultados" / "dataset_validacao" / "charts"
PREVIEWS_DIR   = REPO_ROOT / "resultados" / "dataset_validacao" / "previews"
for d in (CHECKPOINT_DIR, VALIDATION_DIR, CHARTS_DIR, PREVIEWS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CKPT_PATH = CHECKPOINT_DIR / "vocals_crnn_best.pt"
META_PATH = CHECKPOINT_DIR / "vocals_crnn_meta.pt"

# ── hiperparâmetros ─────────────────────────────────────────
N_MELS         = 128
BATCH_SIZE     = 8
EPOCHS         = 80
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
PATIENCE       = 15
POS_WEIGHT_CAP = 50.0
VAL_FRAC       = 0.15
MIN_NOTE_COUNT = 10   # descarta músicas sem vocals (e.g., instrumentais)

print(f"Vocal pitch range filtrado: MIDI {VOCAL_MIDI_MIN} – {VOCAL_MIDI_MAX}")
print(f"N_LANES = {N_LANES}  ({LANE_NAMES})")

# ── teste final ─────────────────────────────────────────────
N_DEMO_SONGS = 1
""")

md("## 3. Pré-processamento do dataset")

code("""\
print(f"Procurando músicas em: {DATASET_ROOT}")
song_dirs = list_song_dirs(DATASET_ROOT, audio_stem=AUDIO_STEM)
print(f"Encontradas {len(song_dirs)} músicas com {AUDIO_STEM} + notes.xlsx")
assert len(song_dirs) > 0, (
    "nenhuma música encontrada — rode organizar_dataset.py antes "
    "e certifique-se que as músicas têm PART VOCALS chart + vocals.ogg"
)

print("Carregando + extraindo mel...")
songs, skipped_none, skipped_empty = [], 0, 0
for d in tqdm(song_dirs):
    s = preprocess_song(d, lanes_map=LANES, n_lanes=N_LANES,
                        n_mels=N_MELS, audio_stem=AUDIO_STEM,
                        parse_events_fn=parse_vocals_events)
    if s is None:
        skipped_none += 1
        continue
    if s.target.sum() < MIN_NOTE_COUNT:
        skipped_empty += 1   # música instrumental ou vocais fora do range
        continue
    songs.append(s)

print(f"OK: {len(songs)} músicas pré-processadas")
print(f"   Puladas por erro:         {skipped_none}")
print(f"   Puladas (sem vocal ativo): {skipped_empty}")
assert len(songs) > 0, "nenhuma música com vocals — verifique o dataset"
""")

md("## 4. Split + Dataset + Loaders")

code("""\
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
""")

md("## 5. Modelo + pos_weight")

code("""\
def compute_pos_weight(songs):
    pos = torch.zeros(N_LANES)
    neg = torch.zeros(N_LANES)
    for s in songs:
        pos += s.target.sum(dim=0)
        neg += (1 - s.target).sum(dim=0)
    pw = (neg / pos.clamp(min=1)).clamp(max=POS_WEIGHT_CAP)
    print({n: round(v, 2) for n, v in zip(LANE_NAMES, pw.tolist())})
    return pw

model = VocalsCRNN(n_mels=N_MELS).to(DEVICE)
print(f"Parâmetros: {count_params(model):,}")

pos_weight = compute_pos_weight(train_songs)
""")

md("""## 6. Treino com early stopping

Loss: BCE + pos_weight. Mixed precision na GPU. Checkpoint pelo melhor F1.
""")

code("""\
@torch.no_grad()
def compute_metrics(probs, targets, threshold=0.5):
    preds = (probs >= threshold).float()
    out = {}
    f1s = []
    for lane in range(N_LANES):
        p = preds[..., lane].flatten()
        t = targets[..., lane].flatten()
        tp = ((p == 1) & (t == 1)).sum().item()
        fp = ((p == 1) & (t == 0)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[f"f1_{LANE_NAMES[lane]}"] = f1
        f1s.append(f1)
    out["f1_macro"] = float(np.mean(f1s))

    # Onset/contagem viram quase iguais ao estrito (N_LANES=1) mas mantemos
    # pra coerência de schema com os outros modelos.
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

    out["count_acc"] = (pred_cnt == true_cnt).float().mean().item()
    out["count_mae"] = (pred_cnt - true_cnt).abs().float().mean().item()
    return out


def train_one_epoch(model, loader, optim, criterion, scaler):
    model.train()
    losses = []
    for mel, target in loader:
        mel    = mel.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        optim.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
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
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(mel)
            loss   = criterion(logits, target)
        losses.append(loss.item())
        P.append(torch.sigmoid(logits).cpu())
        T.append(target.cpu())
    probs = torch.cat(P, dim=0)
    tgts  = torch.cat(T, dim=0)
    out = compute_metrics(probs, tgts)
    out["loss"] = float(np.mean(losses))
    return out


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optim_    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=EPOCHS)
scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_f1, no_improve, history = -1.0, 0, []
for epoch in range(1, EPOCHS + 1):
    tl = train_one_epoch(model, train_loader, optim_, criterion, scaler)
    v  = evaluate(model, val_loader, criterion)
    sched.step()
    history.append({"epoch": epoch, "train_loss": tl, **v})

    print(f"[ep {epoch:3d}] tr={tl:.4f} vl={v['loss']:.4f} | "
          f"F1m={v['f1_macro']:.3f} ons={v['onset_f1']:.3f} cnt={v['count_acc']:.3f} | "
          f"V={v['f1_VocalActive']:.3f}")

    if v["f1_macro"] > best_f1:
        best_f1, no_improve = v["f1_macro"], 0
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "val_f1_macro": best_f1}, CKPT_PATH)
        print(f"   → checkpoint salvo (F1m={best_f1:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping (sem melhora por {PATIENCE} epochs)")
            break

torch.save({"mel_mean": mel_mean, "mel_std": mel_std,
            "history": history, "n_mels": N_MELS}, META_PATH)
print(f"Checkpoint: {CKPT_PATH}")
print(f"Meta:       {META_PATH}")
""")

md("""## 7. Threshold tuning (lane única)

Pega o melhor threshold da lane no val set e salva no meta.
""")

code("""\
@torch.no_grad()
def collect_val_predictions(model, val_songs):
    model.eval()
    P, T = [], []
    for s in val_songs:
        mel = normalize_mel(s.mel, mel_mean, mel_std).unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()
        P.append(probs)
        T.append(s.target.numpy())
    return np.concatenate(P, 0), np.concatenate(T, 0)


def f1_at(probs, targets, lane, t):
    p = (probs[:, lane] >= t).astype(np.int32)
    g = targets[:, lane].astype(np.int32)
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
print(f"Carregado: epoch {ckpt['epoch']}  F1_macro={ckpt['val_f1_macro']:.4f}")

probs, targets = collect_val_predictions(model, val_songs)
thresholds = np.arange(0.05, 0.95, 0.025)

optimal_thresholds = {}
print(f"\\n{'Lane':<14s} {'thr':>6s} {'F1':>6s} {'P':>6s} {'R':>6s}")
print("─" * 44)
for lane in range(N_LANES):
    best = max(((t, *f1_at(probs, targets, lane, t)) for t in thresholds),
               key=lambda x: x[1])
    t, f1, p, r = best
    optimal_thresholds[LANE_NAMES[lane]] = float(t)
    print(f"{LANE_NAMES[lane]:<14s} {t:>6.3f} {f1:>6.3f} {p:>6.3f} {r:>6.3f}")

meta = torch.load(META_PATH, map_location="cpu")
meta["optimal_thresholds"] = optimal_thresholds
torch.save(meta, META_PATH)
print(f"\\nThresholds salvos em {META_PATH}")
""")

md("""## 7.5 Métricas comparativas: estrito × onset × contagem

⚠ Com N_LANES=1, `estrito ≡ onset` (trivialmente). A `contagem` ainda dá
informação útil — só quantifica em quantos steps o modelo erra a presença.
""")

code("""\
import collections

thr_arr   = np.array([optimal_thresholds[LANE_NAMES[i]] for i in range(N_LANES)])
preds_opt = (probs >= thr_arr[None, :]).astype(np.float32)
gt        = targets.astype(np.float32)

f1s = []
for lane in range(N_LANES):
    f1, _, _ = f1_at(probs, gt, lane, optimal_thresholds[LANE_NAMES[lane]])
    f1s.append(f1)
f1_strict = float(np.mean(f1s))

pred_cnt = preds_opt.sum(axis=1)
true_cnt = gt.sum(axis=1)
pred_has = (pred_cnt >= 1).astype(np.float32)
true_has = (true_cnt >= 1).astype(np.float32)
tp = float(((pred_has == 1) & (true_has == 1)).sum())
fp = float(((pred_has == 1) & (true_has == 0)).sum())
fn = float(((pred_has == 0) & (true_has == 1)).sum())
prec_o = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_o  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_onset = 2 * prec_o * rec_o / (prec_o + rec_o) if (prec_o + rec_o) > 0 else 0.0

count_acc = float((pred_cnt == true_cnt).mean())
count_mae = float(np.abs(pred_cnt - true_cnt).mean())

diff = (pred_cnt - true_cnt).astype(int)
dist = collections.Counter(diff.tolist())
n_total = len(diff)
sorted_keys = sorted(dist.keys())

print(f"\\n{'='*56}")
print(f"  Métricas finais com threshold otimizado")
print(f"{'='*56}")
print(f"  estrito   F1_macro = {f1_strict:.4f}")
print(f"  onset     F1       = {f1_onset:.4f}  (P={prec_o:.3f}  R={rec_o:.3f})")
print(f"  contagem  acurácia = {count_acc:.4f}  MAE = {count_mae:.3f}")
print(f"  (esperado: estrito == onset pois N_LANES=1)")

print(f"\\n  Distribuição (pred_count - true_count):")
for d in sorted_keys:
    n = dist[d]
    pct = 100.0 * n / n_total
    bar = "█" * int(50 * n / n_total)
    print(f"    {d:+3d}: {n:7d} steps ({pct:5.1f}%) {bar}")

meta = torch.load(META_PATH, map_location="cpu")
meta["final_metrics"] = {
    "f1_strict":  f1_strict,
    "f1_onset":   f1_onset,
    "onset_prec": prec_o,
    "onset_rec":  rec_o,
    "count_acc":  count_acc,
    "count_mae":  count_mae,
    "count_diff_distribution": {int(k): int(v) for k, v in dist.items()},
}
torch.save(meta, META_PATH)
print(f"\\nMétricas salvas em {META_PATH}")
""")

md("## 8. Curvas de treino + visualização de uma música do val")

code("""\
def plot_training_curves(history):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    ax[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    ax[0].plot(epochs, [h["loss"]       for h in history], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Loss"); ax[0].grid(alpha=0.3)

    for lane in LANE_NAMES:
        ax[1].plot(epochs, [h[f"f1_{lane}"] for h in history], label=lane, alpha=0.7)
    ax[1].plot(epochs, [h["f1_macro"] for h in history], "k--", lw=2, label="macro")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("F1"); ax[1].set_ylim(0, 1)
    ax[1].set_title("F1 por lane"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    ax[2].plot(epochs, [h["f1_macro"]  for h in history], "k-", lw=2, label="estrito (macro)")
    ax[2].plot(epochs, [h["onset_f1"]  for h in history], "g-", lw=2, label="onset")
    ax[2].plot(epochs, [h["count_acc"] for h in history], "b-", lw=2, label="count_acc")
    ax[2].set_xlabel("epoch"); ax[2].set_ylim(0, 1)
    ax[2].set_title("Estrito vs onset vs contagem"); ax[2].legend(); ax[2].grid(alpha=0.3)

    plt.tight_layout(); plt.show()


@torch.no_grad()
def piano_roll(song: SongData, max_steps: int = 400):
    mel = normalize_mel(song.mel, mel_mean, mel_std).unsqueeze(0).to(DEVICE)
    probs = torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()
    thr = np.array([optimal_thresholds[LANE_NAMES[i]] for i in range(N_LANES)])
    preds = (probs >= thr[None, :]).astype(np.float32)

    target = song.target.numpy()
    n = min(target.shape[0], probs.shape[0], max_steps)
    fig, ax = plt.subplots(3, 1, figsize=(14, 4), sharex=True)
    ax[0].imshow(target[:n].T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax[0].set_title(f"Alvo (VocalActive) — {song.song_id}")
    ax[0].set_yticks([]); ax[0].set_yticklabels([])
    ax[1].imshow(probs[:n].T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[1].set_title("Probabilidade predita")
    ax[1].set_yticks([])
    ax[2].imshow(preds[:n].T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax[2].set_title("Predição (threshold otimizado)")
    ax[2].set_yticks([]); ax[2].set_xlabel("step (semicolcheia)")
    plt.tight_layout(); plt.show()


plot_training_curves(history)
piano_roll(val_songs[0])
""")

md("""## 9. Teste end-to-end: chart + web preview

⚠ O chart de vocals na v1 é **monotone** (todas as notas em MIDI 60). O
preview vai mostrar uma linha contínua de notas em C4 — não é a música
afinada, é só a validação de que o pipeline áudio→onset→chart fecha.
""")

code("""\
import shutil

def build_chart_from_song(song: SongData,
                          vocals_xlsx: Path,
                          chart_dir: Path) -> Path:
    \"\"\"Monta a pasta Clone Hero com a track de vocals predita (v1: monotone).\"\"\"
    chart_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("drums.ogg", "song.ogg", "guitar.ogg", "rhythm.ogg", "vocals.ogg"):
        src = song.folder / fname
        if src.exists():
            shutil.copy2(src, chart_dir / fname)

    if not (chart_dir / "song.ogg").exists():
        for fallback in ("vocals.ogg", "drums.ogg", "guitar.ogg", "rhythm.ogg"):
            if (chart_dir / fallback).exists():
                shutil.copy2(chart_dir / fallback, chart_dir / "song.ogg")
                print(f"  [WARN] song.ogg ausente; usei {fallback} como fallback")
                break

    if (song.folder / "album.png").exists():
        shutil.copy2(song.folder / "album.png", chart_dir / "album.png")

    excel_to_midi.convert(vocals_xlsx, chart_dir / "notes.mid")

    try:
        song_ini.generate_validation_song_ini(song.folder, chart_dir / "song.ini")
    except Exception as e:
        print(f"  [WARN] generate_validation_song_ini falhou ({e}); escrevendo mínimo")
        meta = {"name": song.song_id, "artist": "Unknown", "album": "Unknown"}
        (chart_dir / "song.ini").write_text(
            song_ini.build_song_ini_text(meta), encoding="utf-8")

    return chart_dir


demo_songs = val_songs[:N_DEMO_SONGS]
print(f"Gerando chart + preview para {len(demo_songs)} música(s)\\n")

for s in demo_songs:
    print(f"━━━ {s.song_id} (BPM={s.bpm:.2f}) ━━━")

    xlsx_path = VALIDATION_DIR / f"{s.song_id}__vocals.xlsx"
    modelo_gera_excel.infer(
        audio_path=s.folder / AUDIO_STEM,
        bpm=s.bpm,
        instrument="vocals",
        model_path=CKPT_PATH,
        meta_path=META_PATH,
        out_xlsx=xlsx_path,
    )
    print(f"  xlsx parcial: {xlsx_path}")

    chart_dir = CHARTS_DIR / s.song_id
    build_chart_from_song(s, xlsx_path, chart_dir)
    print(f"  chart:    {chart_dir}")
    print(f"  arquivos: {sorted(p.name for p in chart_dir.iterdir())}")

    preview_dir = PREVIEWS_DIR / s.song_id
    try:
        html = onyx_web_preview.build_preview(chart_dir, preview_dir)
        print(f"  preview:  {html}")
    except Exception as e:
        print(f"  ✘ preview não gerado: {e}")
        print(f"     (o chart está em {chart_dir} — gere o preview depois)")
    print()
""")

md("""## 10. Próximos passos (v2)

1. **Pitch quantization**: trocar `N_LANES=1` por 24-40 lanes de pitch
   (MIDI 48-71 ou 40-79). Mudar `parse_vocals_events` pra mapear pitch real
   ao lane index. Loss continua BCE multi-label.
2. **Vocal silêncio explícito**: adicionar uma 25ª lane "silence" pra
   regularizar — o modelo aprende quando NÃO cantar.
3. **Sustain length**: hoje todo step ativo vira nota curta. Pra karaoke
   real, o `predictions_to_xlsx` deveria juntar steps consecutivos numa
   nota sustentada.
4. **Lyric alignment**: tópico totalmente separado — exige modelo de fala
   ou ASR alinhado.
""")


# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "treinamento_modelo_vocals.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Gerado: {out}  ({len(CELLS)} células)")
