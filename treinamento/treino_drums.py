"""
treino_drums — re-treino do DrumCRNN no dataset completo, com o mel corrigido
(hop fino + max-pool em audio_features) e crops por música. Reusa os módulos
compartilhados (mesma lógica do notebook), rodável de forma autônoma.

Salva em drums_crnn_best.new.pt / _meta.new.pt durante o treino (não mexe no
checkpoint em uso); troca pelos nomes canônicos só no final.

Uso:  python treinamento/treino_drums.py
"""
from __future__ import annotations
import sys, random, time, shutil, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "treinamento"))
from drum_crnn import DrumCRNN, LANES, LANE_NAMES, N_LANES, GAMEPLAY_MAX_MIDI, count_params
from audio_features import normalize_mel
from training_utils import list_song_dirs, preprocess_song, compute_mel_stats, DrumChartDataset

DATASET = REPO / "dados" / "dataset"
CKPT = REPO / "treinamento" / "checkpoint" / "drums"; CKPT.mkdir(parents=True, exist_ok=True)
BEST_NEW = CKPT / "drums_crnn_best.new.pt"; META_NEW = CKPT / "drums_crnn_meta.new.pt"
BEST = CKPT / "drums_crnn_best.pt"; META = CKPT / "drums_crnn_meta.pt"

N_MELS, BATCH, EPOCHS, LR, WD, CLIP, PATIENCE, POS_CAP, VAL_FRAC = 128, 16, 40, 1e-3, 1e-4, 1.0, 8, 50.0, 0.15
_ap = argparse.ArgumentParser()
_ap.add_argument("--max", type=int, default=None, help="limita nº de músicas (smoke test)")
_ap.add_argument("--epochs", type=int, default=EPOCHS)
_args = _ap.parse_args()
EPOCHS = _args.epochs
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device={DEVICE}", flush=True)
song_dirs = list_song_dirs(DATASET, audio_stem="drums.ogg")
if _args.max:
    song_dirs = song_dirs[:_args.max]
print(f"{len(song_dirs)} músicas com drums.ogg+notes.xlsx. Pré-processando...", flush=True)
songs = []; t0 = time.time()
for i, d in enumerate(song_dirs):
    s = preprocess_song(d, lanes_map=LANES, n_lanes=N_LANES, n_mels=N_MELS,
                        audio_stem="drums.ogg", gameplay_max_midi=GAMEPLAY_MAX_MIDI)
    if s is not None: songs.append(s)
    if (i + 1) % 100 == 0:
        print(f"  pré {i+1}/{len(song_dirs)}  ({time.time()-t0:.0f}s, {len(songs)} ok)", flush=True)
print(f"{len(songs)} músicas pré-processadas em {time.time()-t0:.0f}s", flush=True)

random.shuffle(songs)
nval = max(1, int(len(songs) * VAL_FRAC))
val_songs, train_songs = songs[:nval], songs[nval:]
mel_mean, mel_std = compute_mel_stats(train_songs)
print(f"Train={len(train_songs)} Val={len(val_songs)} mel_mean={mel_mean:.2f} mel_std={mel_std:.2f}", flush=True)

train_loader = DataLoader(DrumChartDataset(train_songs, mel_mean, mel_std, augment=True,
                                           crops_per_song=4),
                          batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(DrumChartDataset(val_songs, mel_mean, mel_std, augment=False), batch_size=BATCH)

pos = torch.zeros(N_LANES); neg = torch.zeros(N_LANES)
for s in train_songs: pos += s.target.sum(0); neg += (1 - s.target).sum(0)
pw = (neg / pos.clamp(min=1)).clamp(max=POS_CAP).to(DEVICE)
model = DrumCRNN(n_mels=N_MELS).to(DEVICE)
print(f"params={count_params(model):,} pos_weight={[round(x,1) for x in pw.tolist()]}", flush=True)

crit = nn.BCEWithLogitsLoss(pos_weight=pw)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

def f1_macro(probs, tgts, thr=0.5):
    preds = (probs >= thr).float(); f1s = []
    for l in range(N_LANES):
        p = preds[..., l].flatten(); t = tgts[..., l].flatten()
        tp = ((p == 1) & (t == 1)).sum().item(); fp = ((p == 1) & (t == 0)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)), f1s

@torch.no_grad()
def evaluate():
    model.eval(); ls = []; P = []; T = []
    for mel, t in val_loader:
        mel = mel.to(DEVICE); t = t.to(DEVICE); logit = model(mel)
        ls.append(crit(logit, t).item()); P.append(torch.sigmoid(logit).cpu()); T.append(t.cpu())
    fm, fs = f1_macro(torch.cat(P), torch.cat(T))
    return float(np.mean(ls)), fm, fs

best, noimp, hist = -1.0, 0, []
for ep in range(1, EPOCHS + 1):
    model.train(); ls = []
    for mel, t in train_loader:
        mel = mel.to(DEVICE); t = t.to(DEVICE); opt.zero_grad()
        loss = crit(model(mel), t); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); ls.append(loss.item())
    tl = float(np.mean(ls)); vl, fm, fs = evaluate(); sched.step()
    hist.append(dict(epoch=ep, train_loss=tl, loss=vl, f1_macro=fm))
    print(f"[ep {ep:2d}] tr={tl:.4f} vl={vl:.4f} F1m={fm:.3f} | " +
          " ".join(f"{n[:1]}={v:.2f}" for n, v in zip(LANE_NAMES, fs)), flush=True)
    if fm > best:
        best, noimp = fm, 0
        torch.save({"model": model.state_dict(), "epoch": ep, "val_f1_macro": best}, BEST_NEW)
        torch.save({"mel_mean": mel_mean, "mel_std": mel_std, "n_mels": N_MELS, "history": hist}, META_NEW)
    else:
        noimp += 1
        if noimp >= PATIENCE:
            print(f"Early stopping (sem melhora por {PATIENCE} epochs)", flush=True); break

# threshold tuning por lane (full-song no val)
ck = torch.load(BEST_NEW, map_location=DEVICE); model.load_state_dict(ck["model"]); model.eval()
@torch.no_grad()
def collect():
    P = []; T = []
    for s in val_songs:
        mel = normalize_mel(s.mel, mel_mean, mel_std).unsqueeze(0).to(DEVICE)
        P.append(torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()); T.append(s.target.numpy())
    return np.concatenate(P), np.concatenate(T)
probs, tgts = collect()
def f1_at(p, g, l, t):
    pp = (p[:, l] >= t).astype(int); gg = g[:, l].astype(int)
    tp = ((pp == 1) & (gg == 1)).sum(); fp = ((pp == 1) & (gg == 0)).sum(); fn = ((pp == 0) & (gg == 1)).sum()
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0
opt_thr = {}
for l in range(N_LANES):
    bt = max(np.arange(0.05, 0.95, 0.025), key=lambda t: f1_at(probs, tgts, l, t))
    opt_thr[LANE_NAMES[l]] = float(bt)
meta = torch.load(META_NEW, map_location="cpu"); meta["optimal_thresholds"] = opt_thr
torch.save(meta, META_NEW)
print(f"\nthresholds: {[(k, round(v,3)) for k,v in opt_thr.items()]}", flush=True)
print(f"F1 macro (best) = {best:.4f}", flush=True)

# troca pelos nomes canônicos (o pipeline passa a usar o modelo novo)
if BEST.exists(): shutil.copy2(BEST, CKPT / "drums_crnn_best.old.pt")
if META.exists(): shutil.copy2(META, CKPT / "drums_crnn_meta.old.pt")
shutil.copy2(BEST_NEW, BEST); shutil.copy2(META_NEW, META)
print(f"✔ modelo novo promovido para {BEST.name} (antigo salvo como .old)", flush=True)
print("DONE", flush=True)
