"""
avaliar_chart — métrica de QUALIDADE de um chart gerado, por jogabilidade +
alinhamento musical (NÃO por semelhança com chart humano).

Motivação: "igual ao humano" é mal-posto (o fret/voicing humano é uma escolha
subjetiva, não recuperável do áudio — ver experimentos/). O que importa é se o
chart é (a) musicalmente ALINHADO — as notas caem em eventos reais do áudio — e
(b) JOGÁVEL — densidade, acordes, sustains e saltos de fret em faixas razoáveis.

Mede, sobre uma pasta de chart (notes.mid + stems .ogg):
  • alinhamento: % de onsets do chart a ±TOL de um onset detectado no stem (librosa)
  • densidade (onsets/s), taxa de acordes, taxa de sustain, salto médio de fret
  • dificuldades monotônicas (Easy ≤ Medium ≤ Hard ≤ Expert)

Uso:
    python processamento/avaliar_chart.py --chart resultados/novas_musicas/charts/<nome>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import mido

TRACK_STEM = {
    "PART GUITAR": ("guitar", "guitar.ogg"),
    "PART BASS":   ("rhythm", "rhythm.ogg"),
    "PART DRUMS":  ("drums",  "drums.ogg"),
    "PART VOCALS": ("vocals", "vocals.ogg"),
}
ALIGN_TOL_S = 0.05          # tolerância de alinhamento chart↔áudio
TAP_TICKS   = 60
DIFF_RANGES = {"Easy": (60, 64), "Medium": (72, 76), "Hard": (84, 88), "Expert": (96, 100)}


def _read_midi(path: Path):
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    tempo = 500000
    for m in mid.tracks[0]:
        if m.type == "set_tempo":
            tempo = m.tempo
    def t2s(t): return t / tpb * (tempo / 1e6)
    sixteenth = (tempo / 1e6) / 4.0          # duração de 1 semicolcheia (s)

    tracks = {}
    for tr in mid.tracks[1:]:
        name = tr.name
        if name not in TRACK_STEM:
            continue
        abst = 0; openn: Dict[int, int] = {}; notes = []
        for m in tr:
            abst += m.time
            if m.type == "note_on" and m.velocity > 0:
                openn[m.note] = abst
            elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
                if m.note in openn:
                    st = openn.pop(m.note)
                    notes.append((m.note, st, abst - st))   # note, start_tick, dur_ticks
        tracks[name] = notes
    return tracks, t2s, sixteenth


def _audio_onsets(stem: Path) -> List[float]:
    import librosa
    y, sr = librosa.load(str(stem), sr=22050, mono=True)
    return list(librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True))


def _coverage(chart_onsets: List[float], audio_onsets: List[float], tol: float) -> float:
    if not chart_onsets or not audio_onsets:
        return 0.0
    import bisect
    a = sorted(audio_onsets); hits = 0
    for t in chart_onsets:
        i = bisect.bisect_left(a, t)
        best = min((abs(a[j] - t) for j in (i - 1, i) if 0 <= j < len(a)), default=1e9)
        if best <= tol:
            hits += 1
    return hits / len(chart_onsets)


def evaluate(chart_dir: Path) -> Dict:
    notes_mid = chart_dir / "notes.mid"
    if not notes_mid.exists():
        raise FileNotFoundError(f"notes.mid não encontrado em {chart_dir}")
    tracks, t2s, sixteenth = _read_midi(notes_mid)
    tol = max(ALIGN_TOL_S, 0.6 * sixteenth)     # tolerância ciente da quantização

    report = {}
    for tname, notes in tracks.items():
        inst, stem_name = TRACK_STEM[tname]
        # Expert (96-100 p/ frets; p/ vocals/drums usamos todas como onsets)
        if inst in ("guitar", "rhythm"):
            expert = [(n, st, d) for n, st, d in notes if 96 <= n <= 100]
        else:
            expert = notes
        if not expert:
            continue

        # onsets (tempos únicos), acordes, sustains, salto de fret
        by_tick: Dict[int, List[int]] = {}
        for n, st, d in expert:
            by_tick.setdefault(st, []).append(n)
        onset_ticks = sorted(by_tick)
        onset_secs  = [t2s(t) for t in onset_ticks]
        dur_total   = max(t2s(st + d) for _n, st, d in expert)
        n_onsets    = len(onset_ticks)
        chord_rate  = sum(1 for v in by_tick.values() if len(v) > 1) / n_onsets
        sust_rate   = sum(1 for _n, _st, d in expert if d > TAP_TICKS) / len(expert)
        density     = n_onsets / dur_total if dur_total else 0.0

        # salto de fret (líder = menor nota do acorde)
        fret_jump = None
        if inst in ("guitar", "rhythm"):
            leads = [min(by_tick[t]) - 96 for t in onset_ticks]
            difs = [abs(b - a) for a, b in zip(leads, leads[1:])]
            fret_jump = sum(difs) / len(difs) if difs else 0.0

        # dificuldades (só p/ instrumentos com codificação por dificuldade;
        # vocals usa pitch real, não cabe esta métrica)
        if inst == "vocals":
            diffs = None; monot = None
        else:
            diffs = {lvl: len([1 for n, _st, _d in notes if lo <= n <= hi])
                     for lvl, (lo, hi) in DIFF_RANGES.items()}
            monot = diffs["Easy"] <= diffs["Medium"] <= diffs["Hard"] <= diffs["Expert"]

        # alinhamento musical (se o stem existir)
        align = None
        stem = chart_dir / stem_name
        if stem.exists():
            align = _coverage(onset_secs, _audio_onsets(stem), tol)

        report[inst] = dict(onsets=n_onsets, density=density, chord_rate=chord_rate,
                            sustain_rate=sust_rate, fret_jump=fret_jump,
                            align=align, diffs=diffs, monotonic=monot)
    return report


def _flag(cond_ok: bool) -> str:
    return "ok " if cond_ok else "!! "


def print_report(chart_dir: Path, report: Dict):
    print(f"\n{'='*64}\n  QUALIDADE DO CHART — {chart_dir.name}\n{'='*64}")
    if not report:
        print("  (nenhuma track jogável encontrada)"); return
    for inst, r in report.items():
        print(f"\n  {inst.upper()}")
        if r["align"] is not None:
            print(f"    {_flag(r['align']>0.6)}alinhamento musical (onsets em eventos reais) = {r['align']:.0%}")
        print(f"    {_flag(0.5<=r['density']<=8)}densidade = {r['density']:.2f} onsets/s")
        # Drums: simultaneidade alta é normal e sustains devem ser ~0 (ver doc §16.5);
        # os limiares de acorde/sustain valem só para instrumentos de fret.
        chord_ok = True if inst == "drums" else r['chord_rate'] <= 0.6
        sust_ok  = (r['sustain_rate'] <= 0.10) if inst == "drums" else (0.03 <= r['sustain_rate'] <= 0.25)
        print(f"    {_flag(chord_ok)}acordes   = {r['chord_rate']:.0%}{'  (simultaneidade normal em bateria)' if inst=='drums' else ''}")
        print(f"    {_flag(sust_ok)}sustains  = {r['sustain_rate']:.0%}")
        if r["fret_jump"] is not None:
            print(f"    {_flag(r['fret_jump']<=1.6)}salto médio de fret = {r['fret_jump']:.2f}")
        if r["diffs"] is not None:
            print(f"    {_flag(r['monotonic'])}dificuldades = {r['diffs']}")
    # score-resumo: média dos alinhamentos disponíveis
    aligns = [r["align"] for r in report.values() if r["align"] is not None]
    if aligns:
        print(f"\n  {'-'*60}")
        print(f"  ALINHAMENTO MÉDIO (qualidade musical) = {sum(aligns)/len(aligns):.0%}")
        print(f"  (>60% = notas caem majoritariamente em eventos reais do áudio)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Avalia a qualidade de um chart gerado.")
    ap.add_argument("--chart", required=True, type=Path, help="pasta do chart (notes.mid + stems)")
    args = ap.parse_args(argv)
    print_report(args.chart, evaluate(args.chart))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
