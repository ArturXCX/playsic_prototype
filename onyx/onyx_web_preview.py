"""
onyx_web_preview — gera o web preview (`index.html`) de uma pasta no formato
Clone Hero usando `onyx web-player`.

Uma pasta Clone Hero válida contém `notes.mid` (ou `notes.chart`) + áudio
(stems ou song.ogg) + song.ini.

Reutiliza `resolve_onyx_binary` de `onyx_rb_to_ch.py` para localizar o
binário do Onyx.

Uso (CLI):
    # Uma pasta:
    python onyx_web_preview.py --input resultados/charts/MinhaMusica/ \
                               --output resultados/previews/MinhaMusica/

    # Lote (todas as sub-pastas):
    python onyx_web_preview.py --input resultados/charts/ \
                               --output resultados/previews/ --batch

Uso (API):
    from onyx.onyx_web_preview import build_preview, batch_build
    build_preview(chart_dir, preview_dir)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


# Permite importar resolve_onyx_binary sem precisar de __init__.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from onyx_rb_to_ch import resolve_onyx_binary  # noqa: E402


log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Descoberta de pastas
# ─────────────────────────────────────────────────────────────────────────────
def collect_song_folders(input_folder: Path) -> List[Path]:
    """Retorna sub-pastas contendo notes.mid ou notes.chart."""
    found: List[Path] = []
    for root, dirs, files in os.walk(input_folder):
        rpath = Path(root)
        lower = {f.lower() for f in files}
        if "notes.mid" in lower or "notes.chart" in lower:
            found.append(rpath)
            dirs.clear()  # não desce mais — a sub-pasta já é a música
    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# Geração de preview
# ─────────────────────────────────────────────────────────────────────────────
def build_preview(chart_dir: Path | str,
                  preview_dir: Path | str,
                  onyx: Optional[Path] = None,
                  timeout: int = 300) -> Path:
    """Gera o web preview de uma pasta de chart.

    Args:
        chart_dir:   pasta Clone Hero (com notes.mid + áudio + song.ini)
        preview_dir: pasta de saída onde o `index.html` será escrito
        onyx:        binário do Onyx (opcional, senão resolve automaticamente)
        timeout:     timeout em segundos para a chamada do Onyx

    Returns:
        Path do `index.html` gerado.

    Levanta RuntimeError se o Onyx falhar ou se o `index.html` não aparecer.
    """
    chart_dir   = Path(chart_dir)
    preview_dir = Path(preview_dir)
    if not chart_dir.is_dir():
        raise FileNotFoundError(f"Pasta de chart não existe: {chart_dir}")
    if onyx is None:
        onyx = resolve_onyx_binary()

    preview_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(onyx), "web-player", str(chart_dir), "--to", str(preview_dir)]
    log.info("Gerando preview: %s → %s", chart_dir.name, preview_dir)
    log.debug("CMD: %s", " ".join(cmd))

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timeout ({timeout}s) gerando preview de {chart_dir}") from exc

    if r.returncode != 0:
        err = (r.stderr or r.stdout or "código de saída não-zero").strip()
        raise RuntimeError(f"onyx web-player falhou: {err[:500]}")

    candidates = list(preview_dir.rglob("index.html"))
    if not candidates:
        raise RuntimeError(f"index.html não foi gerado em {preview_dir}")
    log.info("✔ %s", candidates[0])
    return candidates[0]


def batch_build(chart_folders: Iterable[Path],
                output_folder: Path,
                suffix: str = "_preview",
                workers: int = 2,
                onyx: Optional[Path] = None) -> dict:
    """Roda build_preview em paralelo para uma lista de pastas.

    Cada pasta gera `<output_folder>/<nome>_preview/index.html`.

    Retorna {'ok': [Path], 'fail': [(Path, erro)]}.
    """
    chart_folders = list(chart_folders)
    output_folder.mkdir(parents=True, exist_ok=True)
    if onyx is None:
        onyx = resolve_onyx_binary()

    results = {"ok": [], "fail": []}

    def _one(chart_dir: Path):
        preview_dir = output_folder / (chart_dir.name + suffix)
        try:
            html = build_preview(chart_dir, preview_dir, onyx=onyx)
            return chart_dir, True, html
        except Exception as exc:
            return chart_dir, False, exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, c) for c in chart_folders]
        total = len(futures)
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            chart_dir, ok, payload = fut.result()
            if ok:
                log.info("[%d/%d] ✓ %s", i, total, chart_dir.name)
                results["ok"].append(payload)
            else:
                log.warning("[%d/%d] ✗ %s: %s", i, total, chart_dir.name, payload)
                results["fail"].append((chart_dir, str(payload)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gera o web preview (index.html) de pastas Clone Hero usando `onyx web-player`.",
    )
    p.add_argument("--input",  required=True, type=Path,
                   help="Pasta de chart (modo single) ou pasta-mãe (modo --batch)")
    p.add_argument("--output", required=True, type=Path,
                   help="Pasta de saída do preview (single) ou pasta-mãe (batch)")
    p.add_argument("--batch",  action="store_true",
                   help="Processa todas as sub-pastas com notes.mid/notes.chart")
    p.add_argument("--workers", type=int, default=2,
                   help="Jobs paralelos no modo --batch (default 2)")
    p.add_argument("--onyx",   type=Path, default=None,
                   help="Caminho do binário onyx (opcional)")
    p.add_argument("--quiet",  action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    onyx = resolve_onyx_binary(hint=args.onyx)
    log.info("Onyx: %s", onyx)

    if args.batch:
        chart_folders = collect_song_folders(args.input)
        if not chart_folders:
            log.warning("Nenhuma pasta com notes.mid/notes.chart em %s", args.input)
            return 1
        log.info("Encontradas %d música(s)", len(chart_folders))
        res = batch_build(chart_folders, args.output,
                          workers=args.workers, onyx=onyx)
        print(f"\n✓ {len(res['ok'])} OK   ✗ {len(res['fail'])} falhas")
        return 0 if not res["fail"] else 2

    # modo single
    try:
        html = build_preview(args.input, args.output, onyx=onyx)
    except Exception as exc:
        log.error("Falha: %s", exc)
        return 2
    print(f"Preview gerado: {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
