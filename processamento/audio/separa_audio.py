"""
separa_audio — separação dos stems de uma música usando Demucs htdemucs_6s.

Para uma entrada `musica.mp3`, gera em `<output_dir>/<nome_musica>/`:
    drums.ogg, guitar.ogg, rhythm.ogg, vocals.ogg, song.ogg

- `rhythm.ogg`  = stem `bass.wav` do Demucs convertido para OGG Vorbis.
- `song.ogg`    = mixagem dos 5 stems NÃO-vocais (guitar+bass+drums+piano+other)
                  com `ffmpeg -filter_complex amix`. É o arquivo obrigatório
                  para o Clone Hero.

Os 6 stems do Demucs (`htdemucs_6s`) são: guitar, bass, vocals, drums,
piano, other. Apenas guitar/bass/vocals/drums saem como arquivos individuais.
Piano e other entram somente no song.ogg.

Pré-requisitos do sistema:
    - ffmpeg no PATH
    - demucs instalado (`pip install demucs lameenc`)

Uso (CLI):
    python separa_audio.py --audio musica.mp3 --out resultados/stems/

Uso (API):
    from processamento.audio.separa_audio import separate
    separate("musica.mp3", "resultados/stems/")
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)

# Mapeia o nome do stem do Demucs → nome do arquivo de saída.
STEM_MAP = {
    "guitar": "guitar.ogg",
    "bass":   "rhythm.ogg",
    "vocals": "vocals.ogg",
    "drums":  "drums.ogg",
}
# Stems que entram em song.ogg (instrumental completo).
NON_VOCAL_STEMS = ["guitar", "bass", "drums", "piano", "other"]
DEMUCS_MODEL = "htdemucs_6s"


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_exe(name: str) -> None:
    """Falha cedo se o executável não estiver no PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Executável '{name}' não encontrado no PATH.")


def _wav_to_ogg(wav_path: Path, ogg_path: Path, quality: int = 5) -> None:
    """Converte WAV → OGG Vorbis."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-c:a", "libvorbis", "-q:a", str(quality),
        str(ogg_path),
    ]
    subprocess.run(cmd, check=True)


def _mix_to_ogg(wav_paths: list[Path], ogg_path: Path, quality: int = 5) -> None:
    """Mistura N stems WAV em um único OGG usando amix."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for w in wav_paths:
        cmd += ["-i", str(w)]
    cmd += [
        "-filter_complex", f"amix=inputs={len(wav_paths)}:duration=longest:normalize=0",
        "-c:a", "libvorbis", "-q:a", str(quality),
        str(ogg_path),
    ]
    subprocess.run(cmd, check=True)


def _run_demucs(audio_path: Path, work_dir: Path) -> Path:
    """Roda demucs e retorna a pasta com os stems WAV."""
    cmd = [
        "demucs",
        "-n", DEMUCS_MODEL,
        "--out", str(work_dir),
        str(audio_path),
    ]
    log.info("Rodando Demucs (%s)...", DEMUCS_MODEL)
    subprocess.run(cmd, check=True)
    stems_dir = work_dir / DEMUCS_MODEL / audio_path.stem
    if not stems_dir.is_dir():
        raise RuntimeError(f"Demucs não gerou a pasta esperada: {stems_dir}")
    return stems_dir


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────
def separate(audio_path: Path | str,
             output_dir: Path | str,
             quality: int = 5,
             keep_demucs_temp: bool = False) -> Path:
    """Separa um áudio em 5 OGGs (drums/guitar/rhythm/vocals/song).

    Args:
        audio_path:      caminho do áudio de entrada (mp3, wav, ogg, ...)
        output_dir:      pasta raiz de saída. A função cria `<output_dir>/<nome>/`.
        quality:         qualidade Vorbis (0-10, default 5).
        keep_demucs_temp: se True, mantém a pasta temporária do Demucs.

    Returns:
        Path da pasta criada (`<output_dir>/<nome>/`).
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    output_dir = Path(output_dir).resolve()
    song_name  = audio_path.stem
    out_song   = output_dir / song_name
    out_song.mkdir(parents=True, exist_ok=True)

    _check_exe("demucs")
    _check_exe("ffmpeg")

    # Demucs num diretório temporário
    if keep_demucs_temp:
        tmp_root = output_dir / "_demucs_temp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
        tmp_dir = tmp_root
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="demucs_")
        tmp_dir = Path(tmp_ctx.name)

    try:
        stems_dir = _run_demucs(audio_path, tmp_dir)

        # 1) Stems individuais
        for stem, out_name in STEM_MAP.items():
            src = stems_dir / f"{stem}.wav"
            if not src.exists():
                log.warning("Stem ausente, ignorando: %s", src.name)
                continue
            _wav_to_ogg(src, out_song / out_name, quality=quality)
            log.info("✔ %s → %s", stem.upper(), out_name)

        # 2) Mixagem instrumental → song.ogg
        wavs = [stems_dir / f"{s}.wav" for s in NON_VOCAL_STEMS
                if (stems_dir / f"{s}.wav").exists()]
        if not wavs:
            raise RuntimeError("Nenhum stem não-vocal disponível para mixar song.ogg.")
        _mix_to_ogg(wavs, out_song / "song.ogg", quality=quality)
        log.info("✔ song.ogg (%d stems: %s)",
                 len(wavs), ", ".join(w.stem for w in wavs))

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    log.info("Stems salvos em: %s", out_song)
    return out_song


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Separa uma música em drums/guitar/rhythm/vocals/song.ogg via Demucs htdemucs_6s.",
    )
    p.add_argument("--audio", required=True, type=Path,
                   help="Arquivo de áudio de entrada (mp3/wav/ogg/...)")
    p.add_argument("--out", required=True, type=Path,
                   help="Pasta raiz de saída. Será criada <out>/<nome_musica>/")
    p.add_argument("--quality", type=int, default=5,
                   help="Qualidade Vorbis (0-10, default 5)")
    p.add_argument("--keep-temp", action="store_true",
                   help="Mantém a pasta temporária do Demucs (debug)")
    p.add_argument("--quiet", action="store_true",
                   help="Suprime logs")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    out = separate(args.audio, args.out,
                   quality=args.quality, keep_demucs_temp=args.keep_temp)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
