"""
song_ini — geração do arquivo song.ini do Clone Hero a partir do áudio original.

Lê as tags do arquivo (mp3/ogg/flac/etc.) via mutagen e produz um song.ini
com apenas os campos obrigatórios definidos pelo projeto.

Tags lidas:
- name           ← title
- artist         ← artist / albumartist / composer
- album          ← album
- year           ← date / year / originaldate (4 primeiros chars)
- genre          ← genre
- track          ← tracknumber / track (parte antes de '/')

Calculadas:
- song_length        ← duração do áudio em ms
- preview_start_time ← 7,25% da duração
- preview_end_time   ← 24% da duração

Fixos:
- charter, frets          ← "Playsic, Rhythm Authors"
- pro_drums               ← True
- star_power_note,
  multiplier_note         ← 116

Uso (CLI):
    python song_ini.py --audio musica.mp3 --out resultados/charts/Foo/song.ini

Uso (API):
    from song_ini import get_audio_metadata, generate_song_ini
    meta = get_audio_metadata("musica.mp3")
    generate_song_ini("musica.mp3", "song.ini")
    generate_validation_song_ini("dados/dataset/MinhaMusica/", "out/song.ini")
"""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mutagen import File as MutagenFile


CHARTER = "Playsic, Rhythm Authors"

# Campos obrigatórios — exatamente estes, nada mais.
SONG_INI_FIELDS = [
    "name", "artist", "album", "year", "genre", "track",
    "song_length", "preview_start_time", "preview_end_time",
    "charter", "frets",
    "pro_drums", "star_power_note", "multiplier_note",
]


# ─────────────────────────────────────────────────────────────────────────────
# Leitura de tags
# ─────────────────────────────────────────────────────────────────────────────
def _first_value(tags, keys: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    if not tags:
        return default
    for key in keys:
        value = tags.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = value[0] if value else None
        if value not in (None, ""):
            return str(value)
    return default


def _parse_track_number(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    value = str(value).strip()
    if "/" in value:
        value = value.split("/", 1)[0]
    try:
        return int(value)
    except ValueError:
        return None


def _parse_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return int(str(value).strip()[:4])
    except ValueError:
        return None


def get_audio_metadata(audio_path: Path | str) -> Dict[str, Any]:
    """Lê tags + duração do arquivo. Retorna dict pronto pra alimentar o song.ini."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")

    audio = MutagenFile(audio_path, easy=True)
    if audio is None:
        raise ValueError(f"Não foi possível ler o arquivo de áudio: {audio_path}")

    tags = audio.tags or {}

    name      = _first_value(tags, ["title"], default=audio_path.stem)
    artist    = _first_value(tags, ["artist", "albumartist", "composer"], default="Unknown")
    album     = _first_value(tags, ["album"], default="Unknown")
    track_raw = _first_value(tags, ["tracknumber", "track"])
    track     = _parse_track_number(track_raw)
    year      = _parse_year(_first_value(tags, ["date", "year", "originaldate", "originalyear"]))
    genre     = _first_value(tags, ["genre"], default="Unknown") or "Unknown"

    song_length   = None
    preview_start = None
    preview_end   = None
    if getattr(audio, "info", None) and hasattr(audio.info, "length"):
        song_length   = round(audio.info.length * 1000)
        preview_start = round(song_length * 0.0725)
        preview_end   = round(song_length * 0.24)

    return {
        "name":               name,
        "artist":             artist,
        "album":              album,
        "year":               year,
        "genre":              genre,
        "track":              track,
        "song_length":        song_length,
        "preview_start_time": preview_start,
        "preview_end_time":   preview_end,
        "charter":            CHARTER,
        "frets":              CHARTER,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geração do song.ini (pipeline final)
# ─────────────────────────────────────────────────────────────────────────────
def _format_value(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def build_song_ini_text(meta: Dict[str, Any]) -> str:
    """Constrói o texto do song.ini a partir do dict de metadados.

    Escreve apenas os campos obrigatórios definidos em SONG_INI_FIELDS.
    """
    values: Dict[str, Any] = {
        "charter":         CHARTER,
        "frets":           CHARTER,
        "pro_drums":       True,
        "star_power_note": 116,
        "multiplier_note": 116,
    }

    for k in ("name", "artist", "album", "year", "genre", "track",
              "song_length", "preview_start_time", "preview_end_time"):
        if meta.get(k) is not None:
            values[k] = meta[k]

    lines = ["[song]"]
    for key in SONG_INI_FIELDS:
        v = values.get(key)
        if v is None:
            continue
        lines.append(f"{key} = {_format_value(v)}")
    return "\n".join(lines) + "\n"


def generate_song_ini(audio_path: Path | str, out_ini: Path | str) -> Path:
    """Pipeline completo: lê tags do áudio + escreve song.ini com campos obrigatórios.

    Args:
        audio_path: caminho do áudio original (mp3, ogg, flac, etc.)
        out_ini:    caminho do song.ini a gerar

    Returns:
        Path do arquivo gerado.
    """
    out_ini = Path(out_ini)
    out_ini.parent.mkdir(parents=True, exist_ok=True)
    meta = get_audio_metadata(audio_path)
    text = build_song_ini_text(meta)
    out_ini.write_text(text, encoding="utf-8")
    return out_ini


# ─────────────────────────────────────────────────────────────────────────────
# Geração do song.ini de validação (notebooks de treinamento)
# ─────────────────────────────────────────────────────────────────────────────
def generate_validation_song_ini(song_folder: Path | str, out_ini: Path | str) -> Path:
    """Gera song.ini de validação para os notebooks de treinamento.

    Copia todos os campos do song.ini original da pasta `song_folder` e
    sobrescreve três campos:
        name    ← "<nome original> (PLAYSIC PREVIEW)"
        charter ← "Playsic"
        frets   ← "Playsic"

    Se o song.ini original não existir, cria um mínimo com name/charter/frets.

    Args:
        song_folder: pasta da música no dataset (deve conter um song.ini).
        out_ini:     caminho do song.ini de saída a gerar.

    Returns:
        Path do arquivo gerado.
    """
    song_folder = Path(song_folder)
    out_ini = Path(out_ini)

    original_ini = song_folder / "song.ini"
    fields: Dict[str, str] = {}

    if original_ini.exists():
        text = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = original_ini.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if text:
            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str  # preserva capitalização das chaves
            try:
                parser.read_string(text)
                if parser.has_section("song"):
                    fields = dict(parser["song"])
            except configparser.Error:
                # fallback: parse manual linha a linha
                for line in text.splitlines():
                    clean = line.strip()
                    if not clean or clean.startswith(("#", ";", "[")) or "=" not in clean:
                        continue
                    k, v = clean.split("=", 1)
                    fields[k.strip()] = v.strip()

    # Sobrescreve os campos de validação
    original_name = fields.get("name", song_folder.name)
    fields["name"]    = f"{original_name} (PLAYSIC PREVIEW)"
    fields["charter"] = "Playsic"
    fields["frets"]   = "Playsic"

    lines = ["[song]"]
    for k, v in fields.items():
        lines.append(f"{k} = {v}")

    out_ini.parent.mkdir(parents=True, exist_ok=True)
    out_ini.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_ini


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gera song.ini a partir do áudio original (lê tags ID3/Vorbis/etc.).",
    )
    p.add_argument("--audio", required=True, type=Path,
                   help="Caminho do arquivo de áudio (mp3, ogg, flac, ...)")
    p.add_argument("--out", required=True, type=Path,
                   help="Caminho do song.ini a gerar")
    p.add_argument("--print-meta", action="store_true",
                   help="Imprime os metadados extraídos antes de salvar")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    meta = get_audio_metadata(args.audio)
    if args.print_meta:
        printable = {k: v for k, v in meta.items() if v is not None}
        print(json.dumps(printable, indent=2, ensure_ascii=False))

    out = generate_song_ini(args.audio, args.out)
    print(f"song.ini gerado em: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
