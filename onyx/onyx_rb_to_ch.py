"""
onyx_rb_to_ch — converte arquivos Rock Band para o formato Clone Hero em lote.

Usa o binário do Onyx CLI (Linux AppImage). Detecta pacotes Xbox 360 STFS
(CON/LIVE/PIRS) tanto por extensão quanto por magic bytes, lidando com
rips sem extensão.

Reutilização: `resolve_onyx_binary()` é a função canônica para localizar
e extrair o AppImage do Onyx. Outros scripts (ex: `onyx_web_preview.py`)
devem importar daqui em vez de duplicar.

Uso (CLI):
    python3 onyx/onyx_rb_to_ch.py --input dados/original/ --output dados/pre_dataset/

Uso (API):
    from onyx.onyx_rb_to_ch import resolve_onyx_binary, batch_convert, collect_input_files
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional


log = logging.getLogger(__name__)

# Magic bytes de pacotes Xbox 360 STFS
STFS_MAGIC = {b"CON ", b"LIVE", b"PIRS"}
# Extensões Rock Band reconhecidas
ROCK_BAND_EXTENSIONS = {".con", ".live", ".pkg", ".rba", ".rbproj"}

# Pastas relativas à raiz do repo onde procurar o AppImage
_DEFAULT_APPIMAGE_GLOBS = [
    "onyx-*-linux-x64.AppImage",
    "onyx-*.AppImage",
    "*.AppImage",
]


# ─────────────────────────────────────────────────────────────────────────────
# Resolução do binário Onyx
# ─────────────────────────────────────────────────────────────────────────────
def _repo_root() -> Path:
    """Sobe na hierarquia até encontrar a pasta 'dados/' (marca da raiz do repo)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "dados").is_dir():
            return candidate
    return here.parent  # fallback


def _find_appimage(file_dir: Path) -> Optional[Path]:
    if not file_dir.is_dir():
        return None
    for pat in _DEFAULT_APPIMAGE_GLOBS:
        matches = sorted(file_dir.glob(pat))
        if matches:
            return matches[-1]
    return None


def _extract_appimage(appimage: Path, dest_dir: Path) -> Optional[Path]:
    """Extrai o AppImage em dest_dir. Funciona sem FUSE.

    Retorna o caminho do binário interno `onyx` ou None se falhar.
    """
    try:
        os.chmod(appimage, os.stat(appimage).st_mode |
                 stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    work_dir = appimage.parent
    try:
        subprocess.run(
            [str(appimage), "--appimage-extract"],
            cwd=str(work_dir),
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        log.warning("Falha ao extrair AppImage: %s", exc)
        return None

    extracted = work_dir / "squashfs-root"
    if not extracted.is_dir():
        return None

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    extracted.rename(dest_dir)

    for p in dest_dir.rglob("onyx"):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def resolve_onyx_binary(hint: Optional[Path | str] = None,
                        repo_root: Optional[Path] = None) -> Path:
    """Localiza o binário Onyx CLI.

    Ordem de busca:
        1. `hint` (se passado)
        2. variável de ambiente $ONYX_CLI
        3. `onyx` (ou `onyx.exe`) no PATH
        4. binário já extraído em `<repo>/onyx/onyx_extracted/`
        5. AppImage em `<repo>/onyx/file/onyx-*-linux-x64.AppImage` (extrai)

    Levanta FileNotFoundError se nada for encontrado.
    """
    candidates: List[Path] = []
    if hint:
        candidates.append(Path(hint))

    env = os.environ.get("ONYX_CLI")
    if env:
        candidates.append(Path(env))

    which = shutil.which("onyx") or shutil.which("onyx.exe")
    if which:
        candidates.append(Path(which))

    root = repo_root or _repo_root()
    extract_dir = root / "onyx" / "onyx_extracted"
    if extract_dir.is_dir():
        for p in extract_dir.rglob("onyx"):
            if p.is_file() and os.access(p, os.X_OK):
                candidates.append(p)
                break

    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            log.debug("Onyx resolvido: %s", c)
            return c

    # nada encontrado → tenta extrair o AppImage
    appimage = _find_appimage(root / "onyx" / "file")
    if appimage is not None:
        log.info("Extraindo AppImage: %s", appimage.name)
        bin_path = _extract_appimage(appimage, extract_dir)
        if bin_path is not None:
            return bin_path

    raise FileNotFoundError(
        "Não foi possível localizar o binário Onyx CLI.\n"
        "Opções:\n"
        "  • Defina $ONYX_CLI no ambiente.\n"
        "  • Instale o `onyx` no PATH.\n"
        f"  • Coloque o AppImage em {(root / 'onyx' / 'file')}/."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Descoberta de arquivos
# ─────────────────────────────────────────────────────────────────────────────
def is_stfs_package(path: Path) -> bool:
    """True se o arquivo começa com um magic header Xbox 360 STFS."""
    try:
        with open(path, "rb") as f:
            return f.read(4) in STFS_MAGIC
    except (OSError, IOError):
        return False


def collect_input_files(input_folder: Path) -> List[Path]:
    """Varre input_folder e retorna todos os arquivos Rock Band detectados.

    Detecta tanto por extensão (`.con`, `.live`, `.pkg`, `.rba`, `.rbproj`)
    quanto por magic bytes para arquivos sem extensão.
    """
    found: List[Path] = []
    for root, _dirs, files in os.walk(input_folder):
        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext in ROCK_BAND_EXTENSIONS:
                found.append(fpath)
            elif ext == "" and is_stfs_package(fpath):
                log.debug("Detectado via magic bytes: %s", fpath.name)
                found.append(fpath)
    return sorted(found)


def safe_output_name(input_file: Path) -> str:
    stem = input_file.stem if input_file.suffix else input_file.name
    for bad in r'\/:*?"<>|':
        stem = stem.replace(bad, "_")
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# Conversão
# ─────────────────────────────────────────────────────────────────────────────
def convert_one(onyx: Path,
                input_file: Path,
                output_folder: Path,
                dry_run: bool = False,
                timeout_import: int = 120,
                timeout_build: int = 300) -> tuple[bool, str]:
    """Converte um único arquivo Rock Band em pasta Clone Hero.

    Pipeline (Onyx CLI):
        1. `onyx import <input> --to <tmp>`            → projeto onyx
        2. injeta target `ch-out:\\n    game: ps` no song.yml
        3. `onyx build <song.yml> --target ch-out --to <dest>`

    Retorna (sucesso, mensagem).
    """
    out_name = safe_output_name(input_file)
    dest = output_folder / out_name
    if dry_run:
        log.info("[DRY-RUN] %s → %s", input_file.name, dest)
        return True, "dry-run"

    log.info("Convertendo: %s", input_file.name)
    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)

            # 1) import
            r = subprocess.run(
                [str(onyx), "import", str(input_file), "--to", str(tmp)],
                capture_output=True, text=True, timeout=timeout_import,
            )
            if r.returncode != 0:
                return False, (r.stderr or r.stdout).strip()

            # 2) injeta target PS no song.yml
            yml = tmp / "song.yml"
            if not yml.exists():
                return False, "song.yml não foi gerado pelo import"
            yml.write_text(yml.read_text() + "\n  ch-out:\n    game: ps\n")

            # 3) build
            dest.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(
                [str(onyx), "build", str(yml),
                 "--target", "ch-out", "--to", str(dest)],
                capture_output=True, text=True, timeout=timeout_build,
            )
            if r.returncode != 0:
                return False, (r.stderr or r.stdout).strip()

            generated = list(dest.iterdir())
            if not generated:
                return False, "Build concluiu mas nenhum arquivo gerado"
            return True, f"{len(generated)} arquivo(s)"

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as exc:
        return False, str(exc)


def batch_convert(onyx: Path,
                  files: Iterable[Path],
                  output_folder: Path,
                  workers: int = 10,
                  dry_run: bool = False) -> dict:
    """Roda convert_one em paralelo. Retorna {'ok': [...], 'fail': [...]}."""
    files = list(files)
    output_folder.mkdir(parents=True, exist_ok=True)
    results = {"ok": [], "fail": []}
    total = len(files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_file = {
            pool.submit(convert_one, onyx, f, output_folder, dry_run): f
            for f in files
        }
        for i, fut in enumerate(concurrent.futures.as_completed(future_to_file), 1):
            src = future_to_file[fut]
            try:
                ok, msg = fut.result()
            except Exception as exc:
                ok, msg = False, str(exc)
            log.info("[%d/%d] %s  %s", i, total, "✓" if ok else "✗", src.name)
            if not ok:
                log.warning("    %s", msg)
                results["fail"].append(src)
            else:
                results["ok"].append(src)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Converte arquivos Rock Band para o formato Clone Hero em lote (via Onyx CLI).",
    )
    p.add_argument("--input",   required=True, type=Path,
                   help="Pasta com arquivos Rock Band (.con/.live/.pkg/.rba ou sem extensão STFS)")
    p.add_argument("--output",  required=True, type=Path,
                   help="Pasta de saída (uma sub-pasta por música)")
    p.add_argument("--onyx",    type=Path, default=None,
                   help="Caminho do binário onyx (opcional; senão tenta resolver automaticamente)")
    p.add_argument("--workers", type=int, default=10, help="Jobs paralelos (default 10)")
    p.add_argument("--dry-run", action="store_true", help="Apenas lista o que faria")
    p.add_argument("--quiet",   action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.is_dir():
        raise FileNotFoundError(f"Pasta de entrada não existe: {args.input}")

    onyx = resolve_onyx_binary(hint=args.onyx)
    log.info("Onyx: %s", onyx)

    files = collect_input_files(args.input)
    if not files:
        log.warning("Nenhum arquivo Rock Band encontrado em %s", args.input)
        return 1

    by_ext   = sum(1 for f in files if f.suffix.lower() in ROCK_BAND_EXTENSIONS)
    by_magic = len(files) - by_ext
    log.info("Encontrados %d arquivo(s): %d por extensão, %d por magic bytes",
             len(files), by_ext, by_magic)

    res = batch_convert(onyx, files, args.output,
                        workers=args.workers, dry_run=args.dry_run)
    print("─" * 50)
    print(f"✓ {len(res['ok'])} OK   ✗ {len(res['fail'])} falhas")
    if res["fail"]:
        print("\nFalhas:")
        for f in res["fail"]:
            print(f"  {f}")
    return 0 if not res["fail"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
