#!/usr/bin/env python3
"""
PDF Processing Toolkit — All-in-One CLI

Usage:
    python3 pdf.py <command> [args...]

Commands:
    env.check [--json]          Check environment dependencies
    env.fix                     Auto-install missing dependencies

    extract.text  <pdf> [-p pages]
    extract.table <pdf> [-p pages]
    extract.image <pdf> -o <dir>

    pages.merge <pdf>... -o <out>
    pages.split <pdf> -o <dir>
    pages.rotate <pdf> <deg> -o <out> [-p pages]
    pages.crop <pdf> <box> -o <out> [-p pages]

    meta.get <pdf>
    meta.set <pdf> -o <out> -d <json>
    meta.brand <pdf>... [-o <out>] [-t title] [-q]

    form.info <pdf>
    form.fill <pdf> -o <out> -d <json>
    form.detail <pdf> <output.json>
    form.fill-legacy <pdf> <fields.json> <output.pdf>
    form.annotate <pdf> <fields.json> <output.pdf>
    form.render <pdf> <output_dir> [--max-dim N]
    form.validate <page> <fields.json> <input_img> <output_img>
    form.check-bbox <fields.json>

    convert.office <file> [-o <out>]
    convert.html <file> [-o <out>] [--css <file>]
    convert.latex <file> [--runs N] [--keep-logs]

    code.sanitize <file>
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════
#  Section 0: Framework — Output, @cmd registry, CLI parser
# ═══════════════════════════════════════════════════════════════

class Output:
    """Structured JSON output for all subcommands."""

    @staticmethod
    def success(data: dict):
        payload = {"status": "success", "data": data}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        raise SystemExit(0)

    @staticmethod
    def error(error: str, message: str, hint: Optional[str] = None, code: int = 1):
        payload = {"status": "error", "error": error, "message": message}
        if hint is not None:
            payload["hint"] = hint
        sys.stderr.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        raise SystemExit(code)

    @staticmethod
    def check_file(filepath: str) -> Path:
        target = Path(filepath)
        if not target.exists():
            Output.error("FileNotFound", f"File not found: {filepath}", code=2)
        return target


# Command registry
_COMMANDS: Dict[str, Callable] = {}


def cmd(name: str):
    """Decorator to register a CLI command under a dotted namespace."""
    def decorator(fn: Callable) -> Callable:
        _COMMANDS[name] = fn
        return fn
    return decorator


def _pop_flag(argv: list, short: str, long: str, needs_value: bool = True):
    """Extract a flag (and optional value) from *argv* in-place."""
    for idx, tok in enumerate(argv):
        if tok in (short, long):
            argv.pop(idx)
            if needs_value:
                if idx < len(argv):
                    return argv.pop(idx)
                Output.error("MissingArg", f"Flag {long} requires a value")
            return True
    return None


def _load_json_arg(argv: list) -> dict:
    """Read JSON from -d/--data string or -f/--file path."""
    raw = _pop_flag(argv, "-d", "--data")
    if raw is not None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            Output.error("InvalidJSON", f"JSON parse error: {exc}")

    fpath = _pop_flag(argv, "-f", "--file")
    if fpath is not None:
        try:
            with open(fpath) as fh:
                return json.load(fh)
        except Exception as exc:
            Output.error("FileError", f"Failed to read file: {exc}")

    Output.error("MissingData", "Requires --data or --file argument")


def _resolve_page_indices(range_spec: Optional[str], page_count: int) -> List[int]:
    """Turn a human-friendly range string (1-indexed) into a sorted list of 0-based indices."""
    if not range_spec:
        return list(range(page_count))
    indices: Set[int] = set()
    for segment in range_spec.split(","):
        segment = segment.strip()
        if "-" in segment:
            lo, hi = segment.split("-", 1)
            for i in range(int(lo) - 1, min(int(hi), page_count)):
                indices.add(i)
        else:
            val = int(segment) - 1
            if 0 <= val < page_count:
                indices.add(val)
    return sorted(indices)


_SCRIPT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════
#  Section 1: env — environment diagnostics and auto-fix
# ═══════════════════════════════════════════════════════════════

def _probe_cmd(name: str, version_args: Optional[List[str]] = None) -> Tuple[str, str]:
    """Check if a command exists and optionally get its version. Returns (status, detail)."""
    path = shutil.which(name)
    if path is None:
        return ("missing", "")
    if version_args is None:
        return ("ok", "")
    try:
        result = subprocess.run(
            [path] + version_args,
            capture_output=True, text=True, timeout=10
        )
        ver = result.stdout.strip() or result.stderr.strip()
        return ("ok", ver)
    except Exception:
        return ("ok", "")


def _probe_python_module(mod_name: str) -> Tuple[str, str]:
    """Check if a Python module is importable and get its version."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {mod_name}; print(getattr({mod_name}, '__version__', 'installed'))"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return ("ok", result.stdout.strip())
        return ("missing", "")
    except Exception:
        return ("missing", "")


def _probe_node() -> Tuple[str, str]:
    s, d = _probe_cmd("node", ["--version"])
    if s == "ok" and d:
        d = d.lstrip("v")
    return (s, d)


def _probe_python() -> Tuple[str, str]:
    try:
        import platform
        return ("ok", platform.python_version())
    except Exception:
        return ("ok", "")


def _probe_libreoffice() -> Tuple[str, str]:
    candidates = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        os.path.expanduser("~/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        "/usr/bin/soffice",
        "/usr/local/bin/soffice",
        "/usr/lib/libreoffice/program/soffice",
        "/opt/libreoffice/program/soffice",
        "/snap/bin/libreoffice.soffice",
    ]
    for c in candidates:
        if Path(c).is_file():
            return ("ok", "")
    for alias in ("soffice", "libreoffice"):
        if shutil.which(alias):
            return ("ok", "")
    return ("missing", "")


def _probe_tectonic() -> Tuple[str, str]:
    home_bin = Path.home() / "tectonic"
    if home_bin.exists() and os.access(home_bin, os.X_OK):
        return ("ok", "")
    tec_local = _SCRIPT_DIR / "tectonic"
    if tec_local.exists() and os.access(tec_local, os.X_OK):
        return ("ok", "")
    if shutil.which("tectonic"):
        return ("ok", "")
    return ("missing", "")


def _probe_playwright_npm() -> Tuple[str, str]:
    """Check if playwright npm package is installed."""
    try:
        result = subprocess.run(
            ["node", "-e", "console.log(require('playwright/package.json').version)"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return ("ok", result.stdout.strip())
    except Exception:
        pass
    # Try global
    try:
        result = subprocess.run(
            ["npm", "list", "-g", "playwright", "--depth=0"],
            capture_output=True, text=True, timeout=15
        )
        import re as _re
        m = _re.search(r"playwright@(\S+)", result.stdout)
        if m:
            return ("ok", m.group(1))
    except Exception:
        pass
    return ("missing", "")


def _probe_chromium() -> Tuple[str, str]:
    """Check if Playwright Chromium browser is installed."""
    import platform as _platform
    home = Path.home()
    if _platform.system() == "Darwin":
        cache_dir = home / "Library" / "Caches" / "ms-playwright"
    else:
        cache_dir = home / ".cache" / "ms-playwright"
    if cache_dir.is_dir():
        for entry in sorted(cache_dir.iterdir(), reverse=True):
            if "chromium" in entry.name.lower():
                return ("ok", entry.name)
    return ("missing", "")


@cmd("env.check")
def env_check(argv: list):
    """Check environment dependencies."""
    use_json = _pop_flag(argv, "-j", "--json", needs_value=False)

    s_node = _probe_node()
    s_pw = _probe_playwright_npm()
    s_cr = _probe_chromium()
    s_py = _probe_python()
    s_pike = _probe_python_module("pikepdf")
    s_plumb = _probe_python_module("pdfplumber")
    s_lo = _probe_libreoffice()
    s_tec = _probe_tectonic()
    s_pw_py = _probe_python_module("playwright")

    if use_json:
        report = {
            "html_route": {
                "node": s_node[0], "node_version": s_node[1],
                "playwright": s_pw[0], "playwright_version": s_pw[1],
                "chromium": s_cr[0], "chromium_detail": s_cr[1],
            },
            "process_route": {
                "python3": s_py[0], "python3_version": s_py[1],
                "pikepdf": s_pike[0], "pikepdf_version": s_pike[1],
                "pdfplumber": s_plumb[0], "pdfplumber_version": s_plumb[1],
                "playwright_python": s_pw_py[0], "playwright_python_version": s_pw_py[1],
            },
            "optional": {
                "libreoffice": s_lo[0],
                "tectonic": s_tec[0],
            },
        }
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        # Determine exit code
        rc = 0
        for v in [s_node, s_pw, s_cr, s_py, s_pike, s_plumb]:
            if v[0] != "ok":
                rc = 2
                break
        raise SystemExit(rc)

    # Human-readable output
    rc = 0

    def show(name: str, status: Tuple[str, str], optional: bool = False):
        nonlocal rc
        s, d = status
        if s == "ok":
            detail = f" ({d})" if d else ""
            print(f"  \u2713 {name}{detail}")
        elif optional:
            print(f"  \u25cb {name} (optional, not installed)")
        else:
            print(f"  \u2717 {name} (missing)")
            rc = 2

    print("=== PDF Skill Environment ===\n")
    print("--- HTML Route ---")
    show("node", s_node)
    show("playwright", s_pw)
    show("chromium", s_cr)

    print("\n--- Process Route ---")
    show("python3", s_py)
    show("pikepdf", s_pike)
    show("pdfplumber", s_plumb)
    if s_pw_py[0] == "ok":
        print(f"    (playwright-python: {s_pw_py[1]})")

    print("\n--- Optional ---")
    show("libreoffice", s_lo, optional=True)
    show("tectonic", s_tec, optional=True)

    print("\n=== Install Commands ===")
    print("  Node.js:     brew install node (macOS) / apt install nodejs (Ubuntu)")
    print("  Playwright:  npm install -g playwright && npx playwright install chromium")
    print("  Python:      brew install python3 (macOS) / apt install python3 (Ubuntu)")
    print("  pikepdf:     pip install pikepdf pdfplumber --user")
    print("  LibreOffice: brew install --cask libreoffice (macOS)")
    print("  Tectonic:    curl -fsSL https://drop-sh.fullyjustified.net | sh")
    raise SystemExit(rc)


@cmd("env.fix")
def env_fix(argv: list):
    """Auto-install missing Python dependencies."""
    modules = {
        "pikepdf": "pikepdf",
        "pdfplumber": "pdfplumber",
        "pypdf": "pypdf",
        "pdf2image": "pdf2image",
        "PIL": "Pillow",
    }
    installed = []
    for mod, pkg in modules.items():
        s, _ = _probe_python_module(mod)
        if s == "missing":
            print(f"Installing {pkg}...")
            for attempt in (
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                [sys.executable, "-m", "pip", "install", "-q", "--user", pkg],
                [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pkg],
            ):
                result = subprocess.run(attempt, capture_output=True, text=True)
                if result.returncode == 0:
                    installed.append(pkg)
                    break
            else:
                print(f"  Failed to install {pkg}")

    if installed:
        print(f"\nInstalled: {', '.join(installed)}")
    else:
        print("All Python dependencies are already installed.")
    raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════
#  Section 2: extract — text, tables, and embedded images
# ═══════════════════════════════════════════════════════════════

@cmd("extract.text")
def extract_text(argv: list):
    """Pull plain text from selected pages."""
    if not argv:
        Output.error("MissingArg", "pdf path required")
    pdf_path = argv.pop(0)
    page_range = _pop_flag(argv, "-p", "--pages")

    import pdfplumber
    src = Output.check_file(pdf_path)
    try:
        doc = pdfplumber.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    total_pages = len(doc.pages)
    target_pages = _resolve_page_indices(page_range, total_pages)
    char_total = 0
    page_results = []

    for pg_idx in target_pages:
        content = doc.pages[pg_idx].extract_text() or ""
        char_total += len(content)
        page_results.append({"page": pg_idx + 1, "chars": len(content), "text": content})

    doc.close()
    Output.success({
        "total_pages": total_pages,
        "extracted_pages": len(target_pages),
        "total_chars": char_total,
        "pages": page_results,
    })


@cmd("extract.table")
def extract_table(argv: list):
    """Locate and return every table on selected pages."""
    if not argv:
        Output.error("MissingArg", "pdf path required")
    pdf_path = argv.pop(0)
    page_range = _pop_flag(argv, "-p", "--pages")

    import pdfplumber
    src = Output.check_file(pdf_path)
    try:
        doc = pdfplumber.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    total_pages = len(doc.pages)
    target_pages = _resolve_page_indices(page_range, total_pages)
    collected = []

    for pg_idx in target_pages:
        for tbl_num, raw_table in enumerate(doc.pages[pg_idx].extract_tables()):
            if not raw_table:
                continue
            sanitised = [
                [(cell.strip() if cell else "") for cell in row]
                for row in raw_table
            ]
            collected.append({
                "page": pg_idx + 1,
                "table_index": tbl_num,
                "rows": len(sanitised),
                "cols": len(sanitised[0]) if sanitised else 0,
                "data": sanitised,
            })

    doc.close()
    Output.success({
        "total_pages": total_pages,
        "extracted_pages": len(target_pages),
        "total_tables": len(collected),
        "tables": collected,
    })


@cmd("extract.image")
def extract_image(argv: list):
    """Save every embedded raster image to output dir."""
    if not argv:
        Output.error("MissingArg", "pdf path required")
    pdf_path = argv.pop(0)
    out_dir = _pop_flag(argv, "-o", "--output") or "."

    import pikepdf
    src = Output.check_file(pdf_path)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    saved = []
    seq = 0
    _EXT_MAP = {
        "/DCTDecode": "jpg",
        "/FlateDecode": "png",
        "/JPXDecode": "jp2",
    }

    for page_no, pg in enumerate(doc.pages, 1):
        res = pg.get("/Resources")
        if res is None or "/XObject" not in res:
            continue
        for key, ref in res.XObject.items():
            try:
                img_obj = doc.get_object(ref.objgen)
                if img_obj.get("/Subtype") != "/Image":
                    continue
                seq += 1
                w = int(img_obj.get("/Width", 0))
                h = int(img_obj.get("/Height", 0))
                filt = img_obj.get("/Filter")
                ext = _EXT_MAP.get(str(filt) if filt else None, "bin")
                fname = f"page{page_no}_img{seq}.{ext}"
                out_file = dest / fname
                out_file.write_bytes(img_obj.read_raw_bytes())
                saved.append({
                    "page": page_no, "name": str(key), "file": str(out_file),
                    "width": w, "height": h, "format": ext,
                })
            except Exception:
                continue

    doc.close()
    Output.success({"output_dir": str(dest), "total_images": len(saved), "images": saved})


# ═══════════════════════════════════════════════════════════════
#  Section 3: pages — merge, split, rotate, crop
# ═══════════════════════════════════════════════════════════════

@cmd("pages.merge")
def pages_merge(argv: list):
    """Concatenate several PDF files into one."""
    out_path = _pop_flag(argv, "-o", "--output")
    if out_path is None:
        Output.error("MissingArg", "--output is required")
    if not argv:
        Output.error("MissingArg", "At least one PDF required")

    import pikepdf
    sources = [Output.check_file(p) for p in argv]
    handles = []
    try:
        combined = pikepdf.new()
        descriptions = []
        for src in sources:
            handle = pikepdf.open(src)
            handles.append(handle)
            n = len(handle.pages)
            descriptions.append(f"{src.name} ({n} pages)")
            for pg in handle.pages:
                combined.pages.append(pg)
        total = len(combined.pages)
        combined.save(out_path)
        combined.close()
    except Exception as exc:
        Output.error("MergeError", f"Merge failed: {exc}", code=4)
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass

    Output.success({"output": out_path, "total_pages": total, "sources": descriptions})


@cmd("pages.split")
def pages_split(argv: list):
    """Write each page as a separate single-page PDF."""
    if not argv:
        Output.error("MissingArg", "pdf path required")
    pdf_path = argv.pop(0)
    out_dir = _pop_flag(argv, "-o", "--output") or "."

    import pikepdf
    src = Output.check_file(pdf_path)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    generated = []
    base = src.stem
    try:
        for idx, pg in enumerate(doc.pages, 1):
            fp = dest / f"{base}_page{idx:03d}.pdf"
            single_doc = pikepdf.new()
            single_doc.pages.append(pg)
            single_doc.save(fp)
            single_doc.close()
            generated.append(str(fp))
        doc.close()
    except Exception as exc:
        Output.error("SplitError", f"Split failed: {exc}", code=4)

    Output.success({"output_dir": str(dest), "total_pages": len(generated), "files": generated})


@cmd("pages.rotate")
def pages_rotate(argv: list):
    """Rotate selected pages by 90/180/270 degrees."""
    if len(argv) < 2:
        Output.error("MissingArg", "pdf path and degrees required")
    pdf_path = argv.pop(0)
    degrees = int(argv.pop(0))
    out_path = _pop_flag(argv, "-o", "--output")
    if out_path is None:
        Output.error("MissingArg", "--output is required")
    page_range = _pop_flag(argv, "-p", "--pages")

    if degrees not in (90, 180, 270):
        Output.error("InvalidDegrees", "Rotation angle must be 90, 180, or 270")

    import pikepdf
    src = Output.check_file(pdf_path)
    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    targets = _resolve_page_indices(page_range, len(doc.pages))
    try:
        for i in targets:
            existing = int(doc.pages[i].get("/Rotate", 0))
            doc.pages[i]["/Rotate"] = (existing + degrees) % 360
        doc.save(out_path)
        doc.close()
    except Exception as exc:
        Output.error("RotateError", f"Rotation failed: {exc}", code=4)

    Output.success({"output": out_path, "degrees": degrees, "pages_rotated": len(targets)})


@cmd("pages.crop")
def pages_crop(argv: list):
    """Set the media/crop box on selected pages. box = 'left,bottom,right,top' in pt."""
    if len(argv) < 2:
        Output.error("MissingArg", "pdf path and crop box required")
    pdf_path = argv.pop(0)
    box_str = argv.pop(0)
    out_path = _pop_flag(argv, "-o", "--output")
    if out_path is None:
        Output.error("MissingArg", "--output is required")
    page_range = _pop_flag(argv, "-p", "--pages")

    try:
        coords = [float(v.strip()) for v in box_str.split(",")]
        assert len(coords) == 4
        left, bottom, right, top = coords
    except Exception:
        Output.error("InvalidBox", "Invalid crop box format, should be: left,bottom,right,top",
                      hint="Example: 50,50,550,750")

    import pikepdf
    src = Output.check_file(pdf_path)
    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    targets = _resolve_page_indices(page_range, len(doc.pages))
    try:
        arr = pikepdf.Array([left, bottom, right, top])
        for i in targets:
            doc.pages[i].mediabox = arr
            doc.pages[i].cropbox = arr
        doc.save(out_path)
        doc.close()
    except Exception as exc:
        Output.error("CropError", f"Crop failed: {exc}", code=4)

    Output.success({
        "output": out_path,
        "box": {"left": left, "bottom": bottom, "right": right, "top": top},
        "pages_cropped": len(targets),
    })


# ═══════════════════════════════════════════════════════════════
#  Section 4: meta — metadata reading, writing, and branding
# ═══════════════════════════════════════════════════════════════

_XMP_MAPPING = {
    "Title": "dc:title",
    "Author": "dc:creator",
    "Subject": "dc:description",
    "Keywords": "pdf:Keywords",
    "Creator": "xmp:CreatorTool",
    "Producer": "pdf:Producer",
}
_ACCEPTED_KEYS = set(_XMP_MAPPING.keys())


@cmd("meta.get")
def meta_get(argv: list):
    """Read document information and metadata."""
    if not argv:
        Output.error("MissingArg", "pdf path required")

    import pikepdf
    src = Output.check_file(argv[0])
    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    record: dict = {
        "pages": len(doc.pages),
        "pdf_version": str(doc.pdf_version),
    }

    if doc.pages:
        mb = doc.pages[0].mediabox
        record["page_size"] = {
            "width": float(mb[2] - mb[0]),
            "height": float(mb[3] - mb[1]),
            "unit": "pt",
        }

    kv_pairs = {}
    if doc.docinfo:
        for k in doc.docinfo.keys():
            try:
                kv_pairs[str(k).lstrip("/")] = str(doc.docinfo[k])
            except Exception:
                pass
    record["metadata"] = kv_pairs
    record["encrypted"] = doc.is_encrypted
    record["has_form"] = "/AcroForm" in doc.Root
    record["has_outlines"] = "/Outlines" in doc.Root

    doc.close()
    Output.success(record)


@cmd("meta.set")
def meta_set(argv: list):
    """Update XMP + legacy docinfo metadata fields."""
    if not argv:
        Output.error("MissingArg", "pdf path required")
    pdf_path = argv.pop(0)
    out_path = _pop_flag(argv, "-o", "--output")
    if out_path is None:
        Output.error("MissingArg", "--output is required")
    data = _load_json_arg(argv)

    import pikepdf
    src = Output.check_file(pdf_path)
    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    # XMP layer
    with doc.open_metadata() as xmp:
        for raw_key, raw_val in data.items():
            norm = raw_key.title()
            xmp_key = _XMP_MAPPING.get(norm)
            if xmp_key is None:
                continue
            try:
                xmp[xmp_key] = str(raw_val)
            except Exception:
                pass

    # Legacy docinfo layer
    if not doc.docinfo:
        doc.docinfo = pikepdf.Dictionary()
    for raw_key, raw_val in data.items():
        norm = raw_key.title()
        if norm in _ACCEPTED_KEYS:
            doc.docinfo[pikepdf.Name(f"/{norm}")] = pikepdf.String(str(raw_val))

    doc.docinfo[pikepdf.Name("/ModDate")] = pikepdf.String(
        datetime.now().strftime("D:%Y%m%d%H%M%S")
    )

    try:
        doc.save(out_path)
        doc.close()
    except Exception as exc:
        Output.error("SaveError", f"Save failed: {exc}", code=4)

    Output.success({"output": out_path, "updated_fields": list(data.keys())})


@cmd("meta.brand")
def meta_brand(argv: list):
    """Add Z.ai branding metadata to PDF documents."""
    output_path = _pop_flag(argv, "-o", "--output")
    custom_title = _pop_flag(argv, "-t", "--title")
    quiet = _pop_flag(argv, "-q", "--quiet", needs_value=False)

    if not argv:
        Output.error("MissingArg", "At least one PDF file required")

    # Check if output is specified for multiple files
    if output_path and len(argv) > 1:
        Output.error("InvalidArg", "--output can only be used with a single input file")

    from pypdf import PdfReader, PdfWriter

    for input_path in argv:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            continue

        try:
            reader = PdfReader(input_path)
        except Exception as e:
            print(f"Error: Cannot open PDF: {e}", file=sys.stderr)
            continue

        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        # Determine title
        if custom_title:
            title = custom_title
        else:
            original_meta = reader.metadata
            if original_meta and original_meta.title and original_meta.title not in ('(anonymous)', 'unspecified', None):
                title = original_meta.title
            else:
                title = os.path.splitext(os.path.basename(input_path))[0]

        writer.add_metadata({
            '/Title': title,
            '/Author': 'Z.ai',
            '/Creator': 'Z.ai',
            '/Producer': 'http://z.ai',
        })

        # Write output
        out = output_path if (len(argv) == 1 and output_path) else input_path
        try:
            with open(out, "wb") as f:
                writer.write(f)
        except Exception as e:
            print(f"Error: Cannot write output file: {e}", file=sys.stderr)
            continue

        if not quiet:
            print(f"\u2713 Updated metadata for: {os.path.basename(input_path)}")
            print(f"  Title: {title}")
            print(f"  Author: Z.ai")
            print(f"  Creator: Z.ai")
            print(f"  Producer: http://z.ai")
            if out != input_path:
                print(f"  Output: {out}")

    raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════
#  Section 5: form — inspection, filling, annotation, rendering
# ═══════════════════════════════════════════════════════════════

# --- form.info (pikepdf-based) ---

_FIELD_TYPE_MAP = {
    "/Tx": "text",
    "/Sig": "signature",
}


def _classify_field(node) -> str:
    """Map a PDF field type token to a human label."""
    ft = str(node.get("/FT", ""))
    if ft in _FIELD_TYPE_MAP:
        return _FIELD_TYPE_MAP[ft]
    flags = int(node.get("/Ff", 0))
    if ft == "/Btn":
        return "radio" if (flags & (1 << 15)) else "checkbox"
    if ft == "/Ch":
        return "dropdown" if (flags & (1 << 17)) else "listbox"
    return "unknown"


def _extra_props(node, kind: str) -> dict:
    """Gather type-specific metadata (options, checked value, etc.)."""
    props: dict = {}
    if kind == "checkbox":
        ap = node.get("/AP")
        if ap and "/N" in ap:
            states = [str(s) for s in ap["/N"].keys()]
            props["states"] = states
            props["checked_value"] = next((s for s in states if s != "/Off"), states[0] if states else None)
    elif kind in ("dropdown", "listbox"):
        raw_opts = node.get("/Opt")
        if raw_opts:
            props["options"] = [
                {"value": str(item[0]), "label": str(item[1])} if isinstance(item, list) and len(item) >= 2
                else {"value": str(item), "label": str(item)}
                for item in raw_opts
            ]
    elif kind == "radio":
        kids = node.get("/Kids")
        if kids:
            radio_vals = []
            for child in kids:
                ap = child.get("/AP")
                if ap and "/N" in ap:
                    radio_vals.extend(str(k) for k in ap["/N"].keys() if str(k) != "/Off")
            if radio_vals:
                props["options"] = radio_vals
    return props


def _current_value(node):
    v = node.get("/V")
    return str(v) if v is not None else None


def _gather_fields(doc) -> list:
    """Walk the AcroForm field tree iteratively and return a flat list."""
    if "/AcroForm" not in doc.Root:
        return []
    acro = doc.Root.AcroForm
    if "/Fields" not in acro:
        return []

    page_lookup = {pg.objgen: idx for idx, pg in enumerate(doc.pages)}
    results = []
    stack = [(field, "") for field in reversed(list(acro.Fields))]

    while stack:
        node, parent_path = stack.pop()
        name = str(node.get("/T", ""))
        full = f"{parent_path}.{name}" if parent_path else name

        kids = node.get("/Kids")
        if kids and any("/T" in k for k in kids):
            for kid in reversed(list(kids)):
                stack.append((kid, full))
            continue

        kind = _classify_field(node)
        if kind == "unknown":
            continue

        entry = {"id": full, "type": kind}
        val = _current_value(node)
        if val:
            entry["current_value"] = val
        entry.update(_extra_props(node, kind))

        page_ref = node.get("/P")
        if page_ref and hasattr(page_ref, "objgen"):
            pg_num = page_lookup.get(page_ref.objgen)
            if pg_num is not None:
                entry["page"] = pg_num + 1

        results.append(entry)

    return results


@cmd("form.info")
def form_info(argv: list):
    """Return structured JSON describing every form field (pikepdf + check_fillable)."""
    if not argv:
        Output.error("MissingArg", "pdf path required")

    import pikepdf
    src = Output.check_file(argv[0])
    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    fields = _gather_fields(doc)
    if not fields:
        Output.success({"has_fields": False, "count": 0, "fields": [], "hint": "This PDF has no fillable form fields"})
    Output.success({"has_fields": True, "count": len(fields), "fields": fields})


@cmd("form.fill")
def form_fill(argv: list):
    """Write values into a fillable PDF (pikepdf version)."""
    if not argv:
        Output.error("MissingArg", "pdf path required")
    pdf_path = argv.pop(0)
    out_path = _pop_flag(argv, "-o", "--output")
    if out_path is None:
        Output.error("MissingArg", "--output is required")
    data = _load_json_arg(argv)

    import pikepdf
    src = Output.check_file(pdf_path)
    try:
        doc = pikepdf.open(src)
    except Exception as exc:
        Output.error("PDFError", f"Cannot open PDF: {exc}", code=3)

    if "/AcroForm" not in doc.Root or "/Fields" not in doc.Root.AcroForm:
        Output.error("NoForm", "This PDF has no form fields")

    known = {f["id"]: f for f in _gather_fields(doc)}

    # Validation
    issues = []
    for fid, fval in data.items():
        if fid not in known:
            issues.append(f"Field not found: {fid}")
            continue
        fmeta = known[fid]
        ftype = fmeta["type"]
        if ftype == "checkbox" and "states" in fmeta:
            ok_vals = fmeta["states"]
            if fval not in ok_vals and f"/{fval}" not in ok_vals and fval not in ("true", "True", "false", "False", "1", "0"):
                issues.append(f"Invalid value for field {fid}, options: {ok_vals} or true/false")
        if ftype in ("dropdown", "listbox") and "options" in fmeta:
            ok_vals = [o["value"] for o in fmeta["options"]]
            if fval not in ok_vals:
                issues.append(f"Invalid value for field {fid}, options: {ok_vals}")
    if issues:
        Output.error("ValidationError", "Field validation failed", hint="; ".join(issues))

    # Fill
    written = 0

    def _apply(node, parent_path=""):
        nonlocal written
        name = str(node.get("/T", ""))
        full = f"{parent_path}.{name}" if parent_path else name
        kids = node.get("/Kids")
        if kids and any("/T" in k for k in kids):
            for kid in kids:
                _apply(kid, full)
            return
        if full not in data:
            return
        val = data[full]
        kind = _classify_field(node)
        if kind == "checkbox":
            if val in ("true", "True", "1", True):
                ap = node.get("/AP")
                if ap and "/N" in ap:
                    checked_name = next((str(k) for k in ap["/N"].keys() if str(k) != "/Off"), "/Yes")
                    if not checked_name.startswith("/"):
                        checked_name = f"/{checked_name}"
                    node["/V"] = pikepdf.Name(checked_name)
                    node["/AS"] = pikepdf.Name(checked_name)
            else:
                node["/V"] = pikepdf.Name("/Off")
                node["/AS"] = pikepdf.Name("/Off")
        else:
            node["/V"] = pikepdf.String(str(val))
        written += 1

    for field in doc.Root.AcroForm.Fields:
        _apply(field)

    acro = doc.Root.AcroForm
    if "/NeedAppearances" not in acro:
        acro["/NeedAppearances"] = True

    try:
        doc.save(out_path)
    except Exception as exc:
        Output.error("SaveError", f"Save failed: {exc}", code=4)

    Output.success({"output": out_path, "fields_filled": written, "fields_requested": len(data)})


# --- form.detail (pypdf-based detailed field extraction) ---

def _get_full_annotation_field_id(annotation):
    """Build dotted field ID by walking parent chain."""
    components = []
    while annotation:
        field_name = annotation.get('/T')
        if field_name:
            components.append(field_name)
        annotation = annotation.get('/Parent')
    return ".".join(reversed(components)) if components else None


def _make_field_dict(field, field_id):
    field_dict = {"field_id": field_id}
    ft = field.get('/FT')
    if ft == "/Tx":
        field_dict["type"] = "text"
    elif ft == "/Btn":
        field_dict["type"] = "checkbox"
        states = field.get("/_States_", [])
        if len(states) == 2:
            if "/Off" in states:
                field_dict["checked_value"] = states[0] if states[0] != "/Off" else states[1]
                field_dict["unchecked_value"] = "/Off"
            else:
                print(f"Unexpected state values for checkbox `${field_id}`. Its checked and unchecked values may not be correct; if you're trying to check it, visually verify the results.")
                field_dict["checked_value"] = states[0]
                field_dict["unchecked_value"] = states[1]
    elif ft == "/Ch":
        field_dict["type"] = "choice"
        states = field.get("/_States_", [])
        field_dict["choice_options"] = [{
            "value": state[0],
            "text": state[1],
        } for state in states]
    else:
        field_dict["type"] = f"unknown ({ft})"
    return field_dict


def _get_field_info(reader) -> list:
    """Extract detailed field info from a PdfReader, including radio group aggregation."""
    fields = reader.get_fields()

    field_info_by_id = {}
    possible_radio_names: Set[str] = set()

    for field_id, field in fields.items():
        if field.get("/Kids"):
            if field.get("/FT") == "/Btn":
                possible_radio_names.add(field_id)
            continue
        field_info_by_id[field_id] = _make_field_dict(field, field_id)

    radio_fields_by_id: Dict[str, dict] = {}

    for page_index, page in enumerate(reader.pages):
        annotations = page.get('/Annots', [])
        for ann in annotations:
            field_id = _get_full_annotation_field_id(ann)
            if field_id in field_info_by_id:
                field_info_by_id[field_id]["page"] = page_index + 1
                field_info_by_id[field_id]["rect"] = ann.get('/Rect')
            elif field_id in possible_radio_names:
                try:
                    on_values = [v for v in ann["/AP"]["/N"] if v != "/Off"]
                except KeyError:
                    continue
                if len(on_values) == 1:
                    rect = ann.get("/Rect")
                    if field_id not in radio_fields_by_id:
                        radio_fields_by_id[field_id] = {
                            "field_id": field_id,
                            "type": "radio_group",
                            "page": page_index + 1,
                            "radio_options": [],
                        }
                    radio_fields_by_id[field_id]["radio_options"].append({
                        "value": on_values[0],
                        "rect": rect,
                    })

    # Filter fields without location
    fields_with_location = []
    for field_info in field_info_by_id.values():
        if "page" in field_info:
            fields_with_location.append(field_info)
        else:
            print(f"Unable to determine location for field id: {field_info.get('field_id')}, ignoring")

    # Sort by page number, then Y position (flipped), then X
    def sort_key(f):
        if "radio_options" in f:
            rect = f["radio_options"][0]["rect"] or [0, 0, 0, 0]
        else:
            rect = f.get("rect") or [0, 0, 0, 0]
        adjusted_position = [-rect[1], rect[0]]
        return [f.get("page"), adjusted_position]

    sorted_fields = fields_with_location + list(radio_fields_by_id.values())
    sorted_fields.sort(key=sort_key)

    return sorted_fields


@cmd("form.detail")
def form_detail(argv: list):
    """Extract detailed field info (pypdf version) to JSON."""
    if len(argv) < 2:
        Output.error("MissingArg", "Usage: form.detail <pdf> <output.json>")
    pdf_path = argv[0]
    json_output_path = argv[1]

    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    field_info = _get_field_info(reader)
    with open(json_output_path, "w") as f:
        json.dump(field_info, f, indent=2)
    print(f"Wrote {len(field_info)} fields to {json_output_path}")
    raise SystemExit(0)


# --- form.fill-legacy (pypdf version with monkeypatch) ---

def _validation_error_for_field_value(field_info, field_value):
    field_type = field_info["type"]
    field_id = field_info["field_id"]
    if field_type == "checkbox":
        checked_val = field_info["checked_value"]
        unchecked_val = field_info["unchecked_value"]
        if field_value != checked_val and field_value != unchecked_val:
            return f'ERROR: Invalid value "{field_value}" for checkbox field "{field_id}". The checked value is "{checked_val}" and the unchecked value is "{unchecked_val}"'
    elif field_type == "radio_group":
        option_values = [opt["value"] for opt in field_info["radio_options"]]
        if field_value not in option_values:
            return f'ERROR: Invalid value "{field_value}" for radio group field "{field_id}". Valid values are: {option_values}'
    elif field_type == "choice":
        choice_values = [opt["value"] for opt in field_info["choice_options"]]
        if field_value not in choice_values:
            return f'ERROR: Invalid value "{field_value}" for choice field "{field_id}". Valid values are: {choice_values}'
    return None


def _monkeypatch_pypdf_method():
    """
    Workaround for pypdf bug with selection list fields.
    pypdf's get_inherited returns a list of two-element lists for /Opt fields
    in selection lists, causing join() to throw TypeError. We patch it to
    return just the value strings.
    """
    from pypdf.generic import DictionaryObject
    from pypdf.constants import FieldDictionaryAttributes

    original_get_inherited = DictionaryObject.get_inherited

    def patched_get_inherited(self, key: str, default=None):
        result = original_get_inherited(self, key, default)
        if key == FieldDictionaryAttributes.Opt:
            if isinstance(result, list) and all(isinstance(v, list) and len(v) == 2 for v in result):
                result = [r[0] for r in result]
        return result

    DictionaryObject.get_inherited = patched_get_inherited


@cmd("form.fill-legacy")
def form_fill_legacy(argv: list):
    """Fill fillable form fields (pypdf version with monkeypatch)."""
    if len(argv) < 3:
        Output.error("MissingArg", "Usage: form.fill-legacy <pdf> <fields.json> <output.pdf>")
    input_pdf = argv[0]
    fields_json = argv[1]
    output_pdf = argv[2]

    from pypdf import PdfReader, PdfWriter

    _monkeypatch_pypdf_method()

    with open(fields_json) as f:
        fields = json.load(f)

    # Group by page number
    fields_by_page: Dict[int, dict] = {}
    for field in fields:
        if "value" in field:
            field_id = field["field_id"]
            page = field["page"]
            if page not in fields_by_page:
                fields_by_page[page] = {}
            fields_by_page[page][field_id] = field["value"]

    reader = PdfReader(input_pdf)

    has_error = False
    field_info = _get_field_info(reader)
    fields_by_ids = {f["field_id"]: f for f in field_info}
    for field in fields:
        existing_field = fields_by_ids.get(field["field_id"])
        if not existing_field:
            has_error = True
            print(f"ERROR: `{field['field_id']}` is not a valid field ID")
        elif field["page"] != existing_field["page"]:
            has_error = True
            print(f"ERROR: Incorrect page number for `{field['field_id']}` (got {field['page']}, expected {existing_field['page']})")
        else:
            if "value" in field:
                err = _validation_error_for_field_value(existing_field, field["value"])
                if err:
                    print(err)
                    has_error = True
    if has_error:
        raise SystemExit(1)

    writer = PdfWriter(clone_from=reader)
    for page, field_values in fields_by_page.items():
        writer.update_page_form_field_values(writer.pages[page - 1], field_values, auto_regenerate=False)

    writer.set_need_appearances_writer(True)

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"Filled {len(fields_by_page)} page(s) in {output_pdf}")
    raise SystemExit(0)


# --- form.annotate (annotation-based filling with coordinate transform) ---

def _transform_coordinates(bbox, image_width, image_height, pdf_width, pdf_height):
    """Transform bounding box from image coordinates to PDF coordinates."""
    x_scale = pdf_width / image_width
    y_scale = pdf_height / image_height

    left = bbox[0] * x_scale
    right = bbox[2] * x_scale

    # Flip Y coordinates for PDF
    top = pdf_height - (bbox[1] * y_scale)
    bottom = pdf_height - (bbox[3] * y_scale)

    return left, bottom, right, top


def _normalise_fields_json(raw: dict) -> dict:
    """Accept both the current sheet-based schema and the legacy flat schema.

    Current (v2) schema uses ``sheet[].pg/dims/regions[]`` with nested
    ``label.bbox``, ``target.bbox``, ``ink{}``.

    Legacy (v1) schema uses ``pages[]`` + ``form_fields[]`` with flat keys
    like ``entry_bounding_box``, ``label_bounding_box``, ``entry_text{}``.

    Returns a normalised dict in the **v2** internal format used by all
    downstream functions.
    """
    # Already v2
    if "sheet" in raw:
        return raw

    # Convert legacy → v2
    pages_lut = {p["page_number"]: p for p in raw.get("pages", [])}
    sheets: dict = {}  # pg -> sheet entry

    for f in raw.get("form_fields", []):
        pg = f["page_number"]
        if pg not in sheets:
            pi = pages_lut.get(pg, {})
            sheets[pg] = {
                "pg": pg,
                "dims": [pi.get("image_width", 0), pi.get("image_height", 0)],
                "regions": [],
            }
        et = f.get("entry_text", {})
        region = {
            "id": f.get("field_label", f.get("description", "")),
            "hint": f.get("description", ""),
            "label": {"tag": f.get("field_label", ""), "bbox": f.get("label_bounding_box", [0, 0, 0, 0])},
            "target": {"bbox": f.get("entry_bounding_box", [0, 0, 0, 0])},
            "ink": {},
        }
        if isinstance(et, dict) and et.get("text"):
            region["ink"]["value"] = et["text"]
            if "font_size" in et:
                region["ink"]["size"] = et["font_size"]
            if "font_color" in et:
                region["ink"]["color"] = et["font_color"]
            if "font" in et:
                region["ink"]["font"] = et["font"]
        sheets[pg]["regions"].append(region)

    return {"sheet": list(sheets.values())}


@cmd("form.annotate")
def form_annotate(argv: list):
    """Fill a PDF by adding text annotations (FreeText) defined in fields.json."""
    if len(argv) < 3:
        Output.error("MissingArg", "Usage: form.annotate <pdf> <fields.json> <output.pdf>")
    input_pdf = argv[0]
    fields_json_path = argv[1]
    output_pdf = argv[2]

    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import FreeText

    with open(fields_json_path, "r") as f:
        fields_data = _normalise_fields_json(json.load(f))

    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    writer.append(reader)

    # Get PDF dimensions for each page
    pdf_dimensions = {}
    for i, page in enumerate(reader.pages):
        mediabox = page.mediabox
        pdf_dimensions[i + 1] = [mediabox.width, mediabox.height]

    annotations = []
    for page_entry in fields_data["sheet"]:
        pg = page_entry["pg"]
        image_width, image_height = page_entry["dims"]
        pdf_width, pdf_height = pdf_dimensions[pg]

        for region in page_entry["regions"]:
            ink = region.get("ink", {})
            text = ink.get("value", "")
            if not text:
                continue

            transformed_box = _transform_coordinates(
                region["target"]["bbox"],
                image_width, image_height,
                pdf_width, pdf_height
            )

            font_name = ink.get("font", "Arial")
            font_size = str(ink.get("size", 14)) + "pt"
            font_color = ink.get("color", "000000")

            annotation = FreeText(
                text=text,
                rect=transformed_box,
                font=font_name,
                font_size=font_size,
                font_color=font_color,
                border_color=None,
                background_color=None,
            )
            annotations.append(annotation)
            writer.add_annotation(page_number=pg - 1, annotation=annotation)

    with open(output_pdf, "wb") as output:
        writer.write(output)

    print(f"Successfully filled PDF form and saved to {output_pdf}")
    print(f"Added {len(annotations)} text annotations")
    raise SystemExit(0)


# --- form.render (PDF to PNG images) ---

@cmd("form.render")
def form_render(argv: list):
    """Convert each page of a PDF to a PNG image."""
    if len(argv) < 2:
        Output.error("MissingArg", "Usage: form.render <pdf> <output_dir> [--max-dim N]")
    pdf_path = argv.pop(0)
    output_dir = argv.pop(0)
    max_dim_str = _pop_flag(argv, "-m", "--max-dim")
    max_dim = int(max_dim_str) if max_dim_str else 1000

    from pdf2image import convert_from_path

    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=200)

    for i, image in enumerate(images):
        width, height = image.size
        if width > max_dim or height > max_dim:
            scale_factor = min(max_dim / width, max_dim / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height))

        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        image.save(image_path)
        print(f"Saved page {i+1} as {image_path} (size: {image.size})")

    print(f"Converted {len(images)} pages to PNG images")
    raise SystemExit(0)


# --- form.validate (bounding box validation image) ---

@cmd("form.validate")
def form_validate(argv: list):
    """Create validation images with bounding box rectangles."""
    if len(argv) < 4:
        Output.error("MissingArg", "Usage: form.validate <page> <fields.json> <input_img> <output_img>")
    page_number = int(argv[0])
    fields_json_path = argv[1]
    input_path = argv[2]
    output_path = argv[3]

    from PIL import Image, ImageDraw

    with open(fields_json_path, 'r') as f:
        data = _normalise_fields_json(json.load(f))

    img = Image.open(input_path)
    draw = ImageDraw.Draw(img)
    num_boxes = 0

    for page_entry in data["sheet"]:
        if page_entry["pg"] != page_number:
            continue
        for region in page_entry["regions"]:
            target_box = region["target"]["bbox"]
            label_box = region["label"]["bbox"]
            draw.rectangle(target_box, outline='red', width=2)
            draw.rectangle(label_box, outline='blue', width=2)
            num_boxes += 2

    img.save(output_path)
    print(f"Created validation image at {output_path} with {num_boxes} bounding boxes")
    raise SystemExit(0)


# --- form.check-bbox (bounding box overlap detection) ---

@dataclass
class _RectAndField:
    rect: list
    rect_type: str
    field: dict


def get_bounding_box_messages(fields_json_stream) -> List[str]:
    """Check for overlapping bounding boxes. Returns list of messages (max 20)."""
    messages = []
    raw = json.load(fields_json_stream)
    data = _normalise_fields_json(raw)

    total_regions = sum(len(pe["regions"]) for pe in data["sheet"])
    messages.append(f"Read {total_regions} regions across {len(data['sheet'])} page(s)")

    def rects_intersect(r1, r2):
        disjoint_horizontal = r1[0] >= r2[2] or r1[2] <= r2[0]
        disjoint_vertical = r1[1] >= r2[3] or r1[3] <= r2[1]
        return not (disjoint_horizontal or disjoint_vertical)

    has_error = False

    for page_entry in data["sheet"]:
        pg = page_entry["pg"]
        # Collect all rects on this page
        rects_and_regions = []
        for region in page_entry["regions"]:
            rects_and_regions.append(_RectAndField(region["label"]["bbox"], "label", region))
            rects_and_regions.append(_RectAndField(region["target"]["bbox"], "target", region))

        for i, ri in enumerate(rects_and_regions):
            for j in range(i + 1, len(rects_and_regions)):
                rj = rects_and_regions[j]
                if rects_intersect(ri.rect, rj.rect):
                    has_error = True
                    rid = ri.field.get("id", ri.field.get("hint", "?"))
                    rjd = rj.field.get("id", rj.field.get("hint", "?"))
                    if ri.field is rj.field:
                        messages.append(f"FAILURE: pg {pg} — label/target overlap for `{rid}` ({ri.rect}, {rj.rect})")
                    else:
                        messages.append(f"FAILURE: pg {pg} — {ri.rect_type} of `{rid}` ({ri.rect}) overlaps {rj.rect_type} of `{rjd}` ({rj.rect})")
                    if len(messages) >= 20:
                        messages.append("Aborting further checks; fix bounding boxes and try again")
                        return messages

            # Height check for target rects
            if ri.rect_type == "target":
                ink = ri.field.get("ink", {})
                if ink.get("value"):
                    font_size = ink.get("size", 14)
                    entry_height = ri.rect[3] - ri.rect[1]
                    if entry_height < font_size:
                        has_error = True
                        rid = ri.field.get("id", ri.field.get("hint", "?"))
                        messages.append(f"FAILURE: pg {pg} — target box height ({entry_height}) for `{rid}` is shorter than font size ({font_size}). Increase box height or decrease ink.size.")
                        if len(messages) >= 20:
                            messages.append("Aborting further checks; fix bounding boxes and try again")
                            return messages

    if not has_error:
        messages.append("SUCCESS: All bounding boxes are valid")
    return messages


@cmd("form.check-bbox")
def form_check_bbox(argv: list):
    """Check bounding boxes in fields.json for overlaps."""
    if not argv:
        Output.error("MissingArg", "Usage: form.check-bbox <fields.json>")
    with open(argv[0]) as f:
        messages = get_bounding_box_messages(f)
    for msg in messages:
        print(msg)
    raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════
#  Section 6: convert — office, HTML, and LaTeX
# ═══════════════════════════════════════════════════════════════

_CONVERTIBLE_EXTENSIONS = frozenset({
    ".docx", ".doc", ".odt", ".rtf",
    ".pptx", ".ppt", ".odp",
    ".xlsx", ".xls", ".ods", ".csv",
    ".txt", ".html", ".htm",
})

_SOFFICE_CANDIDATES = [
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    os.path.expanduser("~/Applications/LibreOffice.app/Contents/MacOS/soffice"),
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
    "/usr/lib/libreoffice/program/soffice",
    "/opt/libreoffice/program/soffice",
    "/snap/bin/libreoffice.soffice",
]


def _locate_soffice() -> Optional[str]:
    """Search for a working soffice binary."""
    for candidate in _SOFFICE_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    for alias in ("soffice", "libreoffice"):
        found = shutil.which(alias)
        if found:
            return found
    return None


@cmd("convert.office")
def convert_office(argv: list):
    """Convert an office document to PDF via LibreOffice."""
    if not argv:
        Output.error("MissingArg", "input file required")
    src_path = argv.pop(0)
    out_path = _pop_flag(argv, "-o", "--output")

    src = Output.check_file(src_path)
    ext = src.suffix.lower()

    if ext not in _CONVERTIBLE_EXTENSIONS:
        Output.error(
            "UnsupportedFormat",
            f"Unsupported format: {ext}",
            hint=f"Supported formats: {', '.join(sorted(_CONVERTIBLE_EXTENSIONS))}",
        )

    binary = _locate_soffice()
    if binary is None:
        Output.error(
            "DependencyMissing",
            "LibreOffice not found",
            hint="Please install LibreOffice: https://www.libreoffice.org/download/",
        )

    target_dir = Path(out_path).parent if out_path else src.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd_list = [binary, "--headless", "--convert-to", "pdf", "--outdir", str(target_dir), str(src)]

    try:
        proc = subprocess.run(cmd_list, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            Output.error("ConvertError", f"Conversion failed: {proc.stderr.strip() or 'Unknown error'}", code=4)
    except subprocess.TimeoutExpired:
        Output.error("Timeout", "Conversion timeout (>120s)", code=4)
    except Exception as exc:
        Output.error("ConvertError", f"Conversion failed: {exc}", code=4)

    auto_name = target_dir / f"{src.stem}.pdf"
    final = Path(out_path) if out_path else auto_name

    if out_path and final.name != auto_name.name and auto_name.exists():
        auto_name.rename(final)

    if not final.exists():
        Output.error("ConvertError", "Converted PDF file was not generated", code=4)

    Output.success({"input": str(src), "output": str(final), "format": ext})


@cmd("convert.html")
def convert_html(argv: list):
    """Convert HTML to PDF via node html2pdf.js."""
    if not argv:
        Output.error("MissingArg", "input file required")

    js_path = _SCRIPT_DIR / "html2pdf.js"
    if not js_path.exists():
        Output.error("DependencyMissing", "html2pdf.js not found in scripts directory")

    node_path = shutil.which("node")
    if not node_path:
        Output.error("DependencyMissing", "node not found in PATH")

    cmd_list = [node_path, str(js_path)] + argv
    try:
        proc = subprocess.run(cmd_list, timeout=180)
        raise SystemExit(proc.returncode)
    except subprocess.TimeoutExpired:
        Output.error("Timeout", "HTML conversion timeout (>180s)", code=4)
    except SystemExit:
        raise
    except Exception as exc:
        Output.error("ConvertError", f"HTML conversion failed: {exc}", code=4)


# --- convert.latex (tectonic wrapper with log filtering + PDF stats) ---

_NOISE_RE = re.compile(
    r"^note: (?:"
    r'"version 2" Tectonic'
    r"|Running TeX"
    r"|Rerunning TeX because"
    r"|Running xdvipdfmx"
    r"|downloading "
    r"|Skipped writing .* intermediate files"
    r")"
)


def _find_tectonic() -> Optional[str]:
    """Locate the tectonic binary: script_dir first, ~/tectonic, then PATH."""
    local_bin = _SCRIPT_DIR / "tectonic"
    if local_bin.exists() and os.access(local_bin, os.X_OK):
        return str(local_bin)
    home_bin = Path.home() / "tectonic"
    if home_bin.exists() and os.access(home_bin, os.X_OK):
        return str(home_bin)
    system_bin = shutil.which("tectonic")
    return system_bin


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TB"


def _pdf_stats(pdf_file: Path):
    """Return (pages, word_count, image_count) or Nones."""
    try:
        from pypdf import PdfReader
    except ImportError:
        for attempt in (
            [sys.executable, "-m", "pip", "install", "-q", "pypdf"],
            [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "pypdf"],
            [sys.executable, "-m", "pip", "install", "-q", "--user", "pypdf"],
        ):
            if subprocess.run(attempt, check=False, capture_output=True).returncode == 0:
                break
        try:
            from pypdf import PdfReader
        except ImportError:
            return None, None, None

    try:
        reader = PdfReader(str(pdf_file))
        n_pages = len(reader.pages)
        all_text = "".join(p.extract_text() or "" for p in reader.pages)
        n_words = len([w for w in all_text.split() if w.strip()])
        n_images = 0
        for pg in reader.pages:
            xobj = pg.get("/Resources", {}).get("/XObject")
            if xobj:
                obj = xobj.get_object()
                n_images += sum(1 for k in obj if obj[k].get("/Subtype") == "/Image")
        return n_pages, n_words, n_images
    except Exception as exc:
        print(f"Error extracting PDF info: {exc}", file=sys.stderr)
        return None, None, None


def _classify_lines(lines):
    """Bucket raw output into errors / warnings / layout issues."""
    errors, warnings, layout, pdf_note = [], [], [], None
    for raw in lines:
        ln = raw.rstrip()
        if not ln:
            continue
        if _NOISE_RE.match(ln):
            if ln.startswith("note: Writing"):
                pdf_note = ln
            continue
        if ln.startswith("error:"):
            errors.append(ln)
        elif ln.startswith("warning:"):
            warnings.append(ln)
        elif re.search(r"(Overfull|Underfull) \\[hv]box", ln) or re.search(r"(Font shape|Missing character)", ln):
            layout.append(ln)
    return errors, warnings, layout, pdf_note


def _parse_writing_note(note: Optional[str]):
    m = re.search(r"Writing `(.+?)` \((.+?)\)", note or "")
    return (m.group(1), m.group(2)) if m else (None, None)


@cmd("convert.latex")
def convert_latex(argv: list):
    """Compile LaTeX file via tectonic, filter logs, report PDF stats."""
    if not argv:
        Output.error("MissingArg", "tex file required")
    tex_file = argv.pop(0)
    runs_str = _pop_flag(argv, "-r", "--runs")
    runs = int(runs_str) if runs_str else 1
    keep_logs = _pop_flag(argv, "-k", "--keep-logs", needs_value=False)

    tex = Path(tex_file)
    if not tex.exists():
        print(f"\u2717 Error: File not found {tex_file}")
        raise SystemExit(1)

    print(f"Compiling {tex.name}...", flush=True)
    if runs > 1:
        print(f"Running {runs} passes (for cross-references)", flush=True)

    tectonic = _find_tectonic()
    if tectonic is None:
        print("\n\u2717 Error: tectonic command not found")
        print("Please install tectonic: https://tectonic-typesetting.github.io/")
        print("\nHint: If installed at ~/tectonic, ensure it has execute permission:")
        print("  chmod +x ~/tectonic")
        raise SystemExit(1)

    all_lines = []
    ok = False
    for _ in range(runs):
        try:
            proc = subprocess.run(
                [tectonic, "-X", "compile", str(tex)],
                capture_output=True, text=True, timeout=120,
            )
            all_lines.extend((proc.stdout + proc.stderr).splitlines())
            ok = proc.returncode == 0
            if not ok:
                break
        except subprocess.TimeoutExpired:
            print("\n\u2717 Error: Compilation timeout (>2 minutes)")
            raise SystemExit(1)
        except Exception as exc:
            print(f"\n\u2717 Error: {exc}")
            raise SystemExit(1)

    if keep_logs:
        print("\n" + "=" * 50 + "\nFull logs:\n" + "=" * 50)
        for ln in all_lines:
            print(ln)
        print("=" * 50 + "\n")

    errors, warnings, layout, pdf_note = _classify_lines(all_lines)
    noted_name, noted_size = _parse_writing_note(pdf_note)
    pdf_name = noted_name or (tex.stem + ".pdf")
    pdf_path = tex.parent / pdf_name

    print()
    if ok:
        tag = "\u2713 Compilation successful" + (" (with warnings)" if warnings or layout else "")
        print(tag)
    else:
        print("\u2717 Compilation failed")

    if ok and pdf_path.exists():
        print("\n========================\nPDF Information\n========================")
        print(f"File: {pdf_name}")
        print(f"Size: {noted_size or _human_size(pdf_path.stat().st_size)}")
        pages, words, images = _pdf_stats(pdf_path)
        if pages is not None:
            print(f"Pages: {pages}")
        if words is not None:
            print(f"Words: ~{words:,}")
        if images is not None:
            print(f"Images: {images}")

    if layout:
        print(f"\n========================\nLayout Issues ({len(layout)})\n========================")
        for ln in layout:
            print(ln)

    if warnings:
        print(f"\n========================\nWarnings ({len(warnings)})\n========================")
        for ln in warnings:
            print(ln.replace("warning: ", "", 1))

    if errors:
        print("\n========================\nErrors\n========================")
        for ln in errors:
            print(ln.replace("error: ", "", 1))

    if ok and (layout or warnings):
        print()
        print("<system-reminder>")
        print(f"Detected {len(layout)} layout issues and {len(warnings)} warnings.")
        print("These issues affect PDF typesetting quality and must be fixed.")
        print("Do not dismiss with 'warnings don't affect output'. Fix all issues.")
        print("</system-reminder>")

    raise SystemExit(0 if ok else 1)


# ═══════════════════════════════════════════════════════════════
#  Section 7: code — sanitization pipeline for PDF generation code
# ═══════════════════════════════════════════════════════════════

# --- Step 0: restore literal unicode escapes/entities to real chars ---
_RE_UNICODE_ESC = re.compile(r"(\\u[0-9a-fA-F]{4})|(\\U[0-9a-fA-F]{8})|(\\x[0-9a-fA-F]{2})")


def _restore_escapes(s: str) -> str:
    # HTML entities: &#179; &#x2264; &alpha; ...
    s = html.unescape(s)

    # Literal backslash escapes: "\\u00B3" -> "³"
    def _dec(m: re.Match) -> str:
        esc = m.group(0)
        try:
            if esc.startswith("\\u") or esc.startswith("\\U"):
                return chr(int(esc[2:], 16))
            if esc.startswith("\\x"):
                return chr(int(esc[2:], 16))
        except Exception:
            return esc
        return esc

    return _RE_UNICODE_ESC.sub(_dec, s)


# --- Step 1: superscripts/subscripts -> <super>/<sub> ---
_SUPERSCRIPT_MAP: Dict[str, str] = {
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
    "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
    "\u207a": "+", "\u207b": "-", "\u207c": "=", "\u207d": "(", "\u207e": ")",
    "\u207f": "n", "\u1da6": "i",
}

_SUBSCRIPT_MAP: Dict[str, str] = {
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u208a": "+", "\u208b": "-", "\u208c": "=", "\u208d": "(", "\u208e": ")",
    "\u2090": "a", "\u2091": "e", "\u2095": "h", "\u1d62": "i", "\u2c7c": "j",
    "\u2096": "k", "\u2097": "l", "\u2098": "m", "\u2099": "n", "\u2092": "o",
    "\u209a": "p", "\u1d63": "r", "\u209b": "s", "\u209c": "t", "\u1d64": "u",
    "\u1d65": "v", "\u2093": "x",
}


def _replace_super_sub(s: str) -> str:
    out = []
    for ch in s:
        if ch in _SUPERSCRIPT_MAP:
            out.append(f"<super>{_SUPERSCRIPT_MAP[ch]}</super>")
        elif ch in _SUBSCRIPT_MAP:
            out.append(f"<sub>{_SUBSCRIPT_MAP[ch]}</sub>")
        else:
            out.append(ch)
    return "".join(out)


# --- Step 2: symbol fallback for SimHei (protect tags, then replace) ---
_SYMBOL_FALLBACK: Dict[str, str] = {
    # Currently empty - enable entries as needed for fonts missing specific glyphs
}


def _fallback_symbols(s: str) -> str:
    # Protect <super>/<sub> tags from being modified
    placeholders: Dict[str, str] = {}

    def _protect_tag(m: re.Match) -> str:
        key = f"@@TAG{len(placeholders)}@@"
        placeholders[key] = m.group(0)
        return key

    protected = re.sub(r"</?super>|</?sub>", _protect_tag, s)

    # Replace symbols
    protected = "".join(_SYMBOL_FALLBACK.get(ch, ch) for ch in protected)

    # Restore tags
    for k, v in placeholders.items():
        protected = protected.replace(k, v)

    return protected


def sanitize_code(text: str) -> str:
    """
    Full sanitization pipeline for PDF generation code.
    - Restore unicode escapes/entities to real characters
    - Replace superscript/subscript unicode with <super>/<sub>
    - Replace other risky symbols with ASCII/text fallbacks
    """
    s = _restore_escapes(text)
    s = _replace_super_sub(s)
    s = _fallback_symbols(s)
    return s


@cmd("code.sanitize")
def code_sanitize(argv: list):
    """Sanitize Unicode in a Python script for PDF generation."""
    if not argv:
        Output.error("MissingArg", "Usage: code.sanitize <target_script.py>")
    target = argv[0]
    with open(target, "r", encoding="utf-8") as f:
        code = f.read()
    sanitized = sanitize_code(code)
    with open(target, "w", encoding="utf-8") as f:
        f.write(sanitized)
    print(f"Sanitized: {target}")
    raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════
#  Section 8: CLI dispatcher
# ═══════════════════════════════════════════════════════════════

def _usage():
    sys.stdout.write(__doc__.strip() + "\n")
    raise SystemExit(0)


def main():
    tokens = sys.argv[1:]
    if not tokens or tokens[0] in ("-h", "--help"):
        _usage()

    cmd_name = tokens.pop(0)

    # Direct match
    handler = _COMMANDS.get(cmd_name)
    if handler is not None:
        handler(tokens)
        return

    # Two-word match (e.g., "extract text" -> "extract.text")
    if tokens:
        compound = f"{cmd_name}.{tokens[0]}"
        handler = _COMMANDS.get(compound)
        if handler is not None:
            tokens.pop(0)
            handler(tokens)
            return

    # List commands in group
    group_cmds = [k for k in _COMMANDS if k.startswith(cmd_name + ".")]
    if group_cmds:
        print(f"Available commands in '{cmd_name}':")
        for c in sorted(group_cmds):
            print(f"  {c}")
        raise SystemExit(0)

    print(f"Unknown command: {cmd_name}\n")
    _usage()


if __name__ == "__main__":
    main()
