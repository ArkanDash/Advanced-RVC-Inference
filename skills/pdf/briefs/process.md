# Brief: PDF Processing

Work with existing PDFs: extract, merge, split, fill forms, convert formats, or **reformat** with a new design. Usually a Light triage path — except reformat, which escalates to Standard.

---

## Decision Tree

```
User request
  ├─ "Extract text/tables/images"     → §Extract
  ├─ "Merge/split/rotate/crop pages"  → §Pages
  ├─ "Fill a form"                    → §Forms (check fillable first)
  ├─ "Read/write metadata"            → §Metadata
  ├─ "Convert DOCX/PPTX/XLSX to PDF" → §Convert
  │     └─ DOCX with TOC?            → §DOCX Pipeline (5-step)
  ├─ "Redesign/reformat a document"   → §Reformat
  │     └─ With a reference template? → §Template-Guided Reformat
  └─ Edge cases (OCR, encrypt, batch) → load briefs/process-advanced.md
```

---

## Environment Check

```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" env.check
```

Reports availability but does **not** auto-install. Required: Python 3, pikepdf, pdfplumber.

Entry point: `python3 "$PDF_SKILL_DIR/scripts/pdf.py" <group>.<action> [options]`

All commands return JSON on stdout (`{"status": "success", "data": {...}}`) or stderr (`{"status": "error", ...}`).
Exit codes: 0 = success, 1 = bad args, 2 = file not found, 3 = parse error, 4 = operation failed.

---

## §Extract

```bash
# Text (full or page range)
python3 "$PDF_SKILL_DIR/scripts/pdf.py" extract.text report.pdf
python3 "$PDF_SKILL_DIR/scripts/pdf.py" extract.text report.pdf -p 1-3
python3 "$PDF_SKILL_DIR/scripts/pdf.py" extract.text report.pdf -p 1,4,7

# Tables — returns structured JSON with page/rows/cols/data
python3 "$PDF_SKILL_DIR/scripts/pdf.py" extract.table report.pdf

# Images — dumps embedded rasters to directory
python3 "$PDF_SKILL_DIR/scripts/pdf.py" extract.image report.pdf -o ./images/
```

---

## §Pages

```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.merge a.pdf b.pdf -o combined.pdf
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.split book.pdf -o ./chapters/
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.rotate doc.pdf 90 -o rotated.pdf
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.rotate doc.pdf 180 -o rotated.pdf -p 1-3
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.crop doc.pdf 50,50,550,750 -o trimmed.pdf
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.clean doc.pdf -o cleaned.pdf
```

---

## §Metadata

```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" meta.get doc.pdf
python3 "$PDF_SKILL_DIR/scripts/pdf.py" meta.set doc.pdf -o out.pdf -d '{"Title": "Report", "Author": "Jane"}'
python3 "$PDF_SKILL_DIR/scripts/pdf.py" meta.brand doc.pdf -o branded.pdf
```

Recognised keys: `Title`, `Author`, `Subject`, `Keywords`, `Creator`, `Producer`.

`meta.brand` adds standard branding metadata (producer, creator) in one step.

---

## §Forms

### Step 1 — Check if fillable

```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.info input.pdf
```

If `has_fields: true` → **Fillable workflow**. If `false` → **Non-fillable workflow**.

### Fillable Workflow

```bash
# Inspect fields
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.info input.pdf

# Fill (auto-maps "true"/"false" for checkboxes)
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.fill input.pdf -o filled.pdf \
  -d '{"name": "John", "agree": "true", "country": "US"}'
```

**Value rules:**

| Type | Value | Example |
|------|-------|---------|
| text | Free string | `"name": "Jane Doe"` |
| checkbox | `"true"` / `"false"` (auto-converts to PDF states) | `"agree": "true"` |
| radio | One of `radio_options[].value` | `"gender": "/Choice1"` |
| dropdown | One of `choice_options[].value` | `"country": "US"` |

For complex forms, use `form.detail` and `form.render` for deeper inspection:

```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.detail input.pdf -o fields.json   # full field info (types, options, defaults)
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.render input.pdf -o ./pages/       # render pages as PNG for visual check
```

### Non-Fillable Workflow (Annotation-Based)

For PDFs without interactive fields (scanned forms, image-based). All four steps are mandatory.

**Step 1 — Render pages as PNG** (required):
```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.render input.pdf -o ./pages/
```

**Step 2 — Create `fields.json`** with annotation regions.

To determine bbox coordinates: open the rendered PNG in an image viewer or use Python (`from PIL import Image; img = Image.open('page.png'); print(img.size)`) to get pixel dimensions. Then estimate [left, top, right, bottom] in pixels for each field by inspecting the image. The `dims` field must match the PNG dimensions exactly.

```json
{
  "sheet": [
    {
      "pg": 1,
      "dims": [1000, 1400],
      "regions": [
        {
          "id": "last_name",
          "hint": "Last name field",
          "label": {"tag": "Last name", "bbox": [30, 125, 95, 142]},
          "target": {"bbox": [100, 125, 280, 142]},
          "ink": {"value": "Simpson", "size": 14, "color": "000000"}
        }
      ]
    }
  ]
}
```

Schema: `pg` = 1-based page, `dims` = [w,h] in pixels, `label.bbox` / `target.bbox` = [left, top, right, bottom], `ink` = {value, size?, color?, font?}. Label and target boxes must NOT intersect.

**Step 3 — Validate bounding boxes** (required):
```bash
# Auto-check for intersections
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.check-bbox fields.json

# Visual validation (red=target, blue=label)
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.validate 1 fields.json page1.png validation.png
```

Fix any issues, regenerate, re-check. Red rectangles must only cover input areas.

**Step 4 — Fill via annotations**:
```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" form.annotate input.pdf fields.json -o filled.pdf
```

---

## §Reformat

Take an existing document and rebuild it with a new visual design. Content is preserved; layout, typography, and visual treatment are rebuilt from scratch.

```
1. EXTRACT   → Extract content from source (extract.text / extract.table / read directly)
2. STRUCTURE → Organize into sections (headings, body, tables, lists)
3. DELEGATE  → Route to appropriate brief:
                 Structured → briefs/report.md (ReportLab)
                 Visual     → briefs/creative.md (Playwright)
4. BUILD     → Follow the delegated brief's full workflow
5. DELIVER   → New PDF, same content, new design
```

### §Template-Guided Reformat

When user provides a reference PDF to match:

```
1. ANALYZE  → Extract design DNA from template:
               - python3 "$PDF_SKILL_DIR/scripts/pdf.py" meta.get template.pdf       (page size)
               - python3 "$PDF_SKILL_DIR/scripts/pdf.py" extract.image template.pdf   (color samples)
               - python3 "$PDF_SKILL_DIR/scripts/pdf.py" extract.text template.pdf    (text structure)
               - pdftoppm -png -r 150 template.pdf preview           (visual reference)
2. DOCUMENT → Record: page size, margins, colors, fonts, layout grid,
               header/footer pattern, decorative elements
3. DELEGATE → Route to brief WITH design constraints (not brief defaults)
4. BUILD    → Follow brief workflow, constrained to template DNA
5. COMPARE  → pdftoppm both, visually compare side-by-side
```

**Key principles:**
- Match the spirit, not the pixels — exact replication from PDF is impractical
- Prefer original source files (.docx/.html/.tex) over PDF when available
- Declare font substitutions upfront; don't silently fall back
- Template provides design direction, not content — never leak placeholder text

---

## §Convert

### Office → PDF (LibreOffice)

**Simple conversion** (no TOC needed):
```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" convert.office input.docx -o output.pdf
```

**When to use the 5-step DOCX Pipeline instead**: If the DOCX has (or should have) a Table of Contents, always use §DOCX Pipeline below. Signs: the document has 3+ headings, or the user mentions "table of contents" / "TOC", or the document already contains a TOC section. When in doubt, run `python3 "$PDF_SKILL_DIR/scripts/toc_validate.py" fix-docx input.docx -o fixed.docx` — if it returns `no_toc_needed`, a simple conversion is fine.

Or directly:
```bash
soffice --headless --convert-to pdf --outdir ./output input.docx
```

**Supported**: `.docx`, `.doc`, `.odt`, `.rtf`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.ods`, `.csv`, `.html`

**macOS path**: `/Applications/LibreOffice.app/Contents/MacOS/soffice`

**Gotchas:**
- soffice allows only one instance at a time; close existing LibreOffice windows or use `--env:UserInstallation=file:///tmp/libreoffice_tmp`
- Missing Chinese fonts → squares. Ensure SimHei/SimSun are installed.
- Large files (>50MB) may take 1-2 min; set reasonable timeout
- soffice HTML→PDF is inferior to Playwright for complex CSS

**Priority**: Always prefer soffice for Office→PDF (preserves themes, layouts, master slides). Only fall back to python-pptx/python-docx + HTML + Playwright if soffice is unavailable — fidelity will be lower.

### Fallback: Spreadsheet → PDF without LibreOffice

Use openpyxl + HTML + Playwright. Let data shape drive layout:

| Factor | Decision |
|--------|----------|
| Columns ≤ 6 | Portrait |
| Columns > 6 | Landscape |
| Font size | Scale inversely with column count |
| Styling | Follow user requirements or source file style; if unspecified, use defaults from `typesetting/palette.md` |

### §DOCX Pipeline (5-Step with TOC)

For DOCX files that need TOC generation/correction. Required because LibreOffice `--headless` does not recalculate PAGEREF fields.

```
Step 1: soffice     → Convert original DOCX to PDF (pass1)
Step 2: pages.clean → Remove blank pages from pass1
Step 3: fix-docx    → Add/fix TOC with HYPERLINK + PAGEREF + bookmarks
Step 4: fix-pages   → Correct TOC page numbers using pass1 as reference
Step 5: soffice     → Convert final DOCX to PDF + pages.clean
```

**Step 1 — Pass 1 Convert**:
```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" convert.office input.docx -o pass1.pdf
```

**Step 2 — Clean Blank Pages**:
```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.clean pass1.pdf -o pass1_clean.pdf
```
If `blank_pages_removed == 0`, use pass1.pdf directly.

**Step 3 — Fix TOC**:
```bash
python3 "$PDF_SKILL_DIR/scripts/toc_validate.py" fix-docx input.docx -o fixed.docx
```

Auto-detects and fixes: placeholder TOC, stale TOC (>50% drift), empty TOC, missing TOC (≥3 headings). Each entry gets `<w:hyperlink>` + `PAGEREF` + bookmarks for clickable PDF navigation.

Check output `action` field: `fixed` → use fixed.docx, `skipped` → use original, `no_toc_needed` → skip to Step 5 with pass1 PDF.

**Step 4 — Fix Page Numbers**:
```bash
python3 "$PDF_SKILL_DIR/scripts/toc_validate.py" fix-pages fixed.docx pass1_clean.pdf -o final.docx
```

Corrects PAGEREF display text using actual page positions from pass1 + TOC page offset.

**Step 5 — Final Convert + Clean**:
```bash
python3 "$PDF_SKILL_DIR/scripts/pdf.py" convert.office final.docx -o output.pdf
python3 "$PDF_SKILL_DIR/scripts/pdf.py" pages.clean output.pdf -o output_clean.pdf
```

### Post-Conversion Validation (Optional)

```bash
python3 "$PDF_SKILL_DIR/scripts/toc_validate.py" check-conversion final.docx output_clean.pdf
```

Issues caught: `CONV_TOC_LOST` (TOC disappeared), `CONV_HINT_LEAKED` (placeholder text in PDF), `CONV_HEADING_DRIFT` (heading count mismatch).

---

## Caveats

| Topic | Detail |
|-------|--------|
| Encrypted PDFs | Not supported. User must decrypt externally first. |
| < 50 MB | Instant |
| 50–200 MB | 1–2 minutes |
| > 200 MB | Split first, or extend timeout |
| Memory | ~2-3× input file size |
| Merge failure | Partial output may remain; delete and retry |
| Split failure | Some page files may exist; inspect output dir |
| Form fill | Original never modified; always writes new file |

For edge cases (OCR, batch processing, poppler-utils, qpdf, performance tuning), load `briefs/process-advanced.md`.
