# Beamer Module: LaTeX Academic Presentations

> **Output formats: PDF and HTML only — never `.pptx`.**
> Activate for ALL academic, scientific, research, or scholarly presentation requests,
> including reading a paper to make slides (paper presentations, academic PPTs, thesis defenses, etc.).

## Pipeline

```
User request → LaTeX Beamer (.tex) → Tectonic → PDF
```

Read `references/beamer.md` for theme catalogue, overlay syntax, and content templates.
For non-Beamer LaTeX documents (papers, theses), read `references/latex.md`.

---

## Workflow

### 1. Analyze Source Material

If given a paper/PDF to summarize:

```bash
python3 scripts/pdf.py extract.text paper.pdf
python3 scripts/pdf.py extract.table paper.pdf
python3 scripts/pdf.py extract.image paper.pdf -o ./images_out/
```

For arXiv papers, prefer extracting clean figures from the HTML version:

```python
# Download individual figure images (no surrounding text)
import urllib.request, re
html_url = "https://arxiv.org/html/<PAPER_ID>v1"
req = urllib.request.Request(html_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    html = resp.read().decode('utf-8')
# Find figure images: x1.png, x2.png, ...
for m in re.findall(r'src="(x\d+\.png)"', html):
    urllib.request.urlretrieve(f"{html_url}/{m}", m)
```

Map figure images to figure numbers by parsing `<figure>` elements and their captions.
**Avoid cropping from rendered PDF pages** — the result often includes surrounding text. If unavoidable, crop precisely so zero non-figure content remains (see Section 6, Source Priority).

### 2. Write the Beamer `.tex`

#### Pre-write Checklist (MANDATORY — read before writing any content)

Before writing any slide content, internalize these rules:
1. **Every slide must be self-contained.** Never write "see original paper", "refer to Figure X in the paper", "详见原文", or any similar reference that defers content to the source. If it's worth mentioning, show it.
2. **All figures/tables get sequential numbering** (Figure 1, 2, 3... / Table 1, 2, 3...) in order of appearance — never copy the paper's original numbers.
3. **Slide titles must state conclusions**, not describe content (see Section 5).
4. **Choose the correct navigation scheme** before writing the preamble (see Section 7). Paper-based slides must use `paper-navbar.tex`; general presentations must use `progress-navbar.tex`.

Structure:

```
Title → Outline → Background → Key Ideas → Method →
Experiments/Results → Ablation → Comparison → Conclusion → Q&A
```

Slide count heuristics:

| Instruction | Slides |
|---|---|
| Explicit count | Match exactly |
| No count | 1–2 per point; 12–20 total |
| "Brief" / "Short" | 8–12 |
| "Full" / "Detailed" | 20–30 |

Content density: ≤ 6 bullets per slide, ≤ 6 words per bullet.

**Section count:** Aim for **4–6 sections**. Each section generates an automatic divider page, so more than 6 sections means too many empty pages. If the content naturally has 7+ topics, merge related ones. **Divider pages should not exceed 20% of total slide count.**

### 3. tcolorbox Block Layout Rules (Critical)

**When to use native `block` vs `tcolorbox`:**

Native beamer `block`/`alertblock`/`exampleblock` are fine for **simple, standalone** blocks where you don't need precise control. But switch to `tcolorbox` (`ablock`/`fblock`/etc.) when any of the following apply:

| Scenario | Use |
|---|---|
| Simple standalone block, no special alignment needs | Native `block` is OK |
| Side-by-side blocks that must have **equal height** | `fblock`/`falertblock` (tcolorbox) — native blocks cannot guarantee equal heights |
| Need precise **background/frame color** beyond theme defaults | tcolorbox — native blocks inherit theme colors and are hard to override per-instance |
| Need consistent **padding, border-radius, shadow** control | tcolorbox — `blocksty` gives uniform styling |
| Vertically **stacked blocks** that need visual consistency | `ablock`/`aalertblock` (tcolorbox) — auto-height with uniform styling |

**Rule of thumb:** If blocks appear side-by-side, need custom colors, or must look visually consistent across slides, use tcolorbox. For a quick single block on a simple slide, native `block` is acceptable.

**Choose block type by layout relationship:**

| Layout | Block Type | Rule |
|---|---|---|
| **Side-by-side** (left–right parallel) | `fblock` / `falertblock` (fixed height) | Both blocks **must** use the **same height** parameter |
| **Stacked** (top–bottom vertical) | `ablock` / `aalertblock` (auto height) | Let content determine height naturally |

**Side-by-side** = two blocks that are conceptually parallel (e.g., "Problem vs Hypothesis", "Root Traits vs Shoot Traits", "Innovation vs Conclusion").
**Stacked** = two blocks in one column, one above the other (e.g., image-text pages with "Key Findings" above "Genotype Differences" in the right column).

Define in preamble:

```latex
\usepackage[most]{tcolorbox}
\tcbset{
  blocksty/.style={
    boxrule=0.4pt, arc=1.5pt,
    top=2pt, bottom=2pt, left=4pt, right=4pt,
    fonttitle=\bfseries\small, before upper={\small}, valign=top,
  }
}
% ── Custom colors (MUST define all three in preamble) ──────────────
\definecolor{myblue}{HTML}{2B5EA7}
\definecolor{myred}{HTML}{C0392B}
\definecolor{mygreen}{HTML}{27AE60}
% ── Fixed-height (side-by-side equal-height layouts) ───────────────
\newtcolorbox{fblock}[2][]{blocksty, colback=myblue!8, colframe=myblue!85,
  coltitle=white, title={#2}, height=#1}
\newtcolorbox{falertblock}[2][]{blocksty, colback=myred!8, colframe=myred!80,
  coltitle=white, title={#2}, height=#1}
\newtcolorbox{fexampleblock}[2][]{blocksty, colback=mygreen!8, colframe=mygreen!80,
  coltitle=white, title={#2}, height=#1}
% Auto-height (stacked / standalone blocks)
\newtcolorbox{ablock}[1]{blocksty, colback=myblue!8, colframe=myblue!85,
  coltitle=white, title={#1}}
\newtcolorbox{aalertblock}[1]{blocksty, colback=myred!8, colframe=myred!80,
  coltitle=white, title={#1}}
\newtcolorbox{aexampleblock}[1]{blocksty, colback=mygreen!8, colframe=mygreen!80,
  coltitle=white, title={#1}}
```

**Side-by-side example** — both columns get **identical height**:

```latex
\begin{columns}[T]
  \column{0.48\textwidth}
  \begin{fblock}[5.0cm]{Left Title}
    Content...
  \end{fblock}
  \column{0.48\textwidth}
  \begin{falertblock}[5.0cm]{Right Title}
    Content...
  \end{falertblock}
\end{columns}
```

**Stacked example** — auto height, no fixed parameter:

```latex
\column{0.46\textwidth}
\begin{ablock}{Key Findings}
  \begin{itemize} ... \end{itemize}
\end{ablock}
\begin{aalertblock}{Genotype Differences}
  Content...
\end{aalertblock}
```

Height guidelines for fblock: 5.0cm for full-height side-by-side columns.
**Always test**: if content overflows, increase height. If frame overflows (`Overfull \vbox`), decrease height, reduce content, or split into two slides. **Never use `[allowframebreaks]`** — it produces unpredictable page breaks in Beamer.

#### Block Mixing Rules (MANDATORY)

1. **No mixing tcolorbox and native blocks on the same slide.** Within a single frame, use EITHER tcolorbox blocks (`fblock`/`ablock`/`falertblock`/`aalertblock`/etc.) OR native Beamer blocks (`block`/`alertblock`/`exampleblock`), but **never both**. Mixing produces inconsistent styling (different padding, border radius, shadows).

2. **Formal environments are exempt.** LaTeX theorem-like environments (`theorem`, `definition`, `lemma`, `proof`, `corollary`, etc.) are semantically distinct from layout blocks and MAY coexist with tcolorbox blocks on the same slide. Use them when the content is genuinely a theorem/definition/etc.

3. **General-purpose blocks (e.g., "Key Advantages", "Core Idea", "Key Takeaway") must use tcolorbox**, not native blocks. Reserve native `block`/`alertblock`/`exampleblock` only for slides that have no tcolorbox blocks at all.

#### Stacked Blocks: Equal Total Height Across Columns (MANDATORY)

When a slide has two columns with stacked blocks, the **total height** of each column must be equal. Total height = sum of all block heights + inter-block spacing in that column.

This does NOT mean every individual block must be the same height — blocks within a column can have different heights. But the left column's total occupied height must match the right column's total occupied height, so the slide looks visually balanced.

**How to achieve this:**
- Use `fblock` with explicit height parameters for all blocks
- Calculate: left column total = `h1 + gap + h2` ; right column total = `h3 + gap + h4`
- Ensure both totals are equal (e.g., both sum to 5.6cm)
- Adjust individual block heights to fit their content while maintaining the total

```latex
% Example: 2 blocks per column, equal total height (5.6cm each)
\begin{columns}[T]
  \column{0.48\textwidth}
  \begin{fblock}[3.2cm]{Block A — more content}
    Longer content...
  \end{fblock}
  \vspace{0.2cm}
  \begin{falertblock}[2.2cm]{Block B — less content}
    Shorter content...
  \end{falertblock}
  % Total: 3.2 + 0.2 + 2.2 = 5.6cm

  \column{0.48\textwidth}
  \begin{fblock}[2.6cm]{Block C}
    Content...
  \end{fblock}
  \vspace{0.2cm}
  \begin{falertblock}[2.8cm]{Block D}
    Content...
  \end{falertblock}
  % Total: 2.6 + 0.2 + 2.8 = 5.6cm
\end{columns}
```

### 3.5 Content Overflow Detection and Repair (MANDATORY)

After every compilation, check for content overflow. This is the most common cause of ugly slides.

#### Detection

1. **Compiler warnings** — scan output for `Overfull \vbox` and `Overfull \hbox` warnings. These mean content exceeds the frame boundary.
2. **Visual inspection** — open the PDF and check every slide for:
   - Text being cut off at the bottom or right edge
   - Block content truncated (last bullet missing or partially visible)
   - Elements overlapping the footer bar
   - Unbalanced columns (one column visually much heavier than the other)
   - **Side-by-side block near-overflow** — content touching or nearly touching the block's bottom edge (even if not technically clipped, it looks cramped and is one line away from overflow)
3. **Element occlusion / overlap check** — verify that no element is partially or fully hidden behind another element. This is a **silent failure** — the compiler will NOT produce any warning when elements overlap. Common scenarios:
   - Right-column blocks covering part of a left-column image (especially wide composite figures with multiple sub-panels)
   - A block or text extending beyond its column boundary and overlapping the adjacent column
   - Footer or decorative elements covering bottom-edge content

   **How to detect:** Visually scan every slide in the compiled PDF. For image+block layouts, confirm that the entire image (all sub-panels, axis labels, legends) is fully visible and not occluded by any block, text, or other element.

   **Repair strategy for occlusion:**

   | Priority | Fix | When to use |
   |---|---|---|
   | 1 | **Give the image its own full-width slide** | Composite figures (3+ sub-panels) that need full width to be legible |
   | 2 | **Select fewer sub-panels** | Show only the 1-2 most important panels on this slide, move rest to appendix |
   | 3 | **Adjust column width ratio** | Give the image column more space (e.g., 0.58 left / 0.38 right) |
   | 4 | **Reduce image width + increase right margin** | Shrink image so it fits entirely within its column boundary |

   **Prevention:** Before choosing a layout, estimate whether the image content fits within the allocated column width. Wide composite figures (4+ panels side-by-side) almost never fit in a half-width column — plan a full-width layout from the start.

#### Repair Strategy

When overflow is detected, apply fixes in this priority order:

| Priority | Fix | When to use |
|---|---|---|
| 1 | **Reduce content** | Remove low-value bullets, shorten text, summarize |
| 2 | **Shrink font locally** | Use `\small` or `\footnotesize` for the overflowing slide only |
| 3 | **Adjust block heights** | Reduce `fblock` height parameters, rebalance column totals |
| 4 | **Rebalance columns** | Move content between left/right columns; adjust width ratio |
| 5 | **Split into two slides** | When content genuinely cannot fit — split by sub-topic |

**NEVER ignore overflow.** Every slide must display all its content completely within the frame boundary. If a fix introduces new overflow elsewhere, iterate until all slides are clean.

#### Verification Loop

```
Compile → Check warnings → Visual inspect PDF → Fix issues → Re-compile → Repeat until clean
```

This loop is mandatory. Do not deliver a PDF that has not passed visual inspection.

### 4. Figure and Table Numbering (CRITICAL)

In presentations, use **sequential numbering** (Figure 1, 2, 3...) independent of the source paper. **NEVER copy the original paper's figure/table numbers** — always renumber them consecutively (1, 2, 3...) in the order they appear in the slides. This is a presentation, not a paper reproduction. Add captions below figures and above tables.

#### No "See Original" References (ABSOLUTELY FORBIDDEN)

**Slides must be self-contained.** Never use placeholder text like "see original paper Figure X", "refer to the paper for details", "详见原文", or any similar reference that defers content to the source document. If a figure or result is worth mentioning on a slide, it must be **actually shown** — either as an extracted image, a simplified TikZ redraw, or a text/bullet summary of the key information. A slide that says "see original" is an incomplete slide.

#### All Images and Diagrams: Numbering + Caption + Centering (MANDATORY)

**Every visual element** on a slide — whether it is an `\includegraphics` image, a TikZ diagram, or a table — **MUST** have:

1. **Sequential numbering** — Figure 1, Figure 2, ... / Table 1, Table 2, ... in order of appearance
2. **Descriptive caption** — below figures/TikZ, above tables
3. **Centered display** — wrapped in `\begin{center}...\end{center}`

**No exceptions.** A figure or TikZ diagram without a number and caption is incomplete.

**For `\includegraphics` images:**

```latex
\begin{center}
  \includegraphics[width=\textwidth,height=0.45\textheight,keepaspectratio]{fig.jpg}
  \par\vspace{0.15em}
  {\scriptsize \textbf{Figure 1.} Description of the figure.}
\end{center}
```

**For TikZ diagrams:**

```latex
\begin{center}
  \begin{tikzpicture}[...]
    % ... diagram code ...
  \end{tikzpicture}
  \par\vspace{0.15em}
  {\scriptsize \textbf{Figure 2.} Simplified architecture overview.}
\end{center}
```

**For tables:**

```latex
\begin{center}
  {\small \textbf{Table 1.} Description of the table.}
\end{center}
\vspace{0.3em}
\begin{tabular}{...}
```

### 5. Slide Titles

Frame titles should convey the **conclusion or key message** of the slide, not describe the figure or repeat the figure caption.

| Bad (descriptive) | Good (conclusion-driven) |
|---|---|
| "RCS in Cotton Root Cross-sections" | "First Confirmation of RCS in Cotton" |
| "Effect of GhSAG12 Silencing on Drought Tolerance" | "Silencing RCS-related Gene Significantly Reduces Drought Tolerance" |
| "Relationship Between RCS and Endogenous Hormones" | "Endogenous Hormones Change Significantly with RCS Progression" |
| "Five Endogenous Hormones During RCS" | "GA, ZR, IAA, BR, ABA All Decline with RCS Progression" |
| "Research Background" | "Drought is the Primary Yield Constraint for Cotton" |

**Self-check (MANDATORY):** After writing all slides, review every `\frametitle{}` and ask: *"Does this title tell the audience what to take away, or does it just describe what's on the slide?"* If descriptive, rewrite it as a conclusion. Background/outline/Q&A slides are exempt.

### 6. Image Handling Guidelines

#### Sizing to Prevent Overflow

Always use `height` + `keepaspectratio` to constrain images within the frame. Recommended max heights:

| Layout | Max image height |
|---|---|
| Full-page image (centered) | `0.55\textheight` |
| Image in left column (with text blocks on right) | `0.45\textheight` |

Always leave room for the figure caption below and the footer bar. **Never rely on `width` alone** — tall images will overflow the frame.

#### Source Priority

| Priority | Source | When to use |
|---|---|---|
| 1 (Best) | Original images from arXiv HTML version | Preferred for arXiv papers — download vector/high-res bitmaps directly |
| 2 | Supplementary materials from authors | High-quality original figures |
| 3 | `pdf.py extract.image` extraction | Non-arXiv papers — extract embedded images from PDF |
| 4 | TikZ redraw | Only for simple geometric diagrams (flowcharts, arrow diagrams) when no original exists |
| 5 (Last resort) | PDF page screenshot + precise crop | Acceptable ONLY if the crop contains **zero** surrounding text/captions/margins — any visible non-figure content is forbidden |

#### Precise Cropping from PDF

When extracting images from PDFs, you **must crop precisely to the figure boundary only**. Never include surrounding text, captions, or page margins in the extracted image — this looks unprofessional and is the most common image quality mistake.

```bash
# Correct: extract embedded image objects (no surrounding text)
python3 scripts/pdf.py extract.image paper.pdf -o ./images_out/

# Wrong: screenshot full page then crop — almost always includes extra content
# ❌ Do not do this
```

If an image cannot be cleanly extracted (e.g., it spans multiple pages or is overlaid with text), you may use a PDF page screenshot with `\includegraphics` `trim` and `clip` to crop precisely — but the result **must not include any surrounding text, captions, or page margins**:

```latex
% trim = {left bottom right top} — crop until ONLY the figure remains
\includegraphics[trim=20pt 40pt 20pt 30pt, clip,
  width=0.85\textwidth, keepaspectratio]{page_screenshot.png}
```

**Post-crop verification (MANDATORY):** After cropping, visually inspect the result image. If it contains any surrounding text, captions, page margins, or is missing part of the figure, **adjust the trim values and re-crop** until the image is both complete and clean. Repeat this verify→adjust loop until the crop passes inspection. Prefer extracting the actual image over trimming screenshots whenever possible.

#### Image File Organization

```bash
project/
├── main.tex
└── figures/           # All images in a dedicated directory
    ├── fig1.png
    ├── fig2.jpg
    └── fig3.pdf       # Vector format preferred when available
```

Use `\graphicspath{{figures/}}` in the preamble so `\includegraphics` only needs the filename.

#### Vector vs Raster

| Format | When to use |
|---|---|
| `.pdf` / `.eps` (vector) | Line art, flowcharts, plots — preferred, scales without loss |
| `.png` (raster) | Photos, screenshots, experimental results — ensure resolution ≥ 150dpi |
| `.jpg` (raster) | Photographic images — smaller file size |
| ❌ `.svg` | Not directly supported by Beamer — convert to PDF first |

Prefer vector formats (PDF/EPS). For experimental photos and other raster images, ensure sufficient resolution.

#### Subfigures

For papers with composite figures (a/b/c/d panels), either extract individual panels separately, or show the full composite figure with a clear reference:

```latex
% Option 1: Show full composite figure
\begin{center}
  \includegraphics[width=0.75\textwidth, keepaspectratio]{fig1_composite.png}
  \par\vspace{0.15em}
  {\scriptsize \textbf{Figure 1.} (a) Architecture overview (b) Attention heatmap}
\end{center}

% Option 2: Use minipage for side-by-side subfigures
\begin{center}
  \begin{minipage}{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth, keepaspectratio]{fig1a.png}
    \par\vspace{0.1em}
    {\scriptsize (a) Training curve}
  \end{minipage}
  \hfill
  \begin{minipage}{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth, keepaspectratio]{fig1b.png}
    \par\vspace{0.1em}
    {\scriptsize (b) Test accuracy}
  \end{minipage}
\end{center}
```

#### Image-Text Layout Templates

**Left image + right text (most common):**

Use `[c]` (vertical center) for the columns environment when one side is an image — the image visually centers against the opposite text/block column, which looks more balanced than `[T]`. (See "Columns Layout Best Practices" below for the full alignment decision table.)

```latex
\begin{columns}[c]
  \column{0.50\textwidth}
  \begin{center}
    \includegraphics[height=0.45\textheight, keepaspectratio]{fig1.png}
    \par\vspace{0.15em}
    {\scriptsize \textbf{Figure 1.} Description}
  \end{center}
  \column{0.46\textwidth}
  \begin{ablock}{Key Findings}
    \begin{itemize}
      \item Finding one
      \item Finding two
    \end{itemize}
  \end{ablock}
\end{columns}
```

Note: Use `[T]` for **text-text** or **block-block** side-by-side layouts (see "Columns Layout Best Practices"). Use `[c]` for **image-text** layouts.

#### Image Column Vertical Centering (MANDATORY)

When one column contains an image (with optional caption) and the other contains text blocks, the image column's content often appears **top-heavy** — the image sits at the top of the column with empty space below, while the right column's blocks are vertically distributed. This creates a visual imbalance.

**The fix:** Use `[c]` on `\begin{columns}` so both columns are vertically centered against each other. If `[c]` alone is not enough (e.g., the image+caption combo is much shorter than the block column), add vertical spacing above the image to nudge it toward the visual center:

```latex
% ✅ Correct: [c] alignment centers both columns
\begin{columns}[c]
  \column{0.50\textwidth}
  \begin{center}
    \includegraphics[height=0.45\textheight, keepaspectratio]{fig.png}
    \par\vspace{0.15em}
    {\scriptsize \textbf{Figure 1.} Description}
  \end{center}
  \column{0.46\textwidth}
  \begin{ablock}{Block A}
    Content...
  \end{ablock}
  \begin{aalertblock}{Block B}
    Content...
  \end{aalertblock}
\end{columns}
```

**When `[c]` is still not enough** — if the image column is significantly shorter than the block column, the image will be centered but still look "high" relative to the blocks. In this case, use `[T]` with explicit `\vspace` to manually push the image down:

```latex
% Manual adjustment when image column is much shorter
\begin{columns}[T]
  \column{0.50\textwidth}
  \vspace{0.8cm}  % Push image down to align visual center with right column
  \begin{center}
    \includegraphics[height=0.40\textheight, keepaspectratio]{fig.png}
    \par\vspace{0.15em}
    {\scriptsize \textbf{Figure 1.} Description}
  \end{center}
  \column{0.46\textwidth}
  \begin{ablock}{Block A} ... \end{ablock}
  \begin{aalertblock}{Block B} ... \end{aalertblock}
\end{columns}
```

**Visual check:** After compilation, verify that the visual center of the image column (midpoint of image+caption) roughly aligns with the visual center of the block column (midpoint of all blocks+gaps). Adjust `\vspace` if needed.

#### Table-Block Width Alignment

When a table and block(s) are vertically stacked in the same column and serve as parallel semantic units (e.g., "data table" above "conclusion block"), they should have **equal width**. The key is to ensure both elements share the same width — there are multiple ways to achieve this:

**Method 1: Wrap table in an ablock** (when a title makes sense for the table):
```latex
\column{0.48\textwidth}
\begin{ablock}{Experimental Data}
  {\footnotesize
  \begin{tabularx}{\linewidth}{lCX}
    \toprule
    ... \\
    \bottomrule
  \end{tabularx}
  }
\end{ablock}
\begin{ablock}{Conclusion}
  Analysis text...
\end{ablock}
```

**Method 2: Use tabularx with \linewidth without block** (when table doesn't need a title):
```latex
\column{0.48\textwidth}
{\footnotesize
\begin{tabularx}{\linewidth}{lCX}
  \toprule
  ... \\
  \bottomrule
\end{tabularx}
}
\vspace{0.3em}
\begin{ablock}{Conclusion}
  Analysis text...
\end{ablock}
```

**Method 3: Use \resizebox{\linewidth}{!}{...}** (for fixed-width tables):
```latex
\column{0.48\textwidth}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcc}
  \toprule
  ... \\
  \bottomrule
\end{tabular}
}
\vspace{0.3em}
\begin{ablock}{Conclusion}
  Analysis text...
\end{ablock}
```

Choose the method based on context: Method 1 when the table benefits from a title, Method 2 for clean untitled tables, Method 3 when the table already has a fixed column layout.

When NOT to equal-width align:
- Full-page centered standalone tables — use `\centering` with natural width
- Narrow tables (2-3 columns) — forcing full width looks sparse and hurts readability
- Tables with only a footnote-style caption below — different semantic levels, different widths is fine

**Top image + bottom text:**
```latex
\begin{center}
  \includegraphics[height=0.40\textheight, keepaspectratio]{fig1.png}
\end{center}
\vspace{-0.3em}
\begin{ablock}{Analysis}
  Interpretation of the results...
\end{ablock}
```

**When to redraw vs when to use original:**

| Scenario | Action |
|---|---|
| Paper has clear, high-quality original figures | Use the original directly |
| Original figure is low-resolution or blurry | Try arXiv HTML version first, otherwise redraw a simplified version |
| Need to highlight a specific region of a figure | Use original + TikZ overlay annotations (arrows, boxes) |
| Concept/schematic diagram with no original available | Draw a simplified version in TikZ |
| Complex biological/chemical structure diagrams | Always use the original — do not attempt to redraw |

**Full TikZ rules → see Section 8.**

### 7. Navigation and Footer

Navigation symbol removal is handled automatically by the chosen footer scheme. Do **not** manually add `\setbeamertemplate{navigation symbols}{}` when using `progress-navbar.tex` or `paper-navbar.tex` — they include it internally. Only add it manually when using the Simple Footline.

**Footer scheme selection:**

| Scenario | Scheme |
|---|---|
| User provides a paper/PDF to make slides | **Paper Presentation Navigation** (default for paper-based) |
| General academic presentation / courseware (no specific paper) | **Progress Navigation Bar** |
| User requests minimal footer | **Simple Footline** |

**This is not optional.** When the scenario matches a row above, the corresponding scheme **must** be used. Do not fall back to the default Madrid footline or omit the navigation bar.

#### Paper Presentation Navigation (Default for paper-based slides — Recommended)

When the user provides a paper/PDF and asks for slides, use the paper navigation layout. Add to the preamble (after `\usetheme` and `\definecolor{myblue}`):

```latex
\useoutertheme{miniframes}
\input{paper-navbar.tex}
```

Copy `references/paper-navbar.tex` into your project directory before compiling. This file provides:

**Top navigation bar:**
- Beamer built-in `miniframes` — section names + small dots (one dot per slide)
- Current section auto-highlighted
- Subsection empty row removed (clean single-row layout)
- Background: `myblue!15` — adapts to any Preset Color Palette

**Bottom footer — four columns:**

```
Author (Affiliation) | Paper Title | Journal, Year | Page/Total
      25%                  42%            22%           11%
```

- Author, journal, page columns: `myblue!15` (light tint of theme color)
- Paper title column: `myblue!30` (mid tint), white text — visually distinct
- All colors follow `myblue`, so they automatically adapt when switching Preset Color Palettes (e.g., Teal Academic → light teal, Slate Purple → light purple, Warm Earth → light brown)
- Built-in Beamer hyperlinks preserved (author/title link to first/last page)

**Paper metadata setup** — use Beamer's short-form mechanism:

```latex
\title[Short Paper Title]{Full Paper Title}
\author[Author (Affiliation)]{Full Author List}
\date[Journal Name, Year]{}    % No journal → \date[]{}
```

**Note:** `paper-navbar.tex` includes `\setbeamertemplate{navigation symbols}{}` and both `headline`/`footline` templates — do not set them separately.

For author names that are too long for the title page or footer, use `\scriptsize` font size and abbreviated format (e.g., "Guo C., Zhang K., ...") with the short form in square brackets for the footer: `\author[Guo C. et al.]{...}`.

#### Progress Navigation Bar (Alternative — for non-paper academic presentations)

For general academic presentations, courseware, or lectures without a specific source paper, use the progress navigation bar. Add to the preamble (after `\usetheme`):

```latex
\input{progress-navbar.tex}
```

Copy `references/progress-navbar.tex` into your project directory before compiling. This file provides:

- **Dynamic equal-width boxes** — each box width = (paperwidth − 50pt) ÷ total frames; automatically fills the footer regardless of slide count
- **Three-level symbols** — `≡` home (title page), `1 2 3...` section numbers, `◆` subsection, `-` regular slides
- **Progress coloring** — visited pages get `teal!60` background fill; unvisited pages are hollow
- **Clickable hyperlinks** — every box links to its corresponding frame
- **Auto section/subsection title pages** — `\AtBeginSection` and `\AtBeginSubsection` are included
- **Requires `--runs 2`** — first pass writes total frame count, second pass calculates correct widths

**Color customization:** To match your theme, find-and-replace `teal!60` → `myblue!60` (or any color) in the `.tex` file. Also change `\color{teal}\hrule` for the separator line.

**Note:** This navbar replaces `\setbeamertemplate{navigation symbols}{}` and `\setbeamertemplate{footline}` — do not set them separately when using it.

#### Simple Footline (Minimal alternative)

If you prefer a minimal footer without any navigation bar:

```latex
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{%
  \hfill\insertframenumber/\inserttotalframenumber\hspace{1em}\vspace{0.5em}
}
```

The default footline should include at least frame numbers.

### 8. TikZ Usage Guidelines

**TikZ is the last resort, not the first choice.** Always prefer original figures from the paper (extract via `pdf.py extract.image` or arXiv HTML). Only draw TikZ when no original figure exists, and only for **simple** diagrams:
- Flowcharts, pipelines, comparison arrows, simple block diagrams
- Geometric and structured layouts
- You can verify it compiles correctly

**Avoid complex TikZ** — biological structures, network topologies with many nodes, regulatory pathways, full model architectures. If the original figure is not available and the diagram is too complex to simplify, describe it in text/bullet points instead.

**Simplify for slides (CRITICAL):** A slide is not a textbook. TikZ diagrams on slides must be **simplified abstractions**, not full architecture reproductions. If a diagram has too many nodes, small text, or dense connections, the audience cannot read it during a presentation.

| Scenario | Wrong approach | Right approach |
|---|---|---|
| Transformer architecture | Draw every Q/K/V split, softmax, residual connection, layer norm | Simplified block diagram: Input → Encoder (×N) → Decoder (×N) → Output, with key components labeled |
| Multi-head attention | Draw all matrix operations and individual head computations | Abstract flow: Input → Linear projections → Parallel heads → Concat → Output |
| Complex neural network | Reproduce every layer and skip connection | High-level block diagram with labeled stages |

**Rules:**
1. **Original figure available → use it. No TikZ.**
2. **No original figure + simple diagram → draw with TikZ.** Maximum ~8-10 nodes per slide.
3. **No original figure + complex diagram → do NOT draw with TikZ.** Use text/bullet description instead, or split into multiple simplified diagrams.
4. **All text in TikZ must be readable at presentation distance** — minimum `\small` font size, preferably `\normalsize`.

### 9. Compile

```bash
python3 scripts/pdf.py convert.latex main.tex
python3 scripts/pdf.py convert.latex main.tex --runs 2  # for ToC / refs
```

**Always compile and confirm clean output.** Never deliver only a `.tex` file.

#### Post-compilation Verification Checklist (MANDATORY)

After compilation, verify the following before delivery:

1. **No overflow warnings** — check for `Overfull \vbox` / `Overfull \hbox` (see Section 3.5)
2. **Figure/Table numbering is sequential** — scan the PDF and confirm all figures are numbered 1, 2, 3, ... and all tables are numbered 1, 2, 3, ... with no gaps, no duplicates, and no out-of-order numbers. If any numbering is wrong, fix in the `.tex` and recompile.
3. **No "see original" references** — search the `.tex` for any placeholder text that defers to the source document. If found, replace with actual content.
4. **Slide titles are conclusion-driven** — read every `\frametitle{}` and verify it states a finding/conclusion, not a description. Fix any descriptive titles.
5. **Visual inspection** — open the PDF and check every slide for layout issues (see Section 3.5)

### 10. WPS / PDF Viewer Compatibility

Beamer generates navigation hyperlinks that interfere with WPS and some PDF viewers. Always include in the preamble:

```latex
\hypersetup{hidelinks}
```

Note: `\setbeamertemplate{navigation symbols}{}` is already handled by the chosen footer scheme (see Section 7) — do not add it separately here.

### 11. Deliver

Send compiled PDF to user. On Feishu use:

```bash
openclaw message send --channel feishu --target "<chat_id>" \
  --media "main.pdf" --message "Beamer PDF"
```

---

## Output Language

Match the user's query language. If the user writes in Chinese, produce Chinese slides and add `\usepackage[fontset=fandol]{ctex}` to the preamble.

### Slide Title / Heading Language Consistency (MANDATORY)

All slide titles, section headings, and block titles must use a **single language** — the same language as the slide body content. Do not mix languages in headings.

| Slide language | Correct | Wrong |
|---|---|---|
| Chinese | Single-language Chinese heading | Mixed heading (e.g., adding "Outline" after the Chinese word for TOC) |
| English | Single-language English heading | Mixed heading (e.g., "Outline" followed by a Chinese translation) |

Examples: a Chinese deck should use purely Chinese headings without appending English translations; an English deck should use purely English headings without appending Chinese translations.

**The only exception** is established technical abbreviations that have no standard translation (e.g., "RCS", "VIGS", "PEG6000") — these may appear alongside their native-language explanation in body text, but headings should still be pure single-language.

## Source Material Fidelity (MANDATORY)

When the user provides a reference paper, thesis, or any source document:

1. **Never change the language of source content.** If the paper title is in English, keep it in English on the title slide. If the authors wrote their names in a specific script (Chinese characters, Latin letters, etc.), preserve that script. Do not translate titles, author names, institution names, or reference entries into the slide language.

2. **Author name abbreviation is the ONLY permitted modification.** You may shorten author lists (e.g., "Guo C., Zhang K., Li M., ..." → "Guo C. et al.") for space constraints on the title page or footer. No other changes to author names are allowed — do not transliterate, translate, or reorder them.

   **Author name format:** Use `Surname Initial.` with comma separation. Always include spaces between names:
   ```latex
   \author[Guo C. et al.]{Guo C., Zhang K., Sun H., Zhu L., Zhang Y., Wang G., Li A., Bai Z., Liu L., Li C.}
   ```
   Never concatenate names without spaces (e.g., ❌ `CongcongGuo,KeZhang`).

3. **Reference entries must preserve original language.** If a cited paper has a Chinese title, keep the Chinese title in the reference list. If it has an English title, keep English. Do not translate references to match the slide language.

4. **Title page layout for bilingual scenarios.** When the slide language differs from the paper language (e.g., Chinese slides for an English paper), keep the original paper title in `\title{}` and put the translation in `\subtitle{}`. Never mix two languages at the same level:
   ```latex
   % ✅ Correct: English title + Chinese subtitle
   \title{Root Cortical Senescence Enhances Drought Tolerance in Cotton}
   \subtitle{根系皮层衰老增强棉花耐旱性}
   
   % ❌ Wrong: two languages jammed into \title
   \title{Root Cortical Senescence Enhances Drought Tolerance in Cotton 根系皮层衰老增强棉花耐旱性}
   ```

| Allowed | Forbidden |
|---|---|
| `Guo C. et al.` (abbreviated from full list) | Translating or transliterating author names between scripts/languages |
| Keeping English paper title on Chinese slides | Translating paper title to match slide language |
| Using `\scriptsize` for long author lists | Omitting authors entirely |
| Abbreviating institution names | Translating institution names |
| Original title in `\title{}` + translation in `\subtitle{}` | Mixing two languages in the same `\title{}` |

## Compilation Rules

1. `[fragile]` mandatory for `verbatim` / `lstlisting` frames
2. `\mathbb` takes ONE character — use `\mathrm{KL}` not `\mathbb{KL}`
3. Always brace subscripts — `_{\max}` not `_\max`
4. No Unicode symbols (★✓→) — use `$\star$`, `\checkmark`, `$\rightarrow$`
5. `nullfont` warnings with ctex+Madrid are cosmetic — ignore
6. Always `--runs 2` when using `\tableofcontents` or `\ref`

---

## Typography and Visual Enhancement (NEW)

### Text Alignment

Use justified alignment as the default for body text in Beamer. It produces a cleaner, more professional look compared to left-aligned (ragged right) text, especially for paragraphs with mixed CJK and Latin content.

```latex
% In preamble — set global justified alignment
\usepackage{ragged2e}
\justifying
```

Note: Lists (`itemize`/`enumerate`) remain left-aligned (do not force justified). Body paragraphs can have indentation, but must be justified.

### Font and Spacing

#### Font Size Hierarchy

Establish clear visual hierarchy with distinct size levels:

```latex
\setbeamerfont{title}{size=\LARGE, series=\bfseries}
\setbeamerfont{subtitle}{size=\large}
\setbeamerfont{frametitle}{size=\large, series=\bfseries}
\setbeamerfont{framesubtitle}{size=\normalsize}
\setbeamerfont{block title}{size=\normalsize, series=\bfseries}
\setbeamerfont{footnote}{size=\tiny}
```

**Document base size:** Choose an appropriate base size (`10pt`, `11pt`, or `12pt`) based on content density and audience distance. There is no fixed default — pick what fits the presentation best.
```latex
\documentclass[aspectratio=169, 11pt]{beamer}   % Adjust pt as needed
```

For content-heavy individual slides, use local font size commands:
- Content-heavy slides: use `\small` or `\footnotesize` locally for that slide
- Content-light slides: keep the chosen base size for comfortable reading
- **Overflow warnings are critical**: if compilation produces `Overfull \vbox` or `Overfull \hbox` warnings, the content does not fit — reduce font size, reduce content, or split the slide. Never ignore overflow warnings.

```latex
% Per-slide adjustment for dense content:
{\small
\begin{tabular}{...}
  ...
\end{tabular}
}
```

#### Math Font Protection

**Always add** for presentations with any math formulas:
```latex
\usefonttheme{professionalfonts}
```
Without this, Beamer overrides math fonts causing distorted formulas.

#### Serif vs Sans-serif

| Font Theme | Effect | Best For |
|---|---|---|
| `default` (sans-serif) | Modern, clean | Most scenarios, tech/engineering |
| `serif` | More academic feel | Math-heavy, humanities papers |
| `structurebold` | Bold structural elements | Clear hierarchy emphasis |

```latex
\usefonttheme{default}            % Sans-serif (default)
\usefonttheme{serif}              % Serif body text
\usefonttheme{structurebold}      % Bold titles/headers
```

#### Chinese Font Setup

```latex
\usepackage[fontset=fandol]{ctex}  % Portable, bundled fonts
% macOS with system fonts installed:
% \usepackage[fontset=mac]{ctex}
```

ctex fontsets provide sensible defaults (Song for body, Hei for headings). Only use `\setCJKmainfont` manually if you need a specific font not covered by fontset.

#### Bold Usage Restraint

Use bold **only** for: titles, best values in tables, key terms on first appearance. Never bold entire paragraphs — overuse negates emphasis.

### Consistent List Styles

```latex
\setbeamertemplate{itemize items}[circle]           % Primary level (use [circle], not \textbullet)
\setbeamertemplate{itemize subitem}{--}              % Secondary level
\setbeamercolor{itemize item}{fg=myblue}             % Match theme color
```

> **Note:** `references/beamer.md` Section 3.7 shows `\textbullet` as a generic customization example. For actual slide generation, always use `[circle]` as defined above.

### Color Palette Guidelines

- **Color style must be consistent**: use one color scheme throughout the entire presentation — never switch color styles or mix different palettes mid-deck
- Limit to **2-3 primary colors** (e.g., `myblue` for structure, `myred` for alerts, `mygreen` for examples)
- **All three must be defined** in the preamble via `\definecolor` — the recommended defaults are `myblue=#2B5EA7`, `myred=#C0392B`, `mygreen=#27AE60` (see Section 3 tcolorbox definitions). Adjust hex values to match the presentation's theme, but always define all three.
- **Do not overuse colors**: do not use different primary hues on different slides just for aesthetics. Alert color should only be used when emphasis or contrast is genuinely needed
- Use low-saturation tints for backgrounds (`myblue!8`) and full-saturation for frames/titles (`myblue!85`)
- Define all custom colors in the preamble with `\definecolor` using HTML hex values — do not introduce ad-hoc colors mid-document

#### Preset Color Palettes

When the user does not specify colors, pick a palette that fits the subject area or presentation tone. The **three-color semantic roles are unchanged** (blue = structural, red = warning/negative, green = positive/example) — only the hex values change.

| # | Palette Name | myblue | myred | mygreen | Best For |
|---|---|---|---|---|---|
| 1 | **Classic Blue** (default) | `#2B5EA7` | `#C0392B` | `#27AE60` | General academic, engineering, CS |
| 2 | **Deep Ocean** | `#1A3C6E` | `#B83230` | `#1E8C5A` | Formal conferences, physics, math |
| 3 | **Teal Academic** | `#0E7C7B` | `#D35233` | `#2A9D6F` | Biology, ecology, environmental science |
| 4 | **Slate Purple** | `#4A3F8A` | `#C44536` | `#3A8F6E` | Humanities, social science, psychology |
| 5 | **Warm Earth** | `#6B4C3B` | `#C75C2E` | `#5B8C3E` | Agriculture, geology, archaeology |
| 6 | **Steel Gray** | `#3D4F5F` | `#C0392B` | `#2E8B6E` | Corporate, business, economics |
| 7 | **Burgundy** | `#7B2D4E` | `#C0392B` | `#2A7F62` | Medicine, clinical, health science |
| 8 | **Midnight** | `#1B2A4A` | `#E74C3C` | `#16A085` | Tech keynote, AI/ML, astronomy |

```latex
% Example: Teal Academic palette
\definecolor{myblue}{HTML}{0E7C7B}
\definecolor{myred}{HTML}{D35233}
\definecolor{mygreen}{HTML}{2A9D6F}
```

**Selection heuristic:** If the user's topic or field clearly maps to a palette above, use it. If ambiguous, default to **Classic Blue (#1)**. If the user explicitly provides hex values or says "use red/purple/etc.", define custom colors accordingly — these presets are suggestions, not constraints.

#### Block Color Semantic Rules (MANDATORY)

Each color carries a fixed semantic meaning. **Never assign colors based on aesthetics or visual variety alone — always match color to content meaning.**

| Color | Semantic meaning | When to use |
|---|---|---|
| `myblue` / `fblock` / `ablock` | **Neutral / structural / primary** | Default for most blocks — descriptions, methods, explanations, neutral content |
| `myred` / `falertblock` / `aalertblock` | **Negative / warning / problem** | ONLY for: problems, limitations, risks, caveats, things that went wrong |
| `mygreen` / `fexampleblock` / `aexampleblock` | **Positive / example / solution** | For: solutions, advantages, good results, examples, positive outcomes |

**Hard rules:**

1. **Red (`myred`) is EXCLUSIVELY for negative semantics.** Never use red/alertblock for neutral content like "Physiological Indicators", "Evaluation Setup", "Method Overview". If the content is not a problem/warning/limitation, it must NOT be red.

2. **Parallel blocks of the same level must use the same color.** When multiple blocks on a slide are listing items of the same category (e.g., four research methods, three evaluation metrics), they are semantically equal and must ALL use the same block color — typically `myblue`.

3. **Categorical distinction uses blue + green, never red.** If you want to visually distinguish two categories on the same slide (e.g., left column = observational methods, right column = molecular methods), use `myblue` for one category and `mygreen` for the other. Red is reserved for genuinely negative content only.

```latex
% ✅ Correct: four parallel method blocks, all same color
\begin{columns}[T]
  \column{0.48\textwidth}
  \begin{fblock}[2.5cm]{Morphological Observation}
    Content...
  \end{fblock}
  \vspace{0.2cm}
  \begin{fblock}[2.5cm]{Microscopic Analysis}
    Content...
  \end{fblock}
  \column{0.48\textwidth}
  \begin{fblock}[2.5cm]{Biochemical Assays}
    Content...
  \end{fblock}
  \vspace{0.2cm}
  \begin{fblock}[2.5cm]{Gene Silencing (VIGS)}
    Content...
  \end{fblock}
\end{columns}

% ✅ Also correct: categorical distinction with blue + green
\begin{columns}[T]
  \column{0.48\textwidth}
  % Observation category → blue
  \begin{fblock}[2.5cm]{Morphological Observation}...
  \end{fblock}
  \vspace{0.2cm}
  \begin{fblock}[2.5cm]{Microscopic Analysis}...
  \end{fblock}
  \column{0.48\textwidth}
  % Molecular category → green
  \begin{fexampleblock}[2.5cm]{Biochemical Assays}...
  \end{fexampleblock}
  \vspace{0.2cm}
  \begin{fexampleblock}[2.5cm]{Gene Silencing (VIGS)}...
  \end{fexampleblock}
\end{columns}

% ❌ Wrong: red used for neutral content
\begin{falertblock}[2.5cm]{Biochemical Assays}  % NOT a warning/problem!
  Content...
\end{falertblock}
```

### Comparison Slide Visual Polish (MANDATORY)

When a slide presents a **comparison** (e.g., "before vs after", "problem vs solution", "method A vs method B"), apply these rules:

#### 1. Semantic Color Pairing

Use color to reinforce meaning. If one side is the "problem" and the other is the "solution", their block colors must reflect this:

| Semantic role | Block type | Color |
|---|---|---|
| Problem / limitation / before | `falertblock` | `myred` family |
| Solution / advantage / after | `fexampleblock` | `mygreen` family |
| Neutral / description | `fblock` | `myblue` family |

**Symmetry rule:** If one side uses `\alert{}` (red) to highlight a negative keyword (e.g., "Breaks layout"), the other side SHOULD use `\textcolor{mygreen}{}` to highlight the corresponding positive keyword (e.g., "Eliminates overhead"). Left-red-right-green pairing creates instant visual comprehension.

```latex
% Example: problem vs solution comparison
\begin{columns}[T]
  \column{0.48\textwidth}
  \begin{falertblock}[4.5cm]{Eager Broadcasting}
    \begin{itemize}
      \item Immediate materialization
      \item \alert{Breaks sparsity layout}  % red highlight for problem
    \end{itemize}
  \end{falertblock}
  \column{0.48\textwidth}
  \begin{fexampleblock}[4.5cm]{Lazy Broadcasting}
    \begin{itemize}
      \item Deferred evaluation
      \item \textcolor{mygreen}{\textbf{Eliminates overhead}}  % green highlight for solution
    \end{itemize}
  \end{fexampleblock}
\end{columns}
```

#### 2. Standalone Tables Must Be Wrapped

Tables that appear alongside tcolorbox blocks on the same slide should be wrapped in a tcolorbox block for visual consistency. A bare `tabular` environment next to styled blocks looks unfinished.

| Scenario | Treatment |
|---|---|
| Table relates to a specific topic | Wrap in `ablock{Topic Title}` |
| Table is a comparison/summary | Wrap in `ablock{Comparison}` or `aexampleblock{Summary}` |
| Table is the only element on the slide | May remain unwrapped with `\centering` |

```latex
% ✅ Correct: table wrapped in block, consistent with other blocks on slide
\begin{ablock}{Optimization Level Comparison}
  {\footnotesize
  \begin{tabularx}{\linewidth}{lCCC}
    \toprule
    Level & Broadcast & Overhead & Performance \\
    \midrule
    Eager & Yes & \textcolor{myred}{High} & Baseline \\
    Lazy  & No  & \textcolor{mygreen}{\textbf{None}} & 2.1$\times$ \\
    \bottomrule
  \end{tabularx}
  }
\end{ablock}

% ❌ Wrong: bare table next to styled blocks
\begin{tabular}{lcc}
  ...
\end{tabular}
```

#### 3. Table Cell Color Coding

When table cells represent qualitative values (good/bad, yes/no, high/low), use color to encode meaning instead of relying on text alone:

| Value type | Color treatment |
|---|---|
| Positive / good / improved | `\textcolor{mygreen}{\textbf{value}}` |
| Negative / bad / degraded | `\textcolor{myred}{value}` |
| Neutral / baseline | Default text color |
| Best in column/row | `\textcolor{mygreen}{\textbf{value}}` (bold + green) |

This makes tables scannable at a glance — the audience can see the pattern without reading every cell.

#### 4. Block Title Bar Saturation Consistency

All block title bars on the same slide should have comparable visual weight. If one block uses `myblue!85` for the title bar, don't pair it with a `mygreen!100` title bar — the green will appear much heavier. Keep title bar saturation levels consistent:

```latex
% ✅ Consistent: both use !85 saturation
colframe=myblue!85   % blue block
colframe=mygreen!80  % green block (slightly lower to compensate green's higher perceived brightness)

% ❌ Inconsistent: one muted, one vivid
colframe=myblue!60   % muted blue
colframe=mygreen!100 % vivid green — too dominant
```

#### 5. List Style Consistency Within a Slide (MANDATORY)

All list environments on the same slide must use the **same bullet/numbering style**. Do not mix `itemize` (bullets) with `enumerate` (numbers) or custom circled numbers unless the semantic distinction is clear and intentional.

| Scenario | Correct approach |
|---|---|
| All lists are unordered collections | Use `itemize` with consistent bullet style everywhere |
| All lists are ordered steps/sequences | Use `enumerate` everywhere |
| One list is steps, another is features | OK to differ — but document the semantic reason |
| One block uses ❶❷❸, another uses • bullets | ❌ Inconsistent — pick one style |

**Circled numbers / custom list markers:**
- If using circled numbers (❶❷❸❹), do NOT use raw Unicode characters — they may fail in some compilers
- Use LaTeX-safe alternatives:

```latex
% Option 1: pifont (recommended)
\usepackage{pifont}
\newcommand{\cmark}[1]{\ding{\numexpr201+#1\relax}}  % \cmark{1} → ❶
% Usage: \item[\cmark{1}] First item

% Option 2: tikz inline circles
\newcommand{\circnum}[1]{%
  \tikz[baseline=(char.base)]\node[circle, fill=myblue, text=white,
  inner sep=1.2pt, font=\scriptsize\bfseries] (char) {#1};}
% Usage: \item[\circnum{1}] First item
```

#### 6. Line Break Prevention for Technical Terms (MANDATORY)

Technical terms, version numbers, short parenthetical units, and numeric expressions must not break across lines. Bad line breaks (e.g., "202" on one line and "LoC)" on the next) look unprofessional.

**Prevention techniques:**

| Technique | When to use | Example |
|---|---|---|
| Non-breaking space `~` | Between number and unit | `202~LoC`, `3.5~GHz` |
| `\mbox{...}` | Short phrase that must stay together | `\mbox{PyTorch 2.1}` |
| `\hbox{...}` | Same as mbox, TeX primitive | `\hbox{CUDA 12.0}` |
| `white-space: nowrap` equivalent: put in `\mbox{}` | Version strings, short labels | `\mbox{v2.0-beta}` |

```latex
% ❌ Bad: "202" and "LoC)" split across lines
PyTorch BSR sparse (202
LoC)

% ✅ Good: kept together
PyTorch BSR sparse (\mbox{202 LoC})
% or
PyTorch BSR sparse (202~LoC)
```

**Apply this proactively** — scan all slides for short trailing fragments (1-2 words or a number+unit orphaned on a new line) and fix them before delivery.

#### 7. Content Density Balance Across Columns (MANDATORY)

When using a two-column layout, both columns should have comparable **content density** (amount of meaningful content relative to the space). A column with sparse content (e.g., 7 one-line items in a tall block with excessive line spacing) next to a dense column creates visual imbalance.

**Fixes for sparse columns:**

| Strategy | When to use |
|---|---|
| Add brief descriptions to each item | Items are bare labels (e.g., just baseline names) |
| Split into two smaller blocks | Content naturally groups into sub-categories |
| Reduce block height + add a supplementary block | Room for an extra block below (e.g., "Evaluation Metrics", "Notes") |
| Adjust column width ratio | Give the denser column more space |

```latex
% ❌ Sparse: 7 bare items in a tall block, lots of wasted space
\begin{fblock}[5.5cm]{Baselines}
  \begin{itemize}
    \item cuSPARSE
    \item Triton Block-Sparse
    \item TorchBSR
    ...
  \end{itemize}
\end{fblock}

% ✅ Better: split into two themed blocks, balanced density
\begin{fblock}[2.5cm]{Vendor Libraries}
  \begin{itemize}
    \item cuSPARSE — NVIDIA official sparse BLAS
    \item MKL Sparse — Intel CPU baseline
  \end{itemize}
\end{fblock}
\vspace{0.2cm}
\begin{fblock}[2.8cm]{Research Implementations}
  \begin{itemize}
    \item Triton Block-Sparse — compiler-based
    \item TorchBSR — PyTorch native (\mbox{202 LoC})
    ...
  \end{itemize}
\end{fblock}
```

### icon Support

Use `fontawesome5` to add icons for visual flair in lists and headings:

```latex
\usepackage{fontawesome5}

\begin{itemize}
  \item[\faCheckCircle] Verified result
  \item[\faLightbulb] Key insight
  \item[\faExclamationTriangle] Caveat
\end{itemize}
```

### Overlay Usage (DEFAULT: OFF)

Overlays are **disabled by default** for all Beamer presentations. All slide content should be fully visible without step-by-step reveals. This produces static slides that are easier to read, share as handouts, and navigate in PDF viewers.

**The ONLY exception is the courseware (teaching slides) scenario.** When the user explicitly requests courseware, teaching slides, or lecture materials, overlays are enabled — specifically for **separating problem statements and proofs/solutions** so the instructor can reveal them step by step during class.

#### When overlays are OFF (default — all scenarios except courseware)

Do NOT use any of the following:
- `\item<N->` overlay markers on list items
- `\uncover<N->{...}` or `\only<N->{...}` wrappers
- `\pause` commands
- `\visible<N->{...}` or `\onslide<N->{...}`
- Any other Beamer overlay specification

All content on every slide must be immediately visible. Write plain `\item` without overlay specs. Do not wrap blocks, figures, or equations in `\uncover` or `\only`.

```latex
% ✅ Default (non-courseware): all content static, fully visible
\begin{itemize}
  \item First key point
  \item Second key point
  \item Third key point
\end{itemize}

\begin{ablock}{Background}
  Context information...
\end{ablock}
\begin{aalertblock}{Problem}
  The core challenge...
\end{aalertblock}
```

---

#### When overlays are ON (courseware scenario ONLY)

Activate overlays **only when the user explicitly requests courseware, teaching slides, lecture materials, or classroom presentations.** In this scenario, use overlays specifically to separate:

1. **Problem statements ** — shown first
2. **Proofs / solutions ** — revealed on the next overlay step

This allows the instructor to present the problem, let students think, then reveal the answer.

##### Courseware overlay pattern

```latex
% ✅ Courseware: problem shown first, proof revealed on next click
\begin{frame}{Theorem: Triangle Inequality}
  \begin{block}{Problem}
    Prove that for any real numbers $a$ and $b$: $|a + b| \leq |a| + |b|$.
  \end{block}

  \uncover<2->{%
  \begin{exampleblock}{Proof}
    We know that $-|a| \leq a \leq |a|$ and $-|b| \leq b \leq |b|$.
    Adding these inequalities: $-(|a|+|b|) \leq a+b \leq |a|+|b|$.
    Therefore $|a+b| \leq |a| + |b|$. \qed
  \end{exampleblock}
  }
\end{frame}

% ✅ Courseware: step-by-step solution reveal
\begin{frame}{Example: Solving a Quadratic Equation}
  \begin{block}{Problem}
    Solve $x^2 - 5x + 6 = 0$.
  \end{block}

  \uncover<2->{%
  \begin{ablock}{Solution}
    Factor: $(x-2)(x-3) = 0$ \\
    Therefore $x = 2$ or $x = 3$.
  \end{ablock}
  }
\end{frame}
```

##### What to animate in courseware mode

| Element | Animate? | Method |
|---|---|---|
| **Problem / question statement** | Always visible (overlay `<1->`) | No overlay needed — shown by default |
| **Proof / solution / answer** | Revealed on step 2+ | `\uncover<2->{...}` wrapping the proof block |
| **Hints (optional)** | Revealed between problem and proof | `\uncover<2->{hint}`, `\uncover<3->{proof}` |
| **List items in proofs** | May be revealed step-by-step | `\item<N->` for each proof step |
| **Non-problem slides** (background, outline, summary) | Static — no overlay | No overlay specs |

##### What NOT to animate even in courseware mode

| Element | Reason |
|---|---|
| Slide titles / frame titles | Must be visible from the start |
| Section divider slides | Single-element, nothing to reveal |
| Title / closing slides | No progressive content |
| TOC / outline slides | Overview should be fully visible |
| Background / theory introduction slides | Not problem-solution format, keep static |

##### Consistency rules (MANDATORY — applies when overlays are used)

**Critical rule: consistency within a single list.** Within any single `itemize` or `enumerate` environment, either ALL `\item` specs have overlay markers (e.g. `\item<1->`) or NONE do. Mixing animated and static `\item` specs in the same list looks inconsistent and breaks the visual flow.

**Clarification:** Inline overlay commands like `\alert<2>{text}` or `\textcolor<3>{...}` inside an item's content do NOT count as item-level overlays. It is fine to use `\alert<>` selectively on some items while all `\item` specs themselves remain un-overlayed.

```latex
% ✅ Correct: all items animated (in a proof)
\begin{itemize}
  \item<2-> Step 1: Assume the hypothesis
  \item<3-> Step 2: Apply the theorem
  \item<4-> Step 3: Conclude \qed
\end{itemize}

% ✅ Correct: no items animated (problem statement, always visible)
\begin{itemize}
  \item Given: $\triangle ABC$ with $\angle A = 90°$
  \item Prove: $BC^2 = AB^2 + AC^2$
\end{itemize}

% ❌ Wrong: mixed animation in one list
\begin{itemize}
  \item<1-> First point
  \item Second point        % ← no overlay, inconsistent
  \item<2-> Third point
\end{itemize}
```

##### Nested animation

When a block contains a list, choose ONE level to animate — either animate the block as a whole, or animate the list items inside, but not both simultaneously (double-animation causes confusing timing):

```latex
% ✅ Option A: animate the block, list appears all at once
\uncover<2->{%
\begin{ablock}{Proof}
  \begin{itemize}
    \item Step one
    \item Step two
  \end{itemize}
\end{ablock}
}

% ✅ Option B: block always visible, animate items inside
\begin{ablock}{Solution Steps}
  \begin{itemize}
    \item<2-> Step one
    \item<3-> Step two
  \end{itemize}
\end{ablock}

% ❌ Wrong: double animation (block AND items)
\uncover<2->{%
\begin{ablock}{Proof}
  \begin{itemize}
    \item<3-> Step one   % block appears at 2, item at 3 — confusing
    \item<4-> Step two
  \end{itemize}
\end{ablock}
}
```

### Theme Recommendations

| Style | Recommended Theme | Notes |
|---|---|---|
| Modern / minimal | `metropolis` (install separately) | Clean, built-in progress bar, dark mode support |
| Classic academic | `Madrid` + `\usecolortheme{dolphin}` | Reliable, widely used |
| Structure-heavy | `Berlin` or `Warsaw` | Built-in section navigation |
| Clean / corporate | `CambridgeUS` or `Boadilla` | Simple two-tone |
| Ultra-minimal | `default` + custom `\setbeamercolor` | Maximum flexibility |

For `metropolis`, add to preamble:
```latex
\usetheme{metropolis}
\metroset{progressbar=frametitle}  % progress bar in frame title
```

### Columns Layout Best Practices

```latex
\begin{columns}[T]  % [T] top · [c] center · [b] bottom alignment
```

**Vertical alignment decision table (single source of truth):**

| Layout type | Alignment | Notes |
|---|---|---|
| Block-block / text-text | `[T]` | Top edges align, visually cleanest |
| Image-text | `[c]` | Image centers against text/block column |
| Image-text where `[c]` is insufficient | `[T]` + `\vspace` | Manually push image down to visual center (see Section 6 "Image Column Vertical Centering") |

Do not duplicate these rules elsewhere — all alignment decisions reference this table.

**Column widths:**
- Total of both columns should not exceed `0.96\textwidth` to leave gap between columns
- Give the wider side more space: e.g., image-heavy side gets `0.55\textwidth`, text side gets `0.42\textwidth`
- For equal-width layouts: `0.48\textwidth` each

**Content alignment within columns:**
- Align block titles across columns: use `[T]` + same block type (both `fblock` with identical height)
- Side-by-side blocks must use `fblock`/`falertblock` with the **same height parameter** (see Section 3)
- Center images with `\centering`, keep text/lists left-aligned
- If one column has noticeably less content, either add an `ablock` to balance or adjust column width ratio

---

---

## Compilation Troubleshooting: Missing Fonts in User-Provided Templates

When users provide custom Beamer templates (`.sty` or `.tex`), the template may depend on specific system fonts. The most common compilation failure is `Package fontspec Error: The font "XXX" cannot be found`.

### General Troubleshooting Workflow

**1. Identify the missing font name from the error message**

Extract the font name (e.g., "Kaiti SC") from the error and determine whether it is a system font expected by a ctex fontset, or a font manually specified via `\setCJKmainfont{...}` in the user's template.

**2. Search for the font file on the system**

```bash
# Full system search
find / -iname "*fontname*" 2>/dev/null
# Common font directories — macOS
ls ~/Library/Fonts/ /Library/Fonts/ /System/Library/Fonts/ /System/Library/Fonts/Supplemental/
# macOS may store fonts under AssetsV2
find /System/Library/AssetsV2 -iname "*.ttc" -o -iname "*.ttf" 2>/dev/null
# Common font directory — Linux
ls /usr/share/fonts/truetype/
```

**3. Handle based on search results**

| Situation | Solution |
|---|---|
| Font exists on system but not in standard Fonts directory (macOS) | Copy to `~/Library/Fonts/` so the compiler can find it |
| Font exists on system but not in standard Fonts directory (Linux) | Copy to `/usr/share/fonts/truetype/` then run `fc-cache -fv` |
| Font does not exist on the system (macOS) | Download and install the font file to `~/Library/Fonts/` |
| Font does not exist on the system (Linux) | Download the font file to `/usr/share/fonts/truetype/` then run `fc-cache -fv` |
| Cannot install fonts (server / no permissions) | Switch to `fontset=fandol` (bundled open-source fonts with ctex) or `fontset=none` + manually specify available fonts |

**4. Verify**

Recompile and confirm the error is gone. A noticeably larger PDF file size indicates fonts were successfully embedded.

### Do Not Work Around the Problem

When fonts are missing, **do not** bypass compilation errors by deleting font references (e.g., removing `\kaishu`, `\songti` commands). This causes the output to diverge from the user's expected template style. The correct approach is to install the missing fonts.

### fontset Selection Reference

| Scenario | Recommendation |
|---|---|
| macOS + need native Chinese font consistency | `fontset=mac` (ensure fonts are installed in standard paths) |
| Linux / server / no system fonts | `fontset=fandol` (open-source fonts bundled with ctex) |
| Need specific custom fonts | `fontset=none` + `\setCJKmainfont` / `\setCJKsansfont` manual specification |
| Using a user-provided `.sty` template | Keep the template's original fontset setting — only fix font installation |

### Example: Missing Kaiti SC on macOS

ctex `fontset=mac` requires "Kaiti SC". Newer versions of macOS store this font under the `AssetsV2` directory rather than the standard Fonts path, so tectonic cannot locate it automatically. Solution:

```bash
# Find the font
find /System/Library/AssetsV2 -iname "Kaiti*" 2>/dev/null
# Copy to user font directory
cp /System/Library/AssetsV2/com_apple_MobileAsset_Font7/<hash>/AssetData/Kaiti.ttc ~/Library/Fonts/
```

Similarly, if other macOS system fonts needed by ctex (Songti SC, Heiti SC, etc.) are also missing, use the same method to locate and copy them.

---

## References

- `references/beamer.md` — Theme catalogue, overlay syntax, content templates, TikZ examples
- `references/latex.md` — Non-Beamer LaTeX documents (papers, articles, theses)
- `scripts/pdf.py` — PDF compilation wrapper and extraction tool
