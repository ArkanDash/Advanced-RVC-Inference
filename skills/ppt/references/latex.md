# Route 3: Academic & Scientific PDF via LaTeX

Produce publication-grade PDFs from `.tex` source files compiled with Tectonic. Suited for academic papers, theses, mathematical manuscripts, IEEE/ACM-format submissions, and any document where the user explicitly requests LaTeX.

**Guiding principle**: when someone asks for "a LaTeX PDF," they expect a polished, professional result — not a bare-minimum compilation.

---

## Environment Preparation


The binary of Tectonic lands at `scripts/tectonic`.

### Compilation — Always Use the Wrapper Script

All compilations **must** go through `scripts/pdf.py convert.latex`. It handles:
- Stripping noisy package-download logs
- Filtering progress chatter
- Surfacing genuine errors and warnings
- Reporting PDF statistics (file size, page count, word estimate, image tally)

**Do not invoke Tectonic directly.**

```bash
# One-pass build
python3 scripts/pdf.py convert.latex main.tex

# Two passes (resolves cross-references)
python3 scripts/pdf.py convert.latex main.tex --runs 2

# Verbose mode (retains full log output)
python3 scripts/pdf.py convert.latex main.tex --keep-logs
```

---

## Pre-Writing Planning

Before touching `.tex` code, determine the document's category and let that shape your approach.

### Recognising the Document Type

| Type | Typical outputs | Key concern |
|------|----------------|-------------|
| **Scholarly** | Journal articles, conference proceedings, regulatory specs | Rigorous adherence to academic conventions; bibliography accuracy |
| **Utilitarian** | Technical reports, reference manuals, product specs | Pack maximum information while preserving scannability |
| **Persuasive** | Funding proposals, pitch documents, project roadmaps | Clean professionalism throughout, with one or two visual high-points (title page, KPI dashboards) |
| **Expressive** | Design portfolios, brand guidebooks, showcases | Bold typographic and chromatic choices; deliberate rule-breaking that amplifies impact |

### Fallback Aesthetic (No Style Specified)

When the user gives no visual direction, apply a **measured, high-craft** system:

1. **Contrast** — clear figure-ground separation; headings visually distinct from running text
2. **Hierarchy** — establish reading order through deliberate variation in size, weight, and hue
3. **White space** — ample margins and leading to let the page breathe
4. **Coherence** — one typeface family, one accent colour, one spacing rhythm

#### Enrichment Elements (Add Proactively When Appropriate)

- **Decorative**: shaded callout boxes, sidebars, comparison panels → `tcolorbox`
- **Scholarly**: theorem / definition / proof environments → `amsthm` + `tcolorbox`; process diagrams → TikZ
- **Page furniture**: running headers and footers → `fancyhdr`; chapter openers → `titlesec`

Introduce these components on your own initiative whenever the content benefits — don't wait to be told.

---

## Mandatory Rules

### Rule 1 — Every Build Diagnostic Demands Attention

The wrapper classifies compiler output into three tiers:

| Tier | Impact | Required response |
|------|--------|-------------------|
| **Errors** | Build aborts | Resolve before anything else |
| **Layout defects** | Overfull / underfull boxes, unavailable font shapes, missing glyphs | Repair prior to delivery |
| **Advisories** | Remaining warnings | Assess individually; fix whenever feasible |

**Never acceptable**: shrugging off warnings with "they don't affect the final PDF." Every diagnostic merits investigation.

### Rule 2 — Modular Files for Larger Works

A single generation turn can comfortably handle roughly **500 lines** of TeX.

**When to split**:
- Anticipated output exceeds 5 pages, **or**
- Three or more heavyweight elements are present (wide tables, block equations, TikZ graphics)

Stitch modules together with `\input{chapter1}`, `\input{chapter2}`, etc.

### Rule 3 — Foundation Preamble

```latex
\documentclass{article}

% Essentials
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{geometry}
\usepackage{amsmath}          % Load before hyperref

% hyperref — ALWAYS last among content packages
\usepackage[
    colorlinks=true,
    linkcolor=blue,
    citecolor=darkgray,
    urlcolor=blue,
    bookmarks=true,
    bookmarksnumbered=true,
    unicode=true
]{hyperref}

% Page dimensions
\geometry{a4paper, top=2.5cm, bottom=2.5cm, left=3cm, right=2.5cm}
% CV variant: \geometry{a4paper, margin=1.5cm}

% Superscript citation numbers (scholarly convention)
\usepackage[numbers,super,sort&compress]{natbib}
\bibliographystyle{unsrtnat}

% Commonly needed extras
\usepackage{tcolorbox}
\usepackage{colortbl}
\usepackage{booktabs}
\usepackage{enumitem}
```

**hyperref positioning**: it must load after virtually every other package to avoid option clashes.

### Rule 4 — TeX Source Hygiene

**Prohibited patterns**:
- Emoji glyphs (no native LaTeX engine supports them)
- Markdown `*asterisk*` formatting (generates compile-time errors)

**Use instead**:
```latex
\textbf{bold text}
\emph{emphasised text}
```

These are frequent model-generation slips — catch them proactively.

### Rule 5 — Multi-Language & Font Handling

- `babel` and `polyglossia` are incompatible — load only one
- When using `polyglossia`, ensure `amsmath` appears earlier in the preamble
- Tectonic downloads standard LaTeX typefaces automatically (Latin Modern, etc.)
- To reference system-installed fonts via `\setmainfont{}`, first probe with `fc-list :lang=ar`
- Broadly available choices: DejaVu, Noto typeface families

### Rule 6 — Clickable Navigation Is Non-Negotiable

Interactive navigation is a baseline professional expectation.

#### 6.1 Table of Contents
```latex
\tableofcontents   % Entries are auto-linked courtesy of hyperref
\listoffigures
\listoftables
```

#### 6.2 Internal Cross-References

**Attach labels** right after each numbered element:
```latex
\section{Background}\label{sec:bg}

\begin{figure}[htbp]
    \includegraphics{...}
    \caption{Overview diagram}\label{fig:overview}
\end{figure}

\begin{table}[htbp]
    \caption{Benchmark results}\label{tab:bench}
    ...
\end{table}

\begin{equation}\label{eq:energy}
    E = mc^2
\end{equation}
```

**Cite these labels** (each produces a live hyperlink):
```latex
As noted in Section~\ref{sec:bg}...
Figure~\ref{fig:overview} illustrates...
Table~\ref{tab:bench} summarises...
Equation~\eqref{eq:energy} yields...   % \eqref auto-wraps in parentheses
See page~\pageref{sec:bg}...
```

**Good habits**:
- Place a non-breaking space `~` before `\ref` to avoid orphaned line breaks
- Prefer `\eqref{}` for equations (auto-wraps the number in parentheses)
- Adopt a consistent label-prefix convention: `sec:`, `fig:`, `tab:`, `eq:`, `lst:`

#### 6.3 Bibliography

**Superscript numerals (preferred academic style)**:
```latex
\usepackage[numbers,super,sort&compress]{natbib}
\bibliographystyle{unsrtnat}

Prior work\cite{smith2023} shows...     % → shows^[1]
Several studies\cite{a,b,c} agree...    % → agree^[1–3]

\bibliography{refs}
```

**Numeric bracket alternative**:
```latex
\usepackage[numbers]{natbib}
\bibliographystyle{plainnat}

\cite{smith2023}   % [1]
\citep{smith2023}  % (Smith, 2023)
\citet{smith2023}  % Smith (2023)
```

**biblatex pathway**:
```latex
\usepackage[backend=biber,style=numeric-comp]{biblatex}
\addbibresource{refs.bib}

Per~\cite{smith2023}...
\printbibliography
```

#### 6.4 External Links
```latex
\url{https://example.com}
\href{https://example.com}{Visible label}
\href{mailto:a@b.com}{a@b.com}
```

#### 6.5 PDF Metadata & Outline
```latex
\hypersetup{
    pdftitle={Document Title},
    pdfauthor={Author Name},
    pdfsubject={Topic},
    pdfkeywords={keyword1, keyword2}
}
```
Bookmark trees are auto-generated from `\section` / `\chapter` hierarchy.

#### 6.6 Why Multiple Passes Matter

Label resolution requires at least two compilation passes:
```bash
# Resolve section / figure / table labels
python3 scripts/pdf.py convert.latex main.tex --runs 2

# Also resolves bibliography back-references
python3 scripts/pdf.py convert.latex main.tex --runs 3
```

If `??` placeholders persist after two passes, verify that every `\label` string has an exact `\ref` match.

#### 6.7 Navigation Troubleshooting

| Observation | Root cause | Resolution |
|-------------|-----------|------------|
| `??` in place of numbers | Only a single pass was run | Recompile with `--runs 2` |
| All links render in black | hyperref colour options omitted | Enable `colorlinks=true` |
| TOC items are not clickable | hyperref package missing | Load the package |
| `[?]` beside citations | `.bib` path incorrect or biber step skipped | Confirm path; rebuild |
| Bookmark pane empty | `bookmarks` option set to false | Switch to `bookmarks=true` |

---

## Package Catalogue

### Foundational
- `hyperref` (Rule 3)
- `geometry` (Rule 3)
- `listings` — `\lstset{basicstyle=\ttfamily\small, numbers=left, backgroundcolor=\color{gray!5}}`
- `enumitem` — `\setlist[itemize]{itemsep=0.3em, leftmargin=1.5em}`

### Tabular
`booktabs` · `longtable` · `multirow` · `array` · `colortbl`

### Visual & Charting
`tikz` · `pgfplots` · `float` · `wrapfig` · `subfig` / `subcaption`

### International & Typography
`fontspec` (XeLaTeX / LuaLaTeX) · `ctex`

### Mathematical
`amsmath` · `amssymb` · `amsthm` · `natbib` · `biblatex` · `siunitx`

### Algorithmic & Domain-Specific
`algorithm` + `algpseudocode` · `chemfig`

### Page Design
`tcolorbox` · `fancyhdr` · `titlesec` · `tocloft` · `multicol` · `setspace` · `microtype` · `parskip` · `adjustbox` · `marginnote`

### Code Listings
`listings` · `minted` (depends on Pygments)

---

## Scripts & Backends

| Script | Purpose |
|--------|---------|
| `pdf.py convert.latex` | Tectonic wrapper — log sanitisation, error highlighting, PDF metrics |

| Engine | Notes |
|--------|-------|
| Tectonic | Stand-alone LaTeX engine; packages are fetched transparently on demand |

---

## Operational Notes

### CJK Without Manual Font Setup
Tectonic resolves CJK font bundles on the fly — zero manual installation:

```latex
\usepackage{ctex}   % Tectonic pulls the font files automatically
```

### Cold-Start Latency
The very first compilation of a new document triggers package downloads:
- Initial build: 1–5 min (depends on network speed)
- Repeat builds with warm cache: 10–30 s

### Working Without Internet
Previously fetched packages are stored under `~/.cache/Tectonic/`. When offline, only cached packages are available; attempting to use a new one will fail.

### Tectonic vs a Full TeX Live Installation

| Dimension | Tectonic | Traditional pdflatex |
|-----------|----------|---------------------|
| Package acquisition | On-demand, transparent | Manual via `tlmgr` |
| Multi-pass compilation | Handled by the engine | Explicit re-invocations required |
| Reference resolution | Automatic | Requires bibtex/biber cycles |
| Disk footprint | Single binary | Full TeX Live ≈ 4 GB |
