# Route: LaTeX Beamer → PDF (via Tectonic)

Write LaTeX Beamer source, then compile to PDF:

```bash
python3 scripts/pdf.py convert.latex main.tex
python3 scripts/pdf.py convert.latex main.tex --runs 2   # for ToC / refs
```

Tectonic compiles the full Beamer document — all features work: overlays,
TikZ, custom themes, math, transitions, fragile frames, appendix.

---

## 1. Compilation

```bash
# Standard build
python3 scripts/pdf.py convert.latex main.tex

# Two passes (resolves ToC, \ref, section counters in overlays)
python3 scripts/pdf.py convert.latex main.tex --runs 2

# Verbose log (see all Tectonic output)
python3 scripts/pdf.py convert.latex main.tex --keep-logs
```

---

## 2. Beamer Document Structure

### Minimal Working Example

```latex
\documentclass[aspectratio=169, 11pt]{beamer}
\usepackage[fontset=fandol]{ctex}   % CJK support (Tectonic auto-downloads)
\usetheme{Madrid}

\title{Introduction to Quantum Computing}
\author{Prof. Zhang}
\institute{Tsinghua University}
\date{\today}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}{Table of Contents}
  \tableofcontents
\end{frame}

\section{Quantum Bits}

\begin{frame}{Quantum States}
  \[ |\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2+|\beta|^2=1 \]
\end{frame}

\end{document}
```

**Core elements:**
- `\documentclass{beamer}` — Presentation document class
- Preamble — Theme, colors, fonts, global settings
- `\begin{frame}...\end{frame}` — Each frame = one slide

---

## 3. Theme System (Complete Reference)

Beamer themes consist of five independent layers that can be freely combined:

### 3.1 Presentation Themes (control overall layout)

| Theme | Features | Suitable For |
|---|---|---|
| `default` | Minimal, no decoration | Formal academic, minimalist |
| `Madrid` | Bottom info bar + section navigation | University courses, academic talks |
| `Berlin` | Top mini-frames + side color blocks | Long technical presentations |
| `Warsaw` | Top circular navigation + gradient title | Structured content |
| `CambridgeUS` | Two-tone minimalist | Math-heavy presentations |
| `Boadilla` | Clean footer | Business, concise style |
| `Singapore` | Top dot navigation | Modern feel, multi-section |
| `AnnArbor` | Yellow-blue dual tone | Michigan style |
| `Antibes` | Blue sidebar tree | Multi-subsection structure |
| `Bergen` | Blue title bar | Formal academic |
| `Copenhagen` | Top bar + navigation | Classic blue scheme |
| `Darmstadt` | Top progress dots | Multi-section structure |
| `Dresden` | Top mini-frames | Similar to Berlin |
| `Frankfurt` | Top dots | Clear section structure |
| `Goettingen` | Right sidebar | Navigation-focused |
| `Luebeck` | Top blue bar | Clean modern |
| `Malmoe` | Minimal title | Content-first |
| `Montpellier` | Top tree navigation | Hierarchical content |
| `Pittsburgh` | No color blocks, minimal | White scheme |
| `Rochester` | Dark title | Dark scheme |
| `Szeged` | Blue title bar | Hungarian style |

```latex
\usetheme{Madrid}
```

### 3.2 Color Themes

| Color Theme | Dominant Color |
|---|---|
| `default` | Dark blue |
| `albatross` | Yellow tones |
| `beaver` | Dark red / maroon |
| `beetle` | Gray-blue |
| `crane` | Orange-yellow |
| `dolphin` | Blue-white |
| `dove` | Gray-white (near monochrome) |
| `fly` | Gray tones |
| `lily` | Red-blue |
| `orchid` | Purple tones |
| `rose` | Pink |
| `seagull` | Gray |
| `seahorse` | Blue-purple |
| `whale` | Deep sea blue |
| `wolverine` | Yellow-blue contrast |

```latex
\usecolortheme{dolphin}
```

### 3.3 Font Themes

```latex
\usefonttheme{default}            % Sans-serif (Beamer default)
\usefonttheme{serif}              % Serif font (academic feel)
\usefonttheme{structurebold}      % Bold structural elements
\usefonttheme{structureitalic}    % Italic structural elements
\usefonttheme{structuresmallcapsserif}  % Small caps serif
\usefonttheme{professionalfonts}  % Keep math fonts unchanged
```

### 3.4 Inner Themes (control list, block, and title styles)

```latex
\useinnertheme{default}    % Triangle bullet points
\useinnertheme{circles}    % Circle bullet points
\useinnertheme{rectangles} % Square bullet points
\useinnertheme{rounded}    % Rounded blocks
\useinnertheme{inmargin}   % Margin numbers
```

### 3.5 Outer Themes (control header, footer, and sidebar)

```latex
\useoutertheme{default}      % No decoration
\useoutertheme{infolines}    % Bottom three-column info
\useoutertheme{miniframes}   % Top mini-frame navigation
\useoutertheme{shadow}       % Shadow title
\useoutertheme{sidebar}      % Sidebar
\useoutertheme{smoothbars}   % Gradient top bar
\useoutertheme{smoothtree}   % Gradient tree
\useoutertheme{split}        % Top-bottom split
\useoutertheme{tree}         % Tree navigation
```

### 3.6 Custom Colors

```latex
\definecolor{myblue}{HTML}{2B5EA7}
\definecolor{mygray}{HTML}{F0F0F0}

\setbeamercolor{frametitle}{bg=myblue, fg=white}
\setbeamercolor{structure}{fg=myblue}
\setbeamercolor{block title}{bg=myblue!85, fg=white}
\setbeamercolor{block body}{bg=myblue!8}
\setbeamercolor{alerted text}{fg=red!70!black}
\setbeamercolor{title}{fg=white}
\setbeamercolor{subtitle}{fg=white!80}
```

### 3.7 Common Template Customizations

```latex
% Remove navigation symbols (almost always needed)
\setbeamertemplate{navigation symbols}{}

% Custom footer (frame number)
\setbeamertemplate{footline}{%
  \hfill\insertframenumber/\inserttotalframenumber\hspace{1em}\vspace{0.5em}
}

% Custom bullet points
\setbeamertemplate{itemize item}{\textbullet}
\setbeamertemplate{itemize subitem}{--}

% Logo (bottom-right corner)
\logo{\includegraphics[height=0.7cm]{logo.png}}

% Rounded blocks
\setbeamertemplate{blocks}[rounded][shadow=true]
```

---

## 4. Overlays & Stepwise Reveal

Tectonic fully supports all overlay syntax — rendered as separate pages in the PDF.

### 4.1 Basic Overlay Specifications

```latex
\item<1->    % Show from layer 1, persist
\item<2->    % Show from layer 2
\item<1-3>   % Show on layers 1 to 3 only
\item<2>     % Show on layer 2 only
\item<-3>    % Show before layer 3
```

### 4.2 \pause — Simplest Stepwise Reveal

```latex
\begin{frame}{Research Workflow}
  Data collection phase
  \pause
  Feature engineering phase
  \pause
  Model training phase
\end{frame}
```

### 4.3 List Stepwise Reveal

```latex
\begin{frame}{Research Contributions}
  \begin{itemize}
    \item<1-> Propose a novel loss function
    \item<2-> Prove theoretical convergence
    \item<3-> Outperform SOTA on five benchmarks
  \end{itemize}
\end{frame}
```

### 4.4 \only vs \uncover vs \visible

```latex
% \only: no space reserved (causes layout jump)
\only<1>{Step one explanation}
\only<2>{Step two explanation}

% \uncover: reserves space but transparent (recommended, stable layout)
\uncover<2->{Follow-up steps explanation}

% \visible: reserves space but not rendered
\visible<3->{Step three}
```

### 4.5 \alert — Highlight Specific Layers

```latex
\begin{frame}{Key Findings}
  \begin{itemize}
    \item \alert<1>{Accuracy improved by 13\%}
    \item \alert<2>{Inference speed improved 2.4x}
    \item \alert<3>{Memory reduced by 40\%}
  \end{itemize}
\end{frame}
```

### 4.6 \textcolor & Conditional Content

```latex
\begin{frame}{Comparison}
  Baseline method: \textcolor{red}{82\%}\\
  \only<2->{Our method: \textcolor{green!60!black}{\textbf{93\%}}}
\end{frame}
```

### 4.7 Block Overlays

```latex
\begin{frame}{Analysis}
  \begin{block}<1-2>{Problem Definition}
    Formal description ...
  \end{block}
  \begin{block}<2->{Solution}
    Core approach ...
  \end{block}
\end{frame}
```

---

## 5. Content Layout Templates

### 5.1 Block Series (Three Styles)

```latex
\begin{frame}{Core Findings}
  \begin{block}{Main Conclusion}
    The new method outperforms baselines in both accuracy and speed.
  \end{block}

  \begin{alertblock}{Caveats}
    Results are validated on specific datasets; generalization needs further study.
  \end{alertblock}

  \begin{exampleblock}{Example}
    Achieved 99.1\% accuracy on the MNIST dataset.
  \end{exampleblock}
\end{frame}
```

### 5.2 Column Layout

```latex
\begin{frame}{Experimental Comparison}
  \begin{columns}[T]   % [T] top-aligned
    \column{0.48\textwidth}
    \begin{block}{Baseline Method}
      Accuracy: 82\% \\
      Time: 3.2 hours
    \end{block}

    \column{0.48\textwidth}
    \begin{block}{Our Method}
      Accuracy: \textbf{91\%} \\
      Time: \textbf{1.7 hours}
    \end{block}
  \end{columns}
\end{frame}
```

Column alignment options: `[T]` top-aligned · `[c]` centered · `[b]` bottom-aligned

### 5.3 Math Formula Frame

```latex
\begin{frame}{Loss Function}
  \begin{equation*}
    \mathcal{L} = -\sum_{i=1}^{N} y_i \log \hat{y}_i + \lambda \|\theta\|_2^2
  \end{equation*}
  \begin{itemize}
    \item $y_i$: Ground truth label
    \item $\hat{y}_i$: Predicted probability
    \item $\lambda$: Regularization coefficient
  \end{itemize}
\end{frame}
```

### 5.4 Theorem Environment

```latex
\begin{frame}{Convergence Theorem}
  \begin{theorem}[Convergence Rate]
    Under Lipschitz conditions, gradient descent converges at $O(1/\sqrt{T})$.
  \end{theorem}

  \begin{proof}
    Follows from Lipschitz continuity and step size $\eta = 1/L$.\qed
  \end{proof}

  \begin{corollary}
    The stochastic gradient variant converges in expectation at $O(1/\sqrt{T})$.
  \end{corollary}
\end{frame}
```

### 5.5 Table Frame

```latex
\begin{frame}{Performance Comparison}
  \centering
  \begin{tabular}{lcc}
    \toprule
    Method & Accuracy & Time(s) \\
    \midrule
    SVM           & 0.85 & 120 \\
    Random Forest  & 0.89 & 65  \\
    \textbf{Ours}  & \textbf{0.93} & \textbf{42} \\
    \bottomrule
  \end{tabular}
\end{frame}
```

### 5.6 TikZ Flowchart

TikZ is fully supported in Tectonic — no pre-rendering needed:

```latex
\begin{frame}{Algorithm Pipeline}
  \centering
  \begin{tikzpicture}[
      node distance=2cm,
      box/.style={rectangle, rounded corners=4pt, draw=myblue, thick,
                  fill=myblue!10, minimum width=2.2cm, minimum height=0.9cm, font=\small},
      arr/.style={-Stealth, thick, myblue}
  ]
    \node[box] (A) {Input Data};
    \node[box, right of=A] (B) {Feature Extraction};
    \node[box, right of=B] (C) {Model Inference};
    \node[box, right of=C] (D) {Output Results};
    \draw[arr] (A) -- (B);
    \draw[arr] (B) -- (C);
    \draw[arr] (C) -- (D);
  \end{tikzpicture}
\end{frame}
```

### 5.7 Code Display Frame

```latex
\begin{frame}[fragile]{Training Loop}   % [fragile] is required
  \begin{verbatim}
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
  \end{verbatim}
\end{frame}
```

Or use `listings` for syntax highlighting (more polished):

```latex
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  keywordstyle=\color{blue}\bfseries,
  commentstyle=\color{gray},
  backgroundcolor=\color{gray!5},
  numbers=left, numberstyle=\tiny\color{gray},
  frame=single, framerule=0.5pt
}

\begin{frame}[fragile]{Python Code}
  \begin{lstlisting}[language=Python]
def train(model, dataloader):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
  \end{lstlisting}
\end{frame}
```

### 5.8 Image Frame

```latex
\begin{frame}{System Architecture}
  \centering
  \includegraphics[width=0.85\textwidth]{figures/architecture.png}
  \captionof{figure}{End-to-end training pipeline}
\end{frame}
```

### 5.9 Custom Checklist Frame

```latex
\begin{frame}{Summary of Contributions}
  \begin{itemize}
    \item[\checkmark] New method proposed: ...
    \item[\checkmark] Theoretical proof: ...
    \item[\checkmark] Experimental validation: ...
    \item[$\square$] Future work: ...
  \end{itemize}
\end{frame}
```

---

## 6. Advanced Features

### 6.1 Progress Bar (Manual Implementation)

```latex
% Define in preamble
\definecolor{progressbar}{HTML}{2B5EA7}
\setbeamertemplate{footline}{
  \begin{beamercolorbox}[wd=\paperwidth,ht=2pt]{progressbar}
    \rule{\dimexpr\paperwidth*\insertframenumber/\inserttotalframenumber}{2pt}
  \end{beamercolorbox}
  \vspace{2pt}
  \hfill\tiny\insertframenumber/\inserttotalframenumber\hspace{1em}\vspace{3pt}
}
```

### 6.2 Section Title Pages (Automatic)

```latex
% Add in preamble — auto-inserts a section title page before each \section
\AtBeginSection[]{
  \begin{frame}
    \vfill
    \centering
    \begin{beamercolorbox}[sep=8pt,center,shadow=true,rounded=true]{title}
      \usebeamerfont{title}\insertsectionhead\par
    \end{beamercolorbox}
    \vfill
  \end{frame}
}
```

### 6.3 Handout Mode

```latex
% Generate overlay-free handout version (only final state of each frame)
\documentclass[handout, aspectratio=169]{beamer}
```

Combine with `pgfpages` to compress multiple frames onto one page:

```latex
\usepackage{pgfpages}
\pgfpagesuselayout{4 on 1}[a4paper, border shrink=5mm]
```

### 6.4 Frame Reuse (\againframe)

```latex
\begin{frame}[label=keyresult]{Key Result}
  Accuracy improved by 13\%.
\end{frame}

% Show again at any later point
\againframe{keyresult}
```

### 6.5 Appendix (Excluded from Total Page Count)

```latex
\appendix

\begin{frame}{Appendix: Detailed Derivation}
  % Backup slides for Q&A
\end{frame}
```

### 6.6 Speaker Notes

```latex
\begin{frame}{Main Conclusions}
  \begin{itemize}
    \item Conclusion one
    \item Conclusion two
  \end{itemize}
  \note{
    Emphasize the data supporting conclusion one.\\
    Anticipate questions about dataset size.
  }
\end{frame}
```

Output PDF with notes: add in preamble:
```latex
\setbeameroption{show notes on second screen=right}
```

### 6.7 Widescreen Aspect Ratios

```latex
\documentclass[aspectratio=169]{beamer}   % 16:9 (recommended)
\documentclass[aspectratio=1610]{beamer}  % 16:10
\documentclass[aspectratio=43]{beamer}    % 4:3 (traditional)
\documentclass[aspectratio=141]{beamer}   % 1.41:1 (A4)
```

---

## 7. Complete Template (Academic Presentation, 16:9, with CJK Support)

```latex
\documentclass[aspectratio=169, 11pt]{beamer}
\usepackage[fontset=fandol]{ctex}
\usepackage{amsmath, amssymb, mathtools}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric}

% -- Theme ---------------------------------------------------------------
\usetheme{Madrid}
\usecolortheme{dolphin}
\usefonttheme{professionalfonts}
\setbeamertemplate{navigation symbols}{}

% -- Colors --------------------------------------------------------------
\definecolor{myblue}{HTML}{2B5EA7}
\setbeamercolor{frametitle}{bg=myblue, fg=white}
\setbeamercolor{structure}{fg=myblue}
\setbeamercolor{block title}{bg=myblue!85, fg=white}
\setbeamercolor{block body}{bg=myblue!8}

% -- Auto section title page ---------------------------------------------
\AtBeginSection[]{
  \begin{frame}
    \vfill\centering
    \begin{beamercolorbox}[sep=8pt,center,rounded=true]{title}
      \usebeamerfont{title}\insertsectionhead\par
    \end{beamercolorbox}
    \vfill
  \end{frame}
}

% -- Metadata ------------------------------------------------------------
\title{Paper Title}
\subtitle{Subtitle}
\author{Author Name}
\institute{Affiliation}
\date{\today}

% ========================================================================
\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}{Table of Contents}
  \tableofcontents
\end{frame}

\section{Introduction}

\begin{frame}{Research Background}
  \begin{itemize}
    \item<1-> Importance of the problem
    \item<2-> Limitations of existing methods
    \item<3-> Contributions of this work
  \end{itemize}
\end{frame}

\section{Method}

\begin{frame}{Core Formula}
  \begin{equation*}
    \mathcal{L} = -\sum_{i=1}^{N} y_i \log \hat{y}_i + \lambda \|\theta\|_2^2
  \end{equation*}
  \begin{itemize}
    \item $y_i$: Ground truth label
    \item $\hat{y}_i$: Predicted probability
    \item $\lambda$: Regularization coefficient
  \end{itemize}
\end{frame}

\begin{frame}{Method Comparison}
  \begin{columns}[T]
    \column{0.48\textwidth}
    \begin{block}{Baseline Method}
      Accuracy: 82\% \\
      Time: 3.2 hours
    \end{block}
    \column{0.48\textwidth}
    \begin{block}{Our Method}
      Accuracy: \textbf{91\%} \\
      Time: \textbf{1.7 hours}
    \end{block}
  \end{columns}
\end{frame}

\section{Experiments}

\begin{frame}{Performance Comparison}
  \centering
  \begin{tabular}{lcc}
    \toprule
    Method & Accuracy & Time(s) \\
    \midrule
    SVM           & 0.85 & 120 \\
    Random Forest  & 0.89 & 65  \\
    \textbf{Ours}  & \textbf{0.93} & \textbf{42} \\
    \bottomrule
  \end{tabular}
\end{frame}

\section{Conclusion}

\begin{frame}{Summary}
  \begin{itemize}
    \item[\checkmark] Contribution 1: Proposed new method
    \item[\checkmark] Contribution 2: Theoretical proof
    \item[\checkmark] Contribution 3: Experimental validation
  \end{itemize}
  \begin{block}{Future Work}
    Extend to larger-scale datasets and explore cross-domain generalization.
  \end{block}
\end{frame}

\begin{frame}
  \centering\Large Thank You!\\[0.8em]
  \normalsize Questions Welcome
\end{frame}

\end{document}
```

Compile:
```bash
python3 scripts/pdf.py convert.latex main.tex --runs 2
```

---

## 8. Common Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| CJK characters display as boxes | Missing CJK font package | Add `\usepackage[fontset=fandol]{ctex}` |
| `??` appears in ToC or references | Only compiled once | Add `--runs 2` |
| `[fragile]` missing error | verbatim/lstlisting frame | Add `[fragile]` after `\begin{frame}` |
| Overlay not working | Forgot `\pause` or `<n->` in frame | Check overlay specification syntax |
| TikZ compilation failure | Missing tikzlibrary | Add `\usetikzlibrary{...}` |
| Math font distortion | Missing professionalfonts | Add `\usefonttheme{professionalfonts}` |
| Frame exceeds one page (content overflow) | Too much content | Add `[allowframebreaks]` or split frame |

---

## 9. Dependencies

| Tool | Purpose |
|---|---|
| `scripts/tectonic` | LaTeX compilation engine (local binary) |
| `scripts/pdf.py convert.latex` | Tectonic wrapper with log filtering + PDF stats |
