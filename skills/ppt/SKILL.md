---
name: ppt
metadata:
  author: Z.AI
  version: "1.0"
description: "Presentation creation, editing, and analysis for .pptx files: (1) Creating new presentations, (2) Modifying or editing content, (3) Working with layouts, (4) Adding comments or speaker notes. Academic/paper-based presentations use the embedded Beamer module at end of this file (PDF output only)."
license: Proprietary. LICENSE.txt has complete terms
---

# PPT creation, editing, and analysis

## Quick Setup

```bash
bash "$SKILL_DIR/setup.sh"    # Interactive environment check + install
```

---

## Routing: Academic / Paper-Based Presentations → Beamer

> **STOP — check this before doing any work.**
>
> If the request matches **any** trigger below, **skip the PPTX workflow entirely**.
> Read **[`beamer.md`](beamer.md)** in this directory and follow its instructions.

| Trigger | Typical phrasing |
|---------|-----------------|
| Reading / summarizing a paper to make slides | "read this PDF and make slides", "make slides from this paper" |
| Academic / scientific / research presentation | "conference talk", "research presentation", "academic presentation" |
| Thesis or dissertation defense | "thesis defense", "proposal defense", "defense slides" |
| Any scholarly audience presentation | "academic PPT", "paper presentation", "research talk" |
| STEM / science courseware | "STEM slides", "science lecture", "math/physics/chemistry courseware" |
| User mentions a "paper" or "thesis" in any language | "present this paper", "talk about this thesis", "summarize this article into slides" |
| Uploaded file is clearly an academic paper | "make slides from this", "help me present this" — where the uploaded PDF contains academic indicators (abstract, keywords, references, DOI, author affiliations, journal name) in the first 3 pages |


> **Beamer output format: PDF-style slides only.**
>
> **Routing decision rule:** The trigger table above applies to **all languages**. For non-English requests, match by semantic meaning — mentally translate the user's intent into English and check against the triggers.
>
> **Important:** In many languages (e.g., Chinese, Japanese, Korean), "PPT" is a generic colloquial word for "slides" or "presentation" — it does NOT indicate a preference for `.pptx` format. Only route to the PPTX workflow when the user **explicitly** requests `.pptx` format (e.g., "I need a .pptx file", "export as pptx") or the content is clearly non-academic (e.g., marketing, business, teaching children).

---

## Overview

A user may ask you to create, edit, or analyze the contents of a .pptx file. A .pptx file is essentially a ZIP archive containing XML files and other resources that you can read or edit. You have different tools and workflows available for different tasks.

## Reading and analyzing content

### Text extraction

To read the text content of a presentation, convert it to markdown:

```bash
python -m markitdown path-to-file.pptx
```

### Raw XML access

For comments, speaker notes, slide layouts, animations, design elements, or complex formatting, unpack the presentation and inspect its raw XML.

#### Unpacking a file

```
python ooxml/scripts/unpack.py <office_file> <output_dir>
```

**Note**: `unpack.py` is at `skills/pptx/ooxml/scripts/unpack.py` relative to the project root. If not found, run `find . -name "unpack.py"` to locate it.

#### Key file structures

| Path | Contents |
|------|----------|
| `ppt/presentation.xml` | Main metadata and slide references |
| `ppt/slides/slide{N}.xml` | Per-slide content |
| `ppt/notesSlides/notesSlide{N}.xml` | Speaker notes |
| `ppt/comments/modernComment_*.xml` | Slide comments |
| `ppt/slideLayouts/` | Layout templates |
| `ppt/slideMasters/` | Master slide templates |
| `ppt/theme/` | Theme and styling |
| `ppt/media/` | Images and other media |

#### Typography and color extraction

**When emulating an existing design**, extract typography and colors before starting:

1. **Theme file** — `ppt/theme/theme1.xml`: colors (`<a:clrScheme>`), fonts (`<a:fontScheme>`)
2. **Slide content** — `ppt/slides/slide1.xml`: actual font usage (`<a:rPr>`) and colors
3. **Global search** — grep for `<a:solidFill>`, `<a:srgbClr>`, and font references across all XML files

---


## Creating a new PowerPoint presentation **using a template**

When the user upload a pptx file,and do not ask you to create a fully new pptx,you must create a presentation that follows an existing template's design, you'll need to duplicate and re-arrange template slides before then replacing placeholder content.

### Workflow
1. **Extract template text AND create visual thumbnail grid**:
   * Extract text: `python -m markitdown template.pptx > template-content.md`
   * Read `template-content.md`: Read the entire file. **NEVER set any range limits.**
   * Create thumbnail grids: `python scripts/thumbnail.py template.pptx`
   * See [Creating Thumbnail Grids](#creating-thumbnail-grids) section for details

2. **Analyze template and save inventory to a file**:
   * **Visual Analysis**: Review thumbnail grid(s) to understand layouts and design patterns
   * Create and save `template-inventory.md` containing:
     ```markdown
     # Template Inventory Analysis
     **Total Slides: [count]**
     **IMPORTANT: Slides are 0-indexed (first slide = 0, last slide = count-1)**

     ## [Category Name]
     - Slide 0: [Layout code if available] - Description/purpose
     - Slide 1: [Layout code] - Description/purpose
     [... EVERY slide must be listed ...]
     ```

3. **Create presentation outline based on template inventory**:
   * Choose layouts that match your content structure
   * **CRITICAL: Match layout structure to actual content**:
     - Single-column layouts: Use for unified narrative or single topic
     - Two-column layouts: Use ONLY when you have exactly 2 distinct items
     - Three-column layouts: Use ONLY when you have exactly 3 distinct items
     - Image + text layouts: Use ONLY when you have actual images
     - Count your actual content pieces BEFORE selecting layout
   * Save `outline.md` with content AND template mapping

4. **Duplicate, reorder, and delete slides using `rearrange.py`**:
   ```bash
   python scripts/rearrange.py template.pptx working.pptx 0,34,34,50,52
   ```

5. **Extract ALL text using the `inventory.py` script**:
   ```bash
   python scripts/inventory.py working.pptx text-inventory.json
   ```
   Read `text-inventory.json` entirely. **NEVER set range limits.**

6. **Generate replacement text and save to JSON file**:
   - Verify which shapes exist in inventory — only reference shapes that are present
   - Add `"paragraphs"` field to shapes that need content
   - **ALL text shapes from inventory will be cleared** unless you provide "paragraphs"
   - Paragraphs with bullets are automatically left-aligned
   - **Do NOT include bullet symbols** (bullet, -, *) in text — they're added automatically
   - Save to `replacement-text.json`

   **CRITICAL — JSON generation rules (learned from practice):**
   - **Always use `json.dump()` to write the file** — never write raw JSON strings in Python code.
     Raw strings may embed unescaped `"` characters (e.g. `"三遥"`, `"线上+线下"`) that silently
     break JSON parsing. `json.dump(data, f, ensure_ascii=False, indent=2)` handles all escaping
     automatically.
   - **Respect small box character limits** — check `width` and `font_size` from inventory before
     writing text for numeric/label shapes. A box of `width ≤ 0.7"` at `font_size ≥ 16pt` can hold
     roughly 3–4 characters max. Keep percentage values like `"99.7%"` to ≤ 4 chars, or shorten
     (e.g. `"100%"` instead of `"99.96%"`).
   - **Match replacement length to original** — for body text boxes, count characters in the
     original inventory text. Replacements significantly longer than the original will overflow.
     When in doubt, err shorter rather than longer.
   - **Test incrementally** — validate with a 5-slide subset before writing the full deck.
     `replace.py` blocks on overflow errors; fix them slide-by-slide rather than all at once.

7. **Apply replacements**:
   ```bash
   python scripts/replace.py working.pptx replacement-text.json output.pptx
   ```

## Creating a new PowerPoint presentation **without a template**

When creating a new PowerPoint presentation from scratch, use the **Design System + Component** workflow.

### Step 0 — Scene Classification (MANDATORY)

Before ANY design work, classify the presentation into one of the five scenes below:

| Scene | Keywords / Triggers |
|-------|-------------------|
| **Teaching / Training** | course, teaching, training, lecture, courseware, knowledge points, lesson plan |
| **Work Report / Review** | report, review, summary, retrospective, OKR, KPI, quarterly, annual |
| **Proposal / Pitch** | proposal, plan, pitch, investor, roadshow, business plan |
| **Thesis Defense / Academic** | thesis, defense, research, academic, topic, graduation project — **→ if source is a paper/PDF or audience is academic, read [`beamer.md`](beamer.md) instead** |
| **General** | None of the above, or user did not specify |

### Step 1 — Select Theme (MANDATORY)

Read [`themes.md`](themes.md) and select a theme based on scene and content tone.

**Output**: State your chosen theme name, your chosen **accent variant** (A/B/C), and the full color palette (primary-80, primary-90, primary-5, accent, etc.) before writing any code.

**Accent variant selection**: Each theme offers 3 accent colors (A = default, B, C). Choose the variant that best matches the content tone. You may also mix — e.g., use accent-A for most slides and accent-B for emphasis pages — but keep the primary palette constant.

Theme selection tips:
- Work report → Ocean or Graphite
- Proposal / pitch → Ocean, Sandstone, or Twilight
- Teaching / courseware → Forest or Ocean
- Thesis defense → read [`beamer.md`](beamer.md) (PDF output); if `.pptx` is explicitly required, use Graphite or Forest
- Tech keynote / product launch → Deep Mineral or Mono
- Cultural / lifestyle / brand → Warm Retro
- ESG / sustainability → Deep Forest or Forest
- Consumer / lifestyle / female audience → Coral
- Healthcare / wellness / eco → Mint
- SaaS / AI / onboarding → Azure
- Annual events / launches / high-energy → Ember
- General → any theme that fits the content

**After selecting a theme, note its Image Keywords and Mask Color** — you will need them in Step 5.5.

### Step 2 — Read Design System (MANDATORY)

Read [`design-system.md`](design-system.md). Understand the color system (mandatory) and creative principles (guidelines). The color scale is the only hard constraint — everything else (spacing, font sizes, layout) is flexible.

### Step 3 — Read HTML-to-PPTX Guide (MANDATORY)

Read [`html2pptx.md`](html2pptx.md) completely from start to finish. **NEVER set any range limits when reading this file.**

### Step 4 — Plan Slide Sequence

For each slide, decide:
1. Which **component** from [`components.md`](components.md) to use as a starting point (or design from scratch)
2. What **content** to fill in — **be generous with content; fill the slide**
3. How to **remix** the component — change spacing, font sizes, card styles, backgrounds, proportions to create variety

**KEY PRINCIPLES**:
- Read [`components.md`](components.md) for available starting points
- Also read [`data-viz-components.md`](data-viz-components.md) for data visualization components
- **Adjacent slides MUST look distinctly different** — vary layout structure, background, card treatment
- **Fill the slide**: Avoid large empty areas. If content is sparse, use larger typography, more generous spacing, bigger visual elements, or add decorative elements to create visual richness
- **Every content slide should have at least one non-text visual element** (color block, stat number, chart, icon, accent bar, photo, shape)
- Alternate between high-density and low-density pages for rhythm
- Use data visualization components when content contains numbers, comparisons, percentages
- **Tables are a great way to enrich content** — use `content-table*` or `content-chart-*` from components.md whenever content involves structured data, feature comparisons, schedules, or multi-attribute lists

**Anti-whitespace strategies** (use when a slide feels empty):
- Increase font sizes (e.g., body from 15pt to 17pt, headings from 22pt to 26pt)
- Add colored background blocks or sections
- Use a darker or tinted background instead of pure white
- Add decorative elements (accent bars, shapes, gradient bands)
- Expand card padding and spacing
- Use a photo background with mask overlay
- Switch from a text-heavy layout to a visual-heavy one

**Note: Not all whitespace is bad.** KPI pages, quote pages, and big-number focus pages are *designed* to have generous whitespace — that's intentional emphasis. Only fix whitespace on content-heavy pages (bullet lists, card grids, text blocks) where it looks accidental. See design-system.md §2.2 for the full distinction.

**Slide sequence pattern** (recommended):
```
1. Cover → PREFER cover-photo-mask (with downloaded photo + theme mask)
         → Fallback: cover-dark-hero (no photo needed) or cover-split
2. TOC → toc-card-grid (surface bg) / toc-big-number (visual variety)
3. Section divider (optional) → divider-photo-mask (different photo from cover)
                             → or divider-gradient / divider-bold-center
4. Content page A → light background, visually rich component
5. Content page B → ★ DARK BACKGROUND ★ (content-dark-bullets / dark-kpi / dark-split)
6. Section divider (optional) → different variant from #3
7. Content page C → image-based (split-text-visual, photo-cards, etc.)
8. Content page D → data-focused (kpi-row, big-number-focus, etc.)
9. Closing → dark background echoing cover color

Background rhythm target: ~40% white, ~25% surface/tinted, ~20% dark, ~15% photo
```

### Step 5 — Generate HTML from Components

For each slide:
1. Start from a component template in `components.md`, or design from scratch
2. Replace all `${variable}` placeholders with actual theme color values
3. Replace placeholder text with actual content
4. Adjust element count (add/remove cards, list items) as needed
5. **Before writing each slide's HTML**: Check the **Card Style Cookbook** at the top of components.md. Pick a different card style from the previous slide. Also check if this slide should use a **Dark Background Content Template**.
6. **Be creative**: Freely modify spacing, font sizes, card styles, proportions, background treatments, and decorative elements across pages. The only hard constraints are the html2pptx engine limitations (no flex-wrap, no negative margins, no DIV background-image, no CSS gradients). Use the design-system.md Quick Reference for suggested ranges, but treat them as starting points, not limits.
   - **Card styles**: Use the **Card Style Cookbook** in components.md to swap between shadow/outline/solid/accent-bar/dark card styles. No 2 adjacent slides should use the same card style.
   - **Title bars**: All content slides must use the **same header style** — pick one title bar variant from components.md and apply it consistently across every content page.
   - **Dark pages**: Use the **Dark Background Content Templates** in components.md for rhythm-breaking dark slides. Aim for at least 1 dark content page per 4 slides.
6. **Fill the slide**: If there's visible empty space, increase font sizes, add visual elements, expand padding, or use a colored/photo background. A well-filled slide looks professional; excessive whitespace looks unfinished.


### Step 5.8 — Visual Diversity Check (RECOMMENDED)

Before converting to PPTX, do a quick scan of your slide sequence:
- Are adjacent slides visually distinct (different layout, background, card style)?
- Is there enough variety across the deck (multiple layout structures, card treatments, background approaches)?
- Is there at least one dramatic "rhythm breaker" page?
- Do real photographs appear on covers and at least one other slide?

If the answer to any is "no", revise before proceeding.

### Step 6 — Convert and Validate

1. Create and run a JavaScript file using [`html2pptx.js`](scripts/html2pptx.js) to convert HTML slides to PowerPoint:
   ```javascript
   const pptxgen = require('pptxgenjs');
   const html2pptx = require('./html2pptx');
   
   const pptx = new pptxgen();
   pptx.layout = 'LAYOUT_16x9';
   
   // Optional: custom font configuration
   const fontConfig = { cjk: 'Microsoft YaHei', latin: 'Corbel' };
   
   // Process ALL slides with warnings collection
   const allWarnings = [];
   for (const htmlFile of slideFiles) {
     const { slide, placeholders, warnings } = await html2pptx(htmlFile, pptx, { fontConfig });
     allWarnings.push(...warnings);
   }
   
   // Add charts to placeholder areas if any
   if (placeholders.length > 0) {
       slide.addChart(pptx.charts.LINE, chartData, placeholders[0]);
   }
   
   await pptx.writeFile('output.pptx');
   ```

2. **Check warnings**: Review `warnings` output. **Blocking issues** (overflow, font < 11pt) must be fixed. **Non-blocking warnings** (bounds, balance, density) are suggestions — use judgment on whether to fix them.

3. **Visual validation**: Generate thumbnails and inspect for layout issues:
   ```bash
   python scripts/thumbnail.py output.pptx workspace/thumbnails --cols 4
   ```
   - Read and carefully examine the thumbnail image for:
     - **Text cutoff**: Text being cut off by header bars, shapes, or slide edges
     - **Text overlap**: Text overlapping with other text or shapes
     - **Positioning issues**: Content too close to slide boundaries or other elements
     - **Contrast issues**: Insufficient contrast between text and backgrounds
     - **Consistency check**: All body text same font and size? All page margins consistent? All content slides use the same header style?
   - If issues found, adjust HTML and regenerate
   - Repeat until all slides are visually correct

### Step 6.5 — Final Quality Check (RECOMMENDED)

Before finalizing, do a quick visual quality scan:

- ✅ Real photographs used (cover + at least 1 other slide)
- ✅ Background variety (at least 3 different treatments)
- ✅ No 2 consecutive slides look the same
- ✅ Multiple layout structures used (aim for 5+)
- ✅ Card styles vary across pages
- ✅ Every slide has a clear visual focal point
- ✅ At least 1 dramatic full-bleed or dark page exists
- ✅ All referenced image files exist and are > 10KB

If major issues are found, fix and regenerate. Minor imperfections are acceptable — don't over-optimize at the cost of creativity.

### Design Hard Rules (ALWAYS ENFORCED)

These rules are split into **engine constraints** (violating them causes broken output) and **design principles** (ensuring quality).

**Engine Constraints (technical — cannot be violated):**
```
[ENGINE] Slide canvas is 720×405pt — content exceeding this overflows
[ENGINE] All colors must come from the theme color scale — no arbitrary grays (#666 #999 #DDD)
[ENGINE] font-family must include a CJK font name to trigger correct mapping
[ENGINE] Do not use flex-wrap — multi-row layouts must use separate flex containers
[ENGINE] Do not use negative margins — they cause text stacking in PPT
[ENGINE] Multi-column equal-width cards must use fixed width + flex-shrink:0, not flex:1
[ENGINE] Background images only work on <body>, not <div> — DIV background-image is not supported
[ENGINE] Images must be local file paths, not URLs
[ENGINE] Titles and short labels (<10 characters) should use white-space:nowrap
[ENGINE] Numeric sequences (e.g. 01/02/03, $12M) should use white-space:nowrap
```

**Design Principles (creative quality — strongly encouraged):**
```
[DESIGN] Adjacent slides should use different layouts, backgrounds, and card styles
[DESIGN] Every content slide should have at least one visual focal point
[DESIGN] Cover and closing page should echo each other
[DESIGN] Prefer cutting content over shrinking fonts — only when a slide is already well-filled and overflow is imminent; never cut content preemptively
[DESIGN] Aim for 5+ layout structures, 3+ card styles, 3+ background treatments across the deck
[DESIGN] Insert a rhythm-breaking page every 3-4 slides
[DESIGN] Background images should have a mask overlay for text readability
[DESIGN] Content should be vertically balanced, not crammed at the top
[DESIGN] Accent colors must have HSL saturation ≤ 65% — high-saturation colors look garish on projection
[DESIGN] Large color blocks (>15% page area) should use S ≤ 50% — only small accents can be vivid
[DESIGN] All content slides must use the same header style — pick one title bar variant and apply it consistently across all content pages
```

---

## Editing an existing PowerPoint presentation

When editing slides in an existing PowerPoint presentation, work with raw Office Open XML (OOXML) format.

### Workflow
1. **MANDATORY - READ ENTIRE FILE**: Read [`ooxml.md`](ooxml.md) (~500 lines) completely from start to finish. **NEVER set any range limits when reading this file.**
2. Unpack the presentation: `python ooxml/scripts/unpack.py <office_file> <output_dir>`
3. Edit the XML files (primarily `ppt/slides/slide{N}.xml` and related files)
4. **CRITICAL**: Validate immediately after each edit: `python ooxml/scripts/validate.py <dir> --original <file>`
5. Pack the final presentation: `python ooxml/scripts/pack.py <input_directory> <office_file>`



## Creating Thumbnail Grids

```bash
python scripts/thumbnail.py template.pptx [output_prefix]
```

**Features**:
- Creates: `thumbnails.jpg` (or `thumbnails-1.jpg`, `thumbnails-2.jpg` for large decks)
- Default: 5 columns, max 30 slides per grid (5x6)
- Custom prefix: `python scripts/thumbnail.py template.pptx my-grid`
- Adjust columns: `--cols 4` (range: 3-6)
- Slides are zero-indexed (Slide 0, Slide 1, etc.)

## Converting Slides to Images

1. **Convert PPTX to PDF**:
   ```bash
   soffice --headless --convert-to pdf template.pptx
   ```

2. **Convert PDF pages to JPEG images**:
   ```bash
   pdftoppm -jpeg -r 150 template.pdf slide
   ```

## Code Style Guidelines
**IMPORTANT**: When generating code for PPTX operations:
- Write concise code
- Avoid verbose variable names and redundant operations
- Avoid unnecessary print statements

## Dependencies

Required dependencies (should already be installed):

- **markitdown**: `pip install "markitdown[pptx]"` (text extraction)
- **pptxgenjs**: `npm install -g pptxgenjs` (creating presentations)
- **playwright**: `npm install -g playwright` (HTML rendering)
- **react-icons**: `npm install -g react-icons react react-dom` (icons)
- **sharp**: `npm install -g sharp` (SVG rasterization and image processing)
- **LibreOffice**: `sudo apt-get install libreoffice` (PDF conversion)
- **Poppler**: `sudo apt-get install poppler-utils` (pdftoppm)
- **defusedxml**: `pip install defusedxml` (secure XML parsing)
