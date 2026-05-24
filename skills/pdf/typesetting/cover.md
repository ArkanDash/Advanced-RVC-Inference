# Cover Design V3.0 - Cover Layout Engine Specification

> The cover is the first impression. Either skip it, or build it like architecture.

---

## ⚠️ Critical Rules (Read First)

1. **Cover is OPTIONAL.** Do NOT force a cover on documents that don't need one. When in doubt, skip.
2. **Unified cover system.** All routes (Report, Creative, Academic) use the HTML/Playwright cover system. Templates 01-07 are general-purpose; Templates 08-10 are academic-specific (dark backgrounds, scholarly typography); **Template 11 is institutional (white bg, black border frame, structured fields)**. All routes generate covers via Playwright and merge via pypdf.
3. **Single PDF output.** Never deliver a separate cover PDF.
   - **Creative route**: Cover is part of the same HTML document → single PDF output inherently.
   - **Report route**: Cover PDF (Playwright) + body PDF (ReportLab) → merged via pypdf into one final PDF.
   - **Academic route**: Cover PDF (Playwright) + body PDF (Tectonic) → merged via pypdf into one final PDF.
4. **Page isolation.** Cover content must NEVER share a page with TOC, body text, or any subsequent content. Report/Academic: isolation is inherent in the merge pipeline. Creative: CSS page-break enforces isolation. Cover + TOC on the same page = **critical bug**.

---

## When to Include a Cover

| Document Type | Cover Needed | Notes |
|---------------|-------------|-------|
| Formal report (annual, research, white paper) | ✅ Required | Conveys professionalism |
| Proposal / plan | ✅ Required | First impression is everything |
| Resume | ❌ Not needed | Content itself is the cover |
| Menu / flyer / card | ❌ Not needed | Single page or function-oriented |
| Invitation | ❌ Not needed | The front side IS the cover |
| Lab report / academic paper | ⚠️ Situational | Add when template requires it |
| Portfolio / lookbook | ✅ Required | Cover sets the tone |

---

# PART 0: GLOBAL ENGINE ARCHITECTURE (Mandatory)

These three architectural upgrades are **mandatory** for ALL cover rendering. They eliminate 90% of squished/misaligned/overflow bugs at the engine level.

---

## A0.0 - Global Base Parameters

**All templates MUST obey these.**

```
W = page total width
H = page total height
U = W * 0.05           # Base spacing unit (5% of page width)
                        # All spacing should be multiples of U
```

**Coordinate origin:** `(0, 0)` at top-left corner. Some PDF libraries (ReportLab) use bottom-left origin - invert Y axis accordingly: `y_pdf = H - y_spec`.

---

## A0.1 - Absolute Anchor Grid (Replaces Flow Layout)

**Old (DEPRECATED):** `Y = previous_element_Y + height + spacing` - if one element overflows, everything below shifts and gets crushed.

**New (MANDATORY):** Page height = 100%. Every element group gets an **absolute percentage Y-anchor**. Elements may only grow **within their own bounding box**. They NEVER push or compress other blocks.

### Implementation Rule

```
# WRONG - flow layout (banned)
y_cursor = PAGE_H - top_margin
y_cursor -= title_height
y_cursor -= gap
y_cursor -= subtitle_height   # ← if title wraps, subtitle gets crushed

# RIGHT - absolute anchor grid
ANCHOR_TITLE_Y  = H * 0.30
ANCHOR_SUMMARY_Y = H * 0.50
ANCHOR_META_Y   = H * 0.70
ANCHOR_FOOTER_Y = H * 0.90
# Each block renders at its anchor, independent of others
```

### Bounding Box Containment

Each block has a **maximum bounding box** defined by two consecutive anchors:
- Block can grow downward within `[own_anchor_Y, next_anchor_Y - min_gap]`
- If content exceeds its bounding box → trigger overflow protection (see Part 3)
- Blocks NEVER consume space belonging to adjacent blocks

---

## A0.2 - Typography Weight & Spacing System

Use **weight**, **letter-spacing**, and **opacity** to create hierarchy - not just font size.

### Mandatory Type Roles

| Role | Size | Weight | Letter-Spacing | Line-Height | Opacity | Purpose |
|------|------|--------|----------------|-------------|---------|---------|
| **Kicker / Footer** (decorative text) | 16pt | Regular | 3pt (very wide) | - | 60% | Wide spacing + transparency makes 16pt text feel delicate and recessive |
| **Summary / Description** (summary paragraph) 🆕 | 16-18pt | Regular | normal | **1.6** | 85% | **Fill visual space** - 2-4 lines of descriptive text that prevents empty covers |
| **Meta / Subtitle** (secondary text) | 20-22pt | Light / Regular | normal | 1.4 | 85% | Comfortable reading rhythm, clear secondary hierarchy |
| **Hero Title** (main title) | 45-65pt (CJK: 50-80pt) | Black / Heavy (extra bold) | normal-tight | **1.15** (multi-line) | 100% | Must create overwhelming scale contrast; visually dominates the page. CJK characters need +15-20% size to match Latin visual weight |

### 🔴 Data-to-Drawer Binding Rule

> **Hero Title = Company/entity name. Kicker = Report type/subtitle. Never reverse.**

When users provide structured information (company name + report name/type), they must be bound to typography drawers by these rules:

| User Data Field | Bound to Typography Role | Notes |
|-------------|-------------|------|
| **Company/entity name** (e.g. "GREENTECH") | **Hero Title** (45-65pt Heavy) | Company name is the **absolute visual center**, largest font, heaviest weight |
| **Report type/subtitle** (e.g. "2025 Annual Report Summary") | **Kicker** (16pt, letter-spacing 3pt, opacity 60%) | Report name is decorative supplementary text, placed in small text position at top-left/above title |
| **Summary/description** | **Summary** (16-18pt) | Detailed description text |
| **Date/author/version** | **Meta** (20-22pt) | Auxiliary information |
| **Document number/org signature** | **Footer** (16pt, opacity 60%) | Bottom closing |

**Mapping priority (when ambiguous):**
1. If user provides only one name → treat as company/entity name, bind to Hero Title
2. If user provides two names → shorter/more brand-like → Hero Title; longer/descriptive → Kicker or Summary
3. If user explicitly labels "title" and "subtitle" → title → Hero Title, subtitle → Kicker
4. **Never use report type names (e.g. "Annual Report", "White Paper") as Hero Title's largest text** - report type is always Kicker-level decorative text

### The Summary Block Rule (Anti-Void Iron Rule) 🆕

> Every cover MUST include a Summary/Description text block. If the user provides no summary, the system MUST auto-generate one.

**Why:** A cover with only a title and date looks barren. The Summary block physically fills 2-4 lines of space, preventing the "empty field" aesthetic.

**Auto-generation rule:** When no summary/description is provided:
```python
# Generate a default summary
if not summary_text:
    summary_text = f"This report was generated by {org_name or 'the system'}, containing comprehensive data analysis and insights."
    # Or in English:
    # summary_text = f"This report presents comprehensive analysis and key insights prepared by {org_name or 'the organization'}."
```

**Constraints:**
- Width: template-specific (typically `W * 0.5` to `W * 0.6`)
- Lines: 2-4 lines (auto-wrap at width boundary)
- Never truncate summary - if too long, reduce to 4 lines max with `...`

### Font Weight Fallback

- If font family lacks Black/Heavy weight → use Bold + slightly larger size (+4pt)
- If font family lacks Light weight → use Regular + increased letter-spacing (+1pt)
- CJK fonts: STSong-Light for Regular/Light roles; for Heavy/Black → increase size by 15% to compensate for single-weight CJK fonts
- English kickers/footers: **FORCE UPPERCASE** via code (`text.upper()` / `toUpperCase()`)

---

## A0.3 - Z-Index Layer Management

All cover elements must be rendered in strict layer order. No exceptions.

| Layer | Z-Index | Contents | Rules |
|-------|---------|----------|-------|
| **Layer 0** (base) | 0 | Background fill (white / light gray) | Always rendered first; full page |
| **Layer 1** (background) | 1 | Grids, watermark letters, decorative blocks, large clipped graphics | **MUST enable clip-path** - background elements may extend beyond logical bounds but must be clipped to page physical bounds. Never let background elements inflate PDF page size. |
| **Layer 2** (structure) | 2 | Ultra-thin divider lines, sidebars, corner crop marks | Structural guides that define spatial zones |
| **Layer 3** (content) | 3 | All readable text content | Rendered last, always on top |

### Clip-Path Enforcement

> **Since V2.1, all covers are rendered via HTML/CSS.** The canonical clip pattern is CSS `overflow: hidden`. The ReportLab Python example below is kept as legacy reference for body-page background elements only.

```css
/* HTML/CSS cover (CANONICAL): clip background overflow */
.cover-bg-layer {
  position: absolute;
  inset: 0;
  overflow: hidden;  /* MANDATORY */
  z-index: 1;
}
```

```python
# ReportLab (legacy reference, body pages only): clip background elements to page bounds
canvas.saveState()
p = canvas.beginPath()
p.rect(0, 0, W, H)
canvas.clipPath(p, stroke=0)
# ... render background elements (Layer 1 ONLY) ...
canvas.restoreState()
```

> ⚠️ **Clip scope = Layer 1 ONLY。** In HTML/CSS covers, `overflow: hidden` must ONLY be on the Layer 1 background container. Layer 2 (lines) and Layer 3 (text) containers must NOT have `overflow: hidden`.
> For ReportLab body pages: `saveState()`/`restoreState()` must close immediately after Layer 1 background rendering.
> Layer 2 (lines) and Layer 3 (text) must never be rendered within a clip scope, otherwise text will be clipped.

### 🔴 Anti-Clip Bug (Layer 3 Text Truncation Fix)

**Symptom:** Cover text is visible but truncated by an invisible boundary, only half visible.

**Root cause:** clip/overflow scope not closed in time, causing subsequently rendered text to be clipped by the same clip rect.

**Iron rule (HTML/CSS - canonical cover implementation):**
```html
<!-- ✅ CORRECT - overflow:hidden only on Layer 1 -->
<div class="cover-layer-1" style="position:absolute; inset:0; overflow:hidden; z-index:1;">
  <!-- Background decorative elements -->
</div>
<div class="cover-layer-2" style="position:absolute; inset:0; z-index:2;">
  <!-- Structure lines, no overflow:hidden -->
</div>
<div class="cover-layer-3" style="position:absolute; inset:0; z-index:3;">
  <!-- Text content, no overflow:hidden -->
</div>

<!-- ❌ WRONG - global overflow:hidden clips text -->
<div class="cover-container" style="overflow:hidden;">
  <div class="layer-1">...</div>
  <div class="layer-2">...</div>
  <div class="layer-3">...</div>  <!-- Text gets clipped! -->
</div>
```

**Iron rule (ReportLab - body page background element reference):**
```python
# ✅ CORRECT - clip only wraps Layer 1
canvas.saveState()
canvas.clipPath(page_clip, stroke=0)
render_layer_1_background(canvas)   # Background decoration
canvas.restoreState()                # ← Must close here immediately!

render_layer_2_lines(canvas)         # Structure lines - not inside clip
render_layer_3_text(canvas)          # Text content - not inside clip

# ❌ WRONG - text clipped by clip scope
canvas.saveState()
canvas.clipPath(page_clip, stroke=0)
render_layer_1_background(canvas)
render_layer_2_lines(canvas)         # ← Gets clipped
render_layer_3_text(canvas)          # ← Gets clipped! Text only half visible
canvas.restoreState()
```

```css
/* Creative (HTML): Same principle */
.cover-bg-layer  { overflow: hidden; z-index: 1; }  /* clip only on background layer */
.cover-line-layer { overflow: visible; z-index: 2; } /* lines not clipped */
.cover-text-layer { overflow: visible; z-index: 3; } /* text not clipped */
```

### 🔴 No Page Border/Frame

**Symptom:** A rectangular border appears around the entire cover page, looking like a table.

**Root cause:** ReportLab's `Frame()` defaults to `showBoundary=0`, but if set to `1` or `True`, it shows a border. Also `canvas.rect()` may accidentally draw a full-page rectangle.

**Iron rule:**
```python
# Cover page Frame must have showBoundary=0
Frame(x, y, w, h, showBoundary=0)  # Always 0

# Never draw a full-page border on the cover
canvas.rect(0, 0, W, H)  # ❌ BANNED on cover page

# If using doc.showBoundary, cover page must be skipped
doc = SimpleDocTemplate(..., showBoundary=0)  # Always 0 in production
```

```css
/* Creative: No outer border on covers */
.cover-page {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
```

### 🔴 Minimum Spacing Between Decorative Lines and Text (Line-to-Text Spacing)

**Symptom:** Decorative lines on the cover (Layer 2 dividers, corner marks, sidebar edges) are flush against or overlapping with text.

**Iron rule:**
```
Minimum spacing between decorative lines and any text content = U (= W * 0.05)
i.e., at least 1 U of whitespace between line edges and text edges
```

| Line Type | Minimum Spacing | Notes |
|---------|---------|------|
| Horizontal divider | `U` above and below the line | Line must not be flush against title or body text |
| Vertical sidebar edge | `U` to the right of the line | Text inside sidebar must maintain spacing from the edge |
| Corner marks / crop marks | `1.5 * U` from mark endpoint to nearest text | Marks must not touch text |
| Ultra-thick anchor line (Template 01) | `2 * U` to the right of the line | Thick line and title need ample breathing room |

```python
# Example: Template 01 vertical thick line to title spacing
thick_line_x = 0.10 * W
title_x = thick_line_x + 2 * U  # 2U spacing to the right of the thick line
# ❌ WRONG: title_x = thick_line_x + 5  # 5pt is too close, will overlap
```

---

# PART 1: SEVEN COVER TEMPLATES (7 Cover Rendering Specifications)

Coordinates use `W` (page width) and `H` (page height). `U = W * 0.05` (base spacing unit).

All templates inherit the A0.0-A0.3 architecture rules above.

**Each template now includes a mandatory Summary/Description drawer** to fill visual space.

---

## Template 01: HUD Data Terminal - Ultra-Thick Vertical Anchor Line

**Design intent:** A single bold vertical line on the left anchors all visual weight. The thick line eliminates left-side floating. Clean, data-driven, authoritative.

### Layer 1 - Background
- Full-page grid pattern: horizontal + vertical lines at ~50pt intervals
- Grid color: primary color at **2% opacity** (white background, nearly invisible)
- Grid line width: `0.5pt`

### Layer 2 - Structure
- **Left anchor line:** Start `(0.12*W, 0.1*H)`, End `(0.12*W, 0.9*H)`. Line width = **6pt**, primary color.
- **Meta separator line:** At `Y_meta - 10pt`, from `X_content` to `X_content + W*0.4`, line width = **1pt**, primary color at 40% opacity.

### Layer 3 - Content

**Content left edge: `X_content = 0.12*W + 30pt`** (offset from the thick line)

| Drawer | Y-Anchor | Content | Constraints |
|--------|----------|---------|-------------|
| **A - Kicker** | `0.15 * H` | Report type / subtitle (e.g. "2025 Annual Report Summary") | 16pt, Regular, letter-spacing 3pt, opacity 60%, uppercase |
| **B - Hero Title** | `0.30 * H` | **Company/entity name** (e.g. "GREENTECH") | 45-65pt (CJK: 50-80pt), Heavy. Company name is the visual center |
| **C - Summary** 🆕 | `0.50 * H` | 2-3 lines descriptive text about the report | 16-18pt, Regular, line-height 1.6, opacity 85%. **Width limit: `W * 0.6`**, auto-wrap. This drawer fills the mid-page void |
| **D - Meta/Date** | `0.75 * H` | Author, org, date | 16-20pt, Regular. Top edge separated by the 1pt meta line |

### Best For
Technology reports, data analysis, dashboard summaries, technical white papers

---

## Template 02: Corporate Editorial - Top Bar with Bottom Accent

**Design intent:** Top-bottom symmetry. Top bar provides structural weight, bottom-right info block creates diagonal balance. Solves the "empty edges" problem.

### Layer 1 - Background
- **Background giant year watermark:** Text = current year (e.g. "2026"), **Max font size = 180pt**, measure rendered width - if it exceeds `W * 0.85`, scale down proportionally. Position: `X = W - 20pt` (right edge), `Y = 0.15*H`. Color = primary at **4% opacity**. Font weight = Black. ⚠️ **Full-display iron rule: watermark text must be 100% within the visible page area - cropping is strictly forbidden. Prefer reducing font size over truncation.**

### Layer 2 - Structure
- **Top bar (skyline):** Rectangle at `(0, 0)`, width = `W`, height = **15pt**, primary color fill. Edge-to-edge.
- **Right info accent line (edge seal):** Vertical line at `X = 0.88*W`, from `Y = 0.75*H` to `Y = 0.88*H`. Line width = **4pt**, primary color.

### Layer 3 - Content

| Drawer | Position | Content | Constraints |
|--------|----------|---------|-------------|
| **Left upper - Title group** | `X = 0.12*W`, `Y = 0.15*H` | Kicker (report type/subtitle, 16pt) → Hero Title (company/entity name, 45-65pt / CJK 50-80pt, Heavy) | Stack downward from anchor |
| **Mid-left - Summary** 🆕 | `X = 0.12*W`, `Y = 0.50*H` | Descriptive paragraph | 16-18pt, Regular, line-height 1.6. **Width limit: `W * 0.5`** |
| **Right lower - Meta** | Right-aligned at `X = 0.88*W - 20pt`, `Y = 0.70*H` | Date, version, author | **Right-aligned**, 16-20pt. Must hug the 4pt accent line |

### Best For
Annual reports, financial summaries, investor documents, corporate governance reports

---

## Template 03: The Monolith - Hard-Left Alignment + Right-Side Giant Watermark Counterweight

**Design intent:** Everything hard-left. Right-side watermark counterbalances the asymmetry. Solves the "right half is empty" bug.

### Layer 1 - Background
- **Right-side vertical watermark (load-bearing wall):** Extract a short English word (e.g. "REPORT"). **Auto-scaling font size:** `Max_Font_Size = 180pt`, measure total height after rotation - if it exceeds `H * 0.85`, scale down proportionally. **Rotate 90° clockwise** (or use vertical text mode). Anchor at `X = 0.85*W`, vertically centered: `Y = (H - rendered_text_width) / 2`. Color = primary at **3% opacity**. ⚠️ **Full-display iron rule: watermark text must be 100% within the visible page area - cropping is strictly forbidden. Prefer reducing font size over truncation.**

### Layer 2 - Structure
- **Color dash (visual guide line):** At `(0.12*W, 0.15*H)`, draw a horizontal bar: width = **50pt**, height = **5pt**, primary color.
- **Meta accent line:** At `(0.12*W, Y_meta)`, vertical line: height = meta text block height, width = **2pt**, primary color at 50% opacity.

### Layer 3 - Content

**Unified left edge: `X = 0.12*W`**

| Drawer | Y-Anchor | Content | Constraints |
|--------|----------|---------|-------------|
| **A - Color dash** | `0.15 * H` | Structure element (not text) | 50pt × 5pt bar |
| **B - Kicker** | `0.20 * H` | Report type / subtitle | 16pt, Regular, letter-spacing 3pt, uppercase, opacity 60% |
| **C - Hero Title** | `0.28 * H` | **Company/entity name** | 45-65pt (CJK: 50-80pt), Heavy |
| **D - Summary** 🆕 | `0.45 * H` | Descriptive paragraph (key anti-void element) | 16-18pt, Regular, line-height 1.6. **Width limit: `W * 0.55`** (must not collide with right watermark) |
| **E - Meta** | `0.70 * H` | Author, org, version | 20pt, Regular, line-height 2.0. Left of the 2pt accent line |
| **F - Footer** | `0.90 * H` | Date + doc number, right-aligned at `X = 0.88*W` | 16pt, Regular, opacity 60% |

### Best For
White papers, project proposals, government documents, technical standards

---

## Template 04: Museum Minimal - Refined Corner Crop Marks

**Design intent:** Abandon all-over scattered layout. Four corner crop marks form an invisible "force field box" that concentrates all content dead center.

### Layer 2 - Structure
- Set safety margin `M = 0.08 * W`
- **Four corner marks** at inner corners: `(M, M)`, `(W-M, M)`, `(M, H-M)`, `(W-M, H-M)`
- Each mark: L-shaped, arm length = **30pt**, line width = **2pt**, primary color at 60% opacity
- Marks point **inward** (top-left: right arm + down arm)

### Layer 3 - Content

**This template FORBIDS hardcoded absolute Y coordinates.**

**Centering algorithm (mandatory):**
1. Pre-compose ALL text elements (kicker + title + summary + meta) into a single virtual Text Block
2. Calculate the block's total rendered height `Block_H`
3. Position block: `X = 0` (full width, center-aligned text), `Y = (H - Block_H) / 2`
4. **This guarantees the content group is vertically centered regardless of how much content there is**

**Internal spacing within the centered block:**
- Kicker → Title: `24pt`
- Title → Summary: `20pt`
- Summary → Meta: `40pt`

### Type Scale
| Role | Size | Notes |
|------|------|-------|
| Kicker | 16pt | Uppercase, letter-spacing 4pt, opacity 50%. Bound to report type/subtitle |
| Hero Title | 48-60pt | Heavy - slightly smaller than other templates to fit center composition. **Bound to company/entity name** |
| Summary 🆕 | 16-18pt | Regular, line-height 1.6, center-aligned, width ≤ `W * 0.6` |
| Meta | 16pt | Regular, opacity 60%, at bottom of group |

### Best For
Gallery catalogs, design portfolios, exhibition materials, luxury brand documents

---

## Template 05: Floating Diagonal - Premium Whitespace with Binding Line

**Design intent:** "Left-upper to right-lower" diagonal visual flow. The two text groups create tension across whitespace. The gap IS the design.

### Layer 2 - Structure
- **Binding dashed line:** At `X = 0.08*W`, from `Y = 0.05*H` to `Y = 0.95*H`. Line width = **1pt**, dashed (dash 6pt, gap 8pt), color = light gray (#d0d0d0, 40% opacity).

### Layer 3 - Content

| Group | Position | Content | Constraints |
|-------|----------|---------|-------------|
| **Upper-left group** | Anchor: `X = 0.15*W`, `Y = 0.20*H` | Kicker (report type/subtitle, 16pt gap) → Hero Title (company/entity name, 45-65pt / CJK 50-80pt, Heavy) | Left-aligned. Width limit: `W * 0.7` |
| **Lower-right group** 🆕 | Anchor: `X = 0.45*W`, `Y = 0.60*H` | Summary + Meta + Footer | **Left-aligned** (NOT right-aligned - intentional asymmetry). A **3pt vertical accent line** of height = group text height is drawn at `X = 0.45*W - 12pt` as a visual anchor. Line-height 2.0 for meta, 24pt gap before footer |

**Visual effect:** Upper-left and lower-right groups are "pulled apart" across the diagonal. The empty top-right and bottom-left create tension, not emptiness.

### Best For
Creative reports, editorial layouts, art direction documents, brand guidelines

---

## Template 06: Swiss Grid - Ultimate Precision, Curing All Misalignment

**Design intent:** The typographic "multiplication table." Thick lines physically slice the page into cells. Content fills its assigned cell. Impossible to misalign.

### Layer 2 - Structure (ABSOLUTE - non-negotiable)

```
Horizontal line 1: (0.1*W, 0.25*H) → (0.9*W, 0.25*H), width 2pt, primary
Horizontal line 2: (0.1*W, 0.75*H) → (0.9*W, 0.75*H), width 2pt, primary
Vertical line 1:   (0.45*W, 0.25*H) → (0.45*W, 0.75*H), width 2pt, primary
```

These create 4 zones:

```
┌──────────────────────────────────────┐
│         Zone A - Top Strip           │  ← Kicker / report type
├──────────────────┬───────────────────┤
│   Zone B         │   Zone C          │
│   (left cell)    │   (right cell)    │  ← B: Hero Title (MUST fill)
│   X: 0.1W-0.43W │   X: 0.48W-0.9W  │  ← C: Summary text (MUST fill)
├──────────────────┴───────────────────┤
│         Zone D - Bottom Strip        │  ← Footer / year / doc number
└──────────────────────────────────────┘
```

### Layer 3 - Content (STRICT zone containment)

| Zone | Content | X Range | Y Range | Notes |
|------|---------|---------|---------|-------|
| **A** | Kicker / report type | `0.10*W` - `0.90*W` | `0.15*H` - `0.23*H` | Left-aligned at `X = 0.12*W` |
| **B** | **Hero Title (company/entity name)** | `0.10*W` - `0.43*W` | `0.28*H` - `0.70*H` | **Width = `0.33*W`**. Font must be large enough to physically fill the cell. Text wraps at boundary. |
| **C** | **Summary text** 🆕 | `0.48*W` - `0.90*W` | `0.28*H` - `0.70*H` | **Must contain substantial descriptive text** - this zone MUST be filled. 16-18pt, Regular, line-height 1.6. This is the primary anti-empty-page mechanism. |
| **D** | Footer / date / number | `0.10*W` - `0.90*W` | `0.78*H` - `0.88*H` | Can split: left part + right-aligned part |

### Zone Overflow Protection (MANDATORY)

If text in Zone B or C exceeds the vertical boundary (Y > `0.70*H`):
1. **Step 1:** Reduce font size by 2pt increments (minimum: 16pt for summary, 40pt for title)
2. **Step 2:** If still overflows, truncate with `...` ellipsis
3. **NEVER** let text cross a grid line - the grid is sacred

**Hard width enforcement:**
```python
# Zone B: title MUST wrap within its cell width
zone_b_max_width = 0.33 * W
# If title renders wider → word-wrap, NEVER let it bleed into Zone C

# Zone C: summary MUST wrap within its cell width
zone_c_max_width = 0.42 * W  # (0.90 - 0.48) * W
# Wrap at boundary, add lines, NEVER cross the vertical grid line
```

### Best For
Swiss-style design, data-heavy reports, structured corporate documents, annual reports

---

## Template 07: Solid Sidebar - Massive Pillar Anchoring the Page

**Design intent:** A massive solid-color sidebar provides gravitas. The right side can be loosely arranged - the pillar holds everything together.

### Layer 1 - Background
- **Left sidebar block (giant sidebar pillar):** Rectangle at `(0, 0)`, width = **`0.1*W`** (~80pt on A4), height = `H`. Primary color fill.
- **Sidebar watermark:** Inside the sidebar, render a short word (doc type or year) rotated **-90°**, white at **15% opacity**, vertically centered within the sidebar. **Auto-scaling font size:** `Max_Font_Size = H * 0.5`, measure total height after rotation - if it exceeds `H * 0.85`, scale down proportionally. ⚠️ **Full-display iron rule: watermark text must be 100% within the visible page area - cropping is strictly forbidden.**

### Layer 2 - Structure
- **Bottom horizontal line:** At `Y = 0.90*H`, from `X = Left_Edge` to `X = 0.90*W`. Line width = **1pt**, primary color at 30% opacity.

### Layer 3 - Content

**Safety boundary: `Left_Edge = 0.1*W + 40pt`** - ALL text must start at or right of this line. Zero tolerance for collision with sidebar.

**Layout uses relative vertical centering:**
1. Compose full text group: Kicker + Hero Title + Summary + Meta
2. Calculate total group height
3. Position group at `X = Left_Edge`, `Y = (H - group_height) / 2` (vertically centered)

| Element | Notes |
|---------|-------|
| Kicker | 16pt, Regular, uppercase, letter-spacing 3pt, opacity 60%. Bound to report type/subtitle |
| Hero Title | 45-65pt, Heavy. **Bound to company/entity name** |
| Summary 🆕 | 16-18pt, Regular, line-height 1.6. Width ≤ `0.90*W - Left_Edge` |
| Meta | 20pt, Regular, line-height 1.8 |

**Footer (separate from centered group):**
- On/just above the bottom horizontal line at `Y = 0.90*H - 10pt`
- Left-aligned date at `X = Left_Edge`, right-aligned org name at `X = 0.90*W`
- 16pt, Regular, opacity 60%

### Best For
Government/institutional reports, legal documents, formal project deliverables, bidding documents

---

# PART 2: TEMPLATE SELECTION GUIDE

Template selection uses a two-dimensional matrix: **Intent** (from `visual_framework.md` 5-intent system) × **Document Type**. This replaces the old "Document Tone" classification and aligns with the Intent Mapping Table in `creative.md`.

| Intent | Document Type | Recommended Templates | Default |
|--------|---------------|----------------------|---------|
| **Calm** | Healthcare / Wellness / Minimalist | 04 Museum, 01 HUD | **04** |
| **Calm** | Academic / Research | 06 Swiss Grid, 03 Monolith | **06** |
| **Tension** | Crisis / Alert / Disruption | 01 HUD, 05 Diagonal | **01** |
| **Energy** | Marketing / Creative / Design | 05 Diagonal, 06 Swiss Grid | **05** |
| **Energy** | Technology / Data | 01 HUD, 06 Swiss Grid | **01** |
| **Authority** | Formal / Corporate / Financial | 02 Corporate, 03 Monolith | **03** |
| **Authority** | Government / Bidding | 07 Sidebar, 03 Monolith, **11 Institutional** | **07** |
| **Authority** | Thesis proposal / Dissertation cover | **11 Institutional** | **11** |
| **Authority** | Luxury / Editorial | 03 Monolith, 05 Diagonal | **03** |
| **Warmth** | Food / Lifestyle / Home | 04 Museum, 05 Diagonal | **04** |

> **Legacy mapping:** "Formal/Corporate" tone → Authority intent, "Minimalist" tone → Calm intent, "Luxurious/Editorial" tone → Authority intent.

**⚠️ No Global Default.** When no specific style is explicitly requested, the LLM MUST analyze the document's content, tone, and audience to autonomously select the most fitting template. Cross-reference the Intent (derived from content via `design_engine.py derive` or manual judgment) with the Document Type to find the best match. Every cover selection must be a deliberate design decision.

---

# PART 3: CODE-LEVEL SAFETY MEASURES (Pre-Render Safeguards)

These checks run BEFORE final rendering. They are **mandatory** - not optional optimizations.

---

## S3.0 - Cover Overlap Validation (MANDATORY)

**After generating the cover HTML and before rendering to PDF, run:**

```bash
node "$PDF_SKILL_DIR/scripts/cover_validate.js" cover.html
```

This detects text-vs-decorative-line overlaps by rendering the HTML and measuring actual bounding boxes. Exit code 1 = overlap found, must fix before generating the PDF.

The minimum gap between any text element and any decorative line is **1U (= 5% of page width ≈ 40px on A4)**. This catches the exact bug shown in the "text overlapping decorative lines" screenshots.

**If the check fails:**
1. Adjust the decorative line's Y position to maintain ≥ 1U gap from the nearest text
2. Or adjust the text block's position/size to avoid the overlap
3. Re-run `cover_validate.js` until it passes (exit code 0)

---

## S3.1 - Hero Title Overflow Protection (Title Line-Wrapping)

**Rule:** Hero title must NEVER exceed its template's width boundary.

**Algorithm:**
1. Measure the rendered width of the hero title string at the target font size
2. If `rendered_width > max_width`:
   - Word-wrap at boundary (CJK: any character; Latin: space/hyphen; Mixed: CJK/Latin boundaries)
   - Multi-line hero titles: **lock line-height to 1.15**
3. Maximum lines: **3** - if title needs 4+ lines, reduce font size by 4pt increments until ≤ 3 lines
4. Minimum font size floor: **40pt** (below this, truncate with `...`)

```python
def safe_hero_title(text, font, max_size, max_width, min_size=40):
    size = max_size
    while size >= min_size:
        lines = word_wrap(text, font, size, max_width)
        if len(lines) <= 3:
            return lines, size
        size -= 4
    return truncate_with_ellipsis(text, font, min_size, max_width, max_lines=3), min_size
```

---

## S3.2 - Zone Collision Detection

After rendering each text block, check if its bottom edge penetrates the next zone boundary:

1. **Step 1 - Font reduction:** Decrease by 2pt. Floor: 16pt for meta/summary, 40pt for titles.
2. **Step 2 - Truncation:** If font reduction fails, truncate with `...`
3. **Step 3 - Log warning:** Output a warning about content truncation

```python
def enforce_zone_bounds(text, font, size, zone_y_max, min_size=16):
    while size >= min_size:
        rendered_height = measure_text_height(text, font, size)
        if current_y + rendered_height <= zone_y_max:
            return text, size
        size -= 2
    return truncate_to_fit(text, font, min_size, zone_y_max - current_y), min_size
```

---

## S3.3 - Uppercase Lock

**The following text roles MUST be force-uppercased when content is English/Latin:**

- Kicker (category label / lead-in text)
- Footer (closing date / document number)
- Background watermark text
- Any Layer 1 decorative text

```python
kicker_text = kicker_text.upper() if is_latin(kicker_text) else kicker_text
footer_text = footer_text.upper() if is_latin(footer_text) else footer_text
watermark_text = watermark_text.upper()  # Always uppercase
```

**Exception:** CJK text is exempt. Mixed CJK+Latin strings: uppercase only the Latin portions.

---

## S3.4 - Hard Width Boundary Enforcement 🆕

**Every drawer/zone has a maximum width. Text wrapping MUST respect this width exactly.**

```python
# WRONG - text bleeds past boundary
draw_text(x=0.12*W, text=long_title, width=None)  # width unconstrained!

# RIGHT - hard clamp
max_width = 0.6 * W  # or zone-specific value
wrapped_lines = word_wrap(text, font, size, max_width)
for i, line in enumerate(wrapped_lines):
    draw_text(x=x_anchor, y=y_anchor + i * line_height, text=line)
```

**Rule:** It is acceptable for text to add extra lines (grow vertically). It is NEVER acceptable for text to exceed its horizontal boundary (grow horizontally). Vertical overflow triggers S3.2; horizontal overflow is a critical bug.

---

## S3.5 - Mandatory Summary Auto-Generation 🆕

**If the user provides only a title and no description/summary, the system MUST generate placeholder text.**

```python
if not summary_text or summary_text.strip() == "":
    if lang == "zh":
        summary_text = f"本报告由{org_name or '系统'}自动生成,包含了综合数据分析与洞察结论。"
    else:
        summary_text = f"This report presents comprehensive analysis and key insights prepared by {org_name or 'the organization'}."
```

**Why:** A title-only cover looks barren. The Summary drawer physically occupies 2-4 lines, filling mid-page void and making the cover look intentionally designed rather than half-finished.

---

## S3.6 - Background Watermark Full-Display Enforcement 🆕

**All watermark text in the background layer (Layer 1) must be 100% within the visible page area. Cropping, truncation, or extending beyond page boundaries is strictly forbidden.**

**Applicable scope:**
- Template 02 giant year watermark
- Template 03 right-side vertical watermark
- Template 07 sidebar watermark
- `cover-backgrounds.md` Recipe 2.1 giant sidebar pillar
- `cover-backgrounds.md` Recipe 2.2 bottom full-size text
- Any other decorative text in the background layer

**Adaptive algorithm (mandatory):**

```python
def safe_watermark_size(text, font, max_size, available_space):
    """
    Ensure watermark text is fully displayed within available space.
    available_space: available width/height (depending on text direction)
    """
    rendered = measure_text(text, font, max_size)
    if rendered > available_space:
        return max_size * (available_space / rendered)
    return max_size
```

**Rules:**
1. Horizontal text: rendered width must not exceed `W * 0.90` (5% safety margin on each side)
2. Vertical/rotated text: rendered height must not exceed `H * 0.85` (7.5% safety margin top and bottom)
3. If exceeded, scale down font size proportionally - never truncate
4. **Anchor coordinates must never exceed page boundaries** (no negative X/Y values or values exceeding W/H)

**This is a visual quality red line: a truncated "REPO" is worse than no watermark at all. A complete "REPORT" is the design.**

---

## S3.7 - Line-Length Alignment (Line Must Match Text Span)

**Problem:** Decorative lines (vertical accent lines, horizontal dividers, underlines) are arbitrary lengths that don't relate to the text they accompany, creating visual disconnect.

**Iron Rule:** Lines must be sized relative to the text they serve:

### Vertical Lines (e.g., Template 01 thick line, Template 08 accent line)

**Vertical line height = text block height** (from first text element to last text element in the same column).

```
# WRONG - arbitrary fixed height
vline_top = 0.1 * H
vline_bottom = 0.9 * H  # ← line runs full page regardless of content

# RIGHT - measure text block, then draw line
text_top = first_element_y        # e.g., label at 0.12*H
text_bottom = last_element_y + last_element_height  # e.g., footer at 0.88*H
vline_top = text_top - U           # 1U padding above first element
vline_bottom = text_bottom + U     # 1U padding below last element
```

### Horizontal Lines (e.g., Template 08 hline, dividers)

**Horizontal line width ≥ text width of the widest text element in its zone.** Lines may be slightly longer (up to 120% of text width) but NEVER shorter.

```python
# WRONG - fixed short line
hline_width = 200  # ← might be shorter than the title

# RIGHT - measure, then draw
max_text_width = max(measure(title), measure(subtitle), measure(authors))
hline_width = max(max_text_width, max_text_width * 1.1)  # at least as wide, up to 110%
# Clamp to available space
hline_width = min(hline_width, available_width)
```

### HTML/CSS Implementation

For HTML/Playwright covers, use relative sizing:
```css
/* Vertical line spans the content block */
.vline {
  position: absolute;
  top: var(--content-top);     /* align with first text element */
  bottom: var(--content-bottom); /* align with last text element */
}

/* Horizontal divider: min-width matches text container */
.hline {
  width: max(100%, 200px);     /* at least as wide as parent text container */
}
```

**Checklist:**
- [ ] Every vertical line's height matches its adjacent text block span (± 1U padding)
- [ ] Every horizontal line's width ≥ widest text element in its zone
- [ ] No decorative line is shorter than the text it accompanies

---

## S3.8 - Vertical Balance (Anti-Top-Heavy Layout)

**Problem:** Content clusters at the top of the page, leaving the bottom 40%+ as dead whitespace. This happens when anchor points are set too high and don't adapt to content volume.

**Root cause:** Fixed anchor grid with `ANCHOR_TITLE_Y = 0.20*H` pushes everything upward regardless of how much content there is.

### Solution: Adaptive Vertical Centering

**When total content height < 50% of page height, switch to centered distribution mode:**

```python
# Step 1: Calculate total content height
content_elements = [title, subtitle, summary, meta, footer]
total_content_h = sum(elem.height for elem in content_elements) + total_gaps

# Step 2: Check fill ratio
fill_ratio = total_content_h / (H * 0.80)  # usable height (excluding margins)

if fill_ratio < 0.50:
    # LOW CONTENT MODE - vertically center the entire block
    start_y = (H - total_content_h) / 2
    # Distribute elements from start_y downward with standard gaps
else:
    # NORMAL MODE - use anchor grid
    # But shift anchors down: title at H*0.30-0.35 (not 0.20-0.25)
    pass
```

### Anchor Adjustment Rules

| Content Volume | Title Anchor | Summary Anchor | Meta Anchor |
|---------------|-------------|----------------|-------------|
| **Sparse** (fill < 50%) | Centered mode | Centered mode | Centered mode |
| **Normal** (fill 50-80%) | `H * 0.30` | `H * 0.48` | `H * 0.70` |
| **Dense** (fill > 80%) | `H * 0.20` | `H * 0.40` | `H * 0.65` |

### CJK Title Size Compensation

CJK characters at the same pt size as Latin characters appear visually smaller due to denser stroke structure. Compensate:

```
CJK Hero Title:    50-80pt (Latin: 45-65pt) - increase by 15-20%
CJK Kicker:        11-12pt (Latin: 9pt)
CJK Summary:       17-20pt (Latin: 16-18pt)
```

**Detection:** If title string contains CJK characters (`\u4e00-\u9fff`), apply CJK size multiplier.

### HTML/CSS Implementation

```css
/* Vertical centering mode for sparse content */
.cover.sparse-content .center-block {
  justify-content: center;  /* flexbox vertical center */
}

/* CJK title size bump */
.title:lang(zh), .title:lang(ja), .title:lang(ko) {
  font-size: clamp(50pt, 8vw, 80pt);  /* larger than Latin range */
}
```

**Checklist:**
- [ ] No cover has >40% dead whitespace at the bottom
- [ ] Content is visually centered on the page (optical center, not mathematical)
- [ ] CJK titles are 15-20% larger than equivalent Latin titles
- [ ] Sparse-content covers use centered distribution, not fixed anchors

---

## S3.9 - Percentage Positioning Requires Known-Size Container

**Root cause of bad case:** A wrapper div (e.g. `.content-left`) with `position: absolute` but **no explicit height** contains children positioned with `top: XX%`. CSS percentage `top` resolves against the containing block's **height** — if that height is zero or undefined (because all children are also absolutely positioned, contributing no content height), the percentage values collapse and elements stack on top of each other.

**Iron rule:** When using `top: XX%` (or `bottom: XX%`) to position child elements, the containing block MUST have a **deterministic height** — one of:

| Method | Example | When to use |
|--------|---------|-------------|
| Explicit `height` | `height: 100%` or `height: var(--h)` | Wrapper spans full page |
| `top` + `bottom` pair | `top: 0; bottom: 0;` | Wrapper stretches between two edges |
| `inset: 0` | `inset: 0;` | Shorthand for full-page wrapper |

**Preferred pattern — flat structure with px values (safest):**
```css
/* ✅ CORRECT: children positioned directly in .cover with px values */
.cover { position: relative; width: 794px; height: 1123px; }
.kicker   { position: absolute; top: 225px;  left: 95px; }
.title    { position: absolute; top: 292px;  left: 95px; }
.summary  { position: absolute; top: 539px;  left: 95px; }
.meta     { position: absolute; top: 786px;  left: 95px; }
```

**Acceptable — wrapper with deterministic height:**
```css
/* ✅ OK: wrapper has inset:0, so height = parent height = 1123px */
.content-left { position: absolute; inset: 0; width: 55%; }
.title   { position: absolute; top: 26%; }  /* 26% of 1123px = 292px ✓ */
```

**Forbidden — wrapper with no height:**
```css
/* ❌ BANNED: .content-left has no height/bottom, percentage top is undefined */
.content-left { position: absolute; left: 12%; top: 0; width: 55%; }
.title   { position: absolute; top: 26%; }  /* 26% of WHAT? → collapse → overlap */
.summary { position: absolute; top: 48%; }  /* stacks on top of title */
```

**Quick self-check before writing cover CSS:**
1. For every element with `top: XX%` — trace upward: does the containing block have a known height?
2. If unsure → use `px` values instead (calculate from `var(--h)` manually: `26% × 1123 = 292px`)
3. If using a grouping wrapper → give it `inset: 0` or explicit `height: 100%`

---

# PART 4: COVER COLOR RULES

> Cover colors must be consistent with the body color system - they cannot exist independently.

```
Cover primary    = Body theme color
Cover secondary  = Primary lightness variant (±20% lightness)
Cover background = Pure white / very light gray / primary at 5-8% opacity
```

### Absolutely Forbidden

- ❌ Dark large-area solid backgrounds (dark blue, dark green, black filling the page)
- ❌ Gradient backgrounds (any `linear-gradient` / `radial-gradient` as large-area fill)
- ❌ High-saturation color schemes
- ❌ Rainbow / multi-color gradients
- ❌ Dense textures or patterns
- ❌ Piling on decorative elements - restraint > clutter
- ❌ More than 2 typefaces on a cover
- ❌ Centered text + gradient/solid background (PowerPoint aesthetic)

### Safe Cover Color Schemes (Reference Only)

> ⚠️ These are **examples for reference**. In normal workflow, run `palette.cascade --title "<title>" --mode minimal --format css` to generate the actual cover colors. Do NOT copy these hex values directly.

| Name | Primary | Secondary | Background | Use Case |
|------|---------|-----------|------------|----------|
| Ink Stone | `#1a1a2e` | `#4a4a5e` | `#fafafa` | Business, formal |
| Indigo | `#1e3a5f` | `#2d5f8a` | `#f5f8fb` | Technology, reports |
| Warm Chestnut | `#5c3d2e` | `#8a6b5a` | `#faf6f3` | Culture, branding |
| Moss Green | `#2d4a3e` | `#4a7a6a` | `#f5f8f6` | Nature, health |
| Deep Crimson | `#6b2d3e` | `#9a4a5e` | `#faf5f6` | Traditional, elegant |

---

# PART 4.5: ACADEMIC COVER TEMPLATES (Templates 08-10)

> **Academic covers are exempt from PART 4 color rules.** Academic papers, theses, and research reports traditionally use dark backgrounds with light text - this is the established scholarly visual language. Templates 08-10 follow LaTeX title page conventions translated to HTML/CSS.

**All 3 templates share these rules:**
- Page size: `width: 794px; height: 1123px` (A4 at 96dpi)
- Full-bleed dark background (edge-to-edge, no margins)
- Serif font for titles (Playfair Display / Noto Serif SC), sans-serif for metadata
- Generated as HTML → Playwright `page.pdf()` → pypdf merge (same pipeline as Report covers)
- `<link>` tag for Google Fonts (NOT `@import`)

**Content slots (all templates):**
| Slot | Required | Example |
|------|----------|---------|
| `label` | Optional | `RESEARCH PAPER`, `博士论文` |
| `title` | **Required** | Paper title (auto-wrap, max 3 lines) |
| `subtitle` | Optional | Subtitle or abstract excerpt |
| `authors` | **Required** | Author name(s) |
| `institution` | Optional | University / lab / affiliation |
| `keywords` | Optional | Keyword list |
| `footer_left` | Optional | Journal name, DOI |
| `footer_right` | Optional | Date, version |

---

## Template 08: Academic Vertical Anchor - Dark bg + Left vertical line + Left-aligned

**Design intent:** Emulates the classic arXiv/preprint cover. A bold vertical accent line anchors the left edge, all text left-aligned with generous vertical rhythm. Serious, no-frills.

```
┌─────────────────────────────┐
│ ┃                           │
│ ┃  LABEL (9pt, accent)      │  ← y = H - 3.5cm
│ ┃                           │
│ ┃  Title (32pt Bold)        │  ← y = H - 6cm, line-height 42pt
│ ┃  Title line 2             │
│ ┃                           │
│ ┃  Subtitle (12pt)          │  ← y = H - 14cm
│ ┃                           │
│ ┃  Authors (12pt, white)    │  ← y = H - 18cm
│ ┃  Institution (10pt)       │
│ ┃                           │
│ ┃───────────────────      │  ← accent line y=3.5cm
│ ┃  Footer L       Footer R  │
└─────────────────────────────┘
┃ = vertical accent line at x=1.5cm, 2.5pt width
```

**Best for:** Research papers, technical reports, arXiv preprints

**HTML structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Noto+Serif+SC:wght@400;700;900&family=Inter:wght@300;400;500&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    @page { size: 794px 1123px; margin: 0; }
    :root {
      --c-bg: #162032;
      --c-accent: #8B7E5A;
      --c-text: #FFFFFF;
      --c-muted: #8898A8;
      --c-footer: #607080;
    }
    html, body { margin: 0; padding: 0; width: 794px; height: 1123px; background: var(--c-bg); color: var(--c-text); font-family: 'Inter', 'Noto Sans SC', sans-serif; }
    @media screen {
      html { height: auto; display: flex; justify-content: center; min-height: 100vh; background: var(--c-bg); }
      body { transform-origin: top center; scale: min(1, calc(100vw / 794), calc(100vh / 1123)); margin: 0 auto; box-shadow: 0 0 60px rgba(0,0,0,0.3); }
    }
    .cover { width: 794px; height: 1123px; position: relative; box-sizing: border-box; }
    .vline { position: absolute; left: 57px; top: 76px; bottom: 76px; width: 2.5px; background: var(--c-accent); }
    .hline { position: absolute; left: 83px; right: 76px; bottom: 132px; height: 0.5px; background: var(--c-accent); }
    .content { position: absolute; left: 83px; right: 76px; top: 0; bottom: 0; }
    .label { position: absolute; top: 132px; font-size: 9pt; color: var(--c-accent); letter-spacing: 3px; text-transform: uppercase; font-family: 'Inter', 'Noto Sans SC', sans-serif; }
    .title { position: absolute; top: 228px; font-size: 32pt; font-weight: 700; line-height: 1.3; font-family: 'Playfair Display', 'Noto Serif SC', serif; color: var(--c-text); max-width: 580px; }
    .subtitle { position: absolute; top: 530px; font-size: 12pt; line-height: 1.5; color: var(--c-muted); max-width: 500px; }
    .authors { position: absolute; top: 680px; font-size: 12pt; color: var(--c-text); }
    .institution { position: absolute; top: 740px; font-size: 10pt; color: var(--c-muted); line-height: 1.4; }
    .footer { position: absolute; bottom: 76px; left: 0; right: 0; display: flex; justify-content: space-between; font-size: 9pt; color: var(--c-footer); }
  </style>
</head>
<body>
  <div class="cover">
    <div class="vline"></div>
    <div class="hline"></div>
    <div class="content">
      <div class="label"><!-- LABEL --></div>
      <div class="title"><!-- TITLE --></div>
      <div class="subtitle"><!-- SUBTITLE --></div>
      <div class="authors"><!-- AUTHORS --></div>
      <div class="institution"><!-- INSTITUTION --></div>
      <div class="footer">
        <span><!-- FOOTER_LEFT --></span>
        <span><!-- FOOTER_RIGHT --></span>
      </div>
    </div>
  </div>
</body>
</html>
```

**Default palette (override via `palette.cascade --intent <intent> --mode dark --format css`):**
| Name | `--c-bg` | `--c-accent` | `--c-muted` | Use case |
|------|----------|------------|----------|----------|
| Deep Sea | `#162032` | `#8B7E5A` | `#8898A8` | General academic |
| Indigo | `#1e3a5f` | `#2d5f8a` | `#7A90A5` | Technical |
| Ink Stone | `#1a1a2e` | `#4a4a5e` | `#8080A0` | Formal occasions |

> ⚠️ These are **fallback defaults** when the palette system is unavailable. In normal workflow, run `palette.cascade` to generate mathematically harmonious colors and inject them into the `:root` variables.

---

## Template 09: Academic Symmetric - Dark bg + Top/bottom lines + Centered

**Design intent:** Emulates the classic IEEE/ACM Transactions title page. Perfect bilateral symmetry, thick horizontal rules frame the content zone. Formal and authoritative.

```
┌─────────────────────────────┐
│                             │
│   ══════════════════════    │  ← Top rule y=H-3cm, 2pt
│                             │
│        LABEL (centered)     │
│                             │
│     Title (28-30pt Bold)    │
│                             │
│        Subtitle             │
│                             │
│          ───                │  ← Thin divider 3cm, centered
│                             │
│        Authors              │
│       Institution           │
│                             │
│   ══════════════════════    │  ← Bottom rule y=3cm, 2pt
│      Journal · Date         │
└─────────────────────────────┘
```

**Best for:** Top journal submissions, IEEE/ACM papers, theses

**HTML structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Noto+Serif+SC:wght@400;700;900&family=Inter:wght@300;400;500&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    @page { size: 794px 1123px; margin: 0; }
    :root {
      --c-bg: #162032;
      --c-accent: #4A90C4;
      --c-text: #FFFFFF;
      --c-muted: #90A8C0;
    }
    html, body { margin: 0; padding: 0; width: 794px; height: 1123px; background: var(--c-bg); color: var(--c-text); font-family: 'Inter', 'Noto Sans SC', sans-serif; }
    @media screen {
      html { height: auto; display: flex; justify-content: center; min-height: 100vh; background: var(--c-bg); }
      body { transform-origin: top center; scale: min(1, calc(100vw / 794), calc(100vh / 1123)); margin: 0 auto; box-shadow: 0 0 60px rgba(0,0,0,0.3); }
    }
    .cover { width: 794px; height: 1123px; position: relative; display: flex; flex-direction: column; align-items: center; box-sizing: border-box; }
    .rule-top, .rule-bottom { position: absolute; left: 114px; right: 114px; height: 2px; background: var(--c-accent); }
    .rule-top { top: 114px; }
    .rule-bottom { bottom: 114px; }
    .center-block { position: absolute; top: 0; bottom: 0; left: 114px; right: 114px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
    .label { font-size: 9pt; color: var(--c-accent); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 40px; font-family: 'Inter', 'Noto Sans SC', sans-serif; }
    .title { font-size: 30pt; font-weight: 700; line-height: 1.3; font-family: 'Playfair Display', 'Noto Serif SC', serif; margin-bottom: 24px; max-width: 500px; }
    .subtitle { font-size: 14pt; color: var(--c-muted); margin-bottom: 40px; max-width: 450px; line-height: 1.5; }
    .divider { width: 114px; height: 0.5px; background: var(--c-accent); margin-bottom: 40px; }
    .authors { font-size: 12pt; margin-bottom: 12px; }
    .institution { font-size: 10pt; color: var(--c-muted); line-height: 1.4; }
    .footer { position: absolute; bottom: 57px; left: 114px; right: 114px; text-align: center; font-size: 9pt; color: var(--c-muted); }
  </style>
</head>
<body>
  <div class="cover">
    <div class="rule-top"></div>
    <div class="rule-bottom"></div>
    <div class="center-block">
      <div class="label"><!-- LABEL --></div>
      <div class="title"><!-- TITLE --></div>
      <div class="subtitle"><!-- SUBTITLE --></div>
      <div class="divider"></div>
      <div class="authors"><!-- AUTHORS --></div>
      <div class="institution"><!-- INSTITUTION --></div>
    </div>
    <div class="footer"><!-- FOOTER --></div>
  </div>
</body>
</html>
```

**Default palette (override via `palette.cascade --intent <intent> --mode dark --format css`):**
| Name | `--c-bg` | `--c-accent` | `--c-muted` | Use case |
|------|----------|------------|----------|----------|
| Midnight Blue | `#162032` | `#4A90C4` | `#90A8C0` | Math/theoretical |
| Ink Blue | `#0D1B2A` | `#3D5A80` | `#8898A8` | Formal reports |
| Deep Navy | `#0a1628` | `#5B8DB8` | `#7A9AB5` | Engineering |

> ⚠️ These are **fallback defaults**. In normal workflow, run `palette.cascade` to generate colors and inject into `:root`.

---

## Template 10: Academic Journal - Dark bg + Top/bottom lines + Centered + Keywords

**Design intent:** Extended version of Template 09, with dedicated keyword block. Matches the layout of top-tier Chinese journal submissions and thesis covers.

```
┌─────────────────────────────┐
│                             │
│   ══════════════════════    │  ← Top rule
│                             │
│        LABEL (centered)     │
│                             │
│        Title (34pt)         │
│                             │
│        Subtitle             │
│                             │
│          ───                │  ← Thin divider
│                             │
│        Keywords             │
│                             │
│   ══════════════════════    │  ← Bottom rule
│         Footer              │
└─────────────────────────────┘
```

**Best for:** Chinese journal submissions, theses with keywords, formal academic reports

**HTML structure:**
```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700;900&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
  <style>
    @page { size: 794px 1123px; margin: 0; }
    :root {
      --c-bg: #162032;
      --c-accent: #4A90C4;
      --c-text: #FFFFFF;
      --c-muted: #90A8C0;
    }
    html, body { margin: 0; padding: 0; width: 794px; height: 1123px; background: var(--c-bg); color: var(--c-text); font-family: 'Noto Sans SC', 'Inter', sans-serif; }
    .cover { width: 794px; height: 1123px; position: relative; display: flex; flex-direction: column; align-items: center; box-sizing: border-box; }
    .rule-top, .rule-bottom { position: absolute; left: 114px; right: 114px; height: 2px; background: var(--c-accent); }
    .rule-top { top: 114px; }
    .rule-bottom { bottom: 114px; }
    .center-block { position: absolute; top: 0; bottom: 0; left: 114px; right: 114px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
    .label { font-size: 9pt; color: var(--c-accent); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 40px; }
    .title { font-size: 34pt; font-weight: 700; line-height: 1.3; font-family: 'Noto Serif SC', serif; margin-bottom: 20px; max-width: 500px; }
    .subtitle { font-size: 14pt; color: var(--c-muted); margin-bottom: 40px; max-width: 450px; line-height: 1.5; }
    .divider { width: 152px; height: 0.5px; background: var(--c-accent); margin-bottom: 40px; }
    .keywords { font-size: 11pt; color: var(--c-muted); line-height: 1.8; max-width: 400px; }
    .footer { position: absolute; bottom: 57px; left: 114px; right: 114px; text-align: center; font-size: 9pt; color: var(--c-muted); }
  </style>
</head>
<body>
  <div class="cover">
    <div class="rule-top"></div>
    <div class="rule-bottom"></div>
    <div class="center-block">
      <div class="label"><!-- LABEL --></div>
      <div class="title"><!-- TITLE --></div>
      <div class="subtitle"><!-- SUBTITLE --></div>
      <div class="divider"></div>
      <div class="keywords">
        <!-- KEYWORD 1 --><br>
        <!-- KEYWORD 2 --><br>
        <!-- KEYWORD 3 -->
      </div>
    </div>
    <div class="footer"><!-- FOOTER --></div>
  </div>
</body>
</html>
```

**Recommended palettes:** Same as Template 09.

---

## Template 11: Institutional - White bg + Black border frame + Structured field slots

**Design intent:** The universal institutional cover. White background with a thick black border frame, all content centered, structured field slots with underline placeholders. Matches the style required by most universities worldwide for thesis proposals, dissertations, and formal institutional documents. Also suitable for government reports and official submissions. Zero decorative elements - the formality IS the design.

**⚠️ This template is exempt from PART 4 Academic Cover Color Rules (dark backgrounds).** It uses a white/light background by design, aligning with institutional formatting requirements.

```
┌─────────────────────────────────┐
│  ┌─────────────────────────────┐  │
│  │                             │  │
│  │   INSTITUTION NAME           │  │  ← y = 12%, serif 28-34pt Bold
│  │   (校名/机构名)                │  │
│  │                             │  │
│  │   ━━━━━━━━━━━━━━━━━━━━  │  │  ← thick divider (2pt)
│  │                             │  │
│  │   DOCUMENT TYPE              │  │  ← y = 30%, 20-24pt
│  │   (开题报告/毕业论文/申报书)    │  │
│  │                             │  │
│  │   TITLE                     │  │  ← y = 40%, serif 26-30pt Bold
│  │   (论文题目)                   │  │    max 3 lines, centered
│  │                             │  │
│  │   Field: _______________    │  │  ← y = 58-78%, structured fields
│  │   Field: _______________    │  │    left-label + underline value
│  │   Field: _______________    │  │    e.g. 姓名、学号、导师、院系、日期
│  │   Field: _______________    │  │
│  │   Field: _______________    │  │
│  │                             │  │
│  │   DATE                      │  │  ← y = 88%, centered, 14pt
│  │                             │  │
│  └─────────────────────────────┘  │
└─────────────────────────────────┘
││ = 2.5pt black border, inset 5% from page edge
```

**Best for:** Thesis proposals (开题报告), dissertations, institutional reports, government documents, any formal submission with structured metadata fields

**Content slots:**
| Slot | Required | Example |
|------|----------|---------|
| `institution` | **Required** | "北京大学", "Massachusetts Institute of Technology" |
| `doc_type` | Optional | "开题报告", "Thesis Proposal", "毕业设计" |
| `title` | **Required** | Paper/document title (auto-wrap, max 3 lines) |
| `fields` | Optional | Array of `{label, value}` pairs. Common: 姓名/Name, 学号/ID, 导师/Advisor, 院系/Department, 专业/Major |
| `date` | Optional | "2026年4月", "April 2026" |

**HTML structure:**
```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700;900&family=Noto+Sans+SC:wght@300;400;500;700&family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    @page { size: 794px 1123px; margin: 0; }
    :root {
      --c-bg: #ffffff;
      --c-text: #1a1a1a;
      --c-accent: #1a1a1a;
      --c-muted: #4a4a4a;
      --c-line: #333333;
    }
    html, body { margin: 0; padding: 0; width: 794px; height: 1123px; background: var(--c-bg); color: var(--c-text); font-family: 'Noto Sans SC', 'Inter', sans-serif; }
    @media screen {
      html { height: auto; display: flex; justify-content: center; min-height: 100vh; background: #e8e8e8; }
      body { transform-origin: top center; scale: min(1, calc(100vw / 794), calc(100vh / 1123)); margin: 0 auto; box-shadow: 0 0 60px rgba(0,0,0,0.15); }
    }
    .cover {
      width: 794px; height: 1123px; position: relative; box-sizing: border-box;
    }
    /* Black border frame - inset 5% from page edge */
    .border-frame {
      position: absolute;
      top: 56px; left: 40px; right: 40px; bottom: 56px;
      border: 2.5px solid var(--c-accent);
      pointer-events: none;
    }
    /* Content area inside frame */
    .content {
      position: absolute;
      top: 56px; left: 40px; right: 40px; bottom: 56px;
      display: flex; flex-direction: column; align-items: center;
      padding: 60px 50px;
      box-sizing: border-box;
    }
    .institution {
      font-size: 30pt; font-weight: 700; letter-spacing: 6px;
      font-family: 'Noto Serif SC', 'Playfair Display', serif;
      text-align: center; margin-bottom: 30px;
      max-width: 580px;
    }
    .thick-divider {
      width: 70%; height: 2px; background: var(--c-accent);
      margin-bottom: 40px;
    }
    .doc-type {
      font-size: 22pt; font-weight: 400; letter-spacing: 4px;
      text-align: center; margin-bottom: 50px;
      color: var(--c-text);
    }
    .title {
      font-size: 26pt; font-weight: 700; line-height: 1.4;
      font-family: 'Noto Serif SC', 'Playfair Display', serif;
      text-align: center; margin-bottom: 60px;
      max-width: 520px;
    }
    .fields-block {
      width: 400px; margin-bottom: auto;
    }
    .field-row {
      display: flex; align-items: baseline;
      margin-bottom: 28px; font-size: 14pt;
    }
    .field-label {
      white-space: nowrap; margin-right: 12px;
      color: var(--c-text); font-weight: 400;
      letter-spacing: 2px;
    }
    .field-value {
      flex: 1; text-align: center;
      border-bottom: 1px solid var(--c-line);
      padding-bottom: 4px; min-height: 24px;
      font-family: 'Noto Sans SC', 'Inter', sans-serif;
    }
    .date-block {
      font-size: 14pt; color: var(--c-muted);
      text-align: center; letter-spacing: 2px;
      margin-top: auto; padding-top: 30px;
    }
  </style>
</head>
<body>
  <div class="cover">
    <div class="border-frame"></div>
    <div class="content">
      <div class="institution"><!-- INSTITUTION --></div>
      <div class="thick-divider"></div>
      <div class="doc-type"><!-- DOC_TYPE --></div>
      <div class="title"><!-- TITLE --></div>
      <div class="fields-block">
        <div class="field-row">
          <span class="field-label"><!-- LABEL_1 --></span>
          <span class="field-value"><!-- VALUE_1 --></span>
        </div>
        <div class="field-row">
          <span class="field-label"><!-- LABEL_2 --></span>
          <span class="field-value"><!-- VALUE_2 --></span>
        </div>
        <div class="field-row">
          <span class="field-label"><!-- LABEL_3 --></span>
          <span class="field-value"><!-- VALUE_3 --></span>
        </div>
        <div class="field-row">
          <span class="field-label"><!-- LABEL_4 --></span>
          <span class="field-value"><!-- VALUE_4 --></span>
        </div>
        <div class="field-row">
          <span class="field-label"><!-- LABEL_5 --></span>
          <span class="field-value"><!-- VALUE_5 --></span>
        </div>
      </div>
      <div class="date-block"><!-- DATE --></div>
    </div>
  </div>
</body>
</html>
```

**Layout rules:**
1. **Border frame**: 2.5pt solid black, inset ~5% from all page edges (40px left/right, 56px top/bottom on A4 at 96dpi). This is the defining visual element.
2. **Institution name**: Centered, serif, 28-34pt Bold, letter-spacing 4-6px. For CJK names, use wider letter-spacing (6px). For Latin names, use standard (3px).
3. **Thick divider**: 2pt solid line, 70% width, separates institution from content below.
4. **Document type**: 20-24pt, lighter weight than institution name, letter-spacing 3-4px. This slot differentiates document categories (e.g. "开题报告" / "Thesis Proposal" / "毕业论文" / "Graduation Design").
5. **Title**: Serif, 26-30pt Bold, max 3 lines, centered. Auto-wrap at 520px width.
6. **Structured fields**: Left-aligned label + centered underline value. Label width fixed by longest label in the set. 3-7 field rows supported. Common fields: Name/姓名, Student ID/学号, Advisor/导师, Department/院系, Major/专业.
7. **Date**: Centered at bottom, 14pt, letter-spacing 2px.

**Field auto-detection:**
When the user provides structured metadata (name, student ID, advisor, etc.), auto-populate the fields block. When no fields are provided, omit the fields-block entirely and let the title expand vertically into the freed space.

**Variant 11B - Double border:**
For extra formality (government documents, official submissions), replace the single border with a double border (outer 2.5pt + inner 1pt, 6px gap):
```css
.border-frame {
  border: 2.5px solid var(--c-accent);
  outline: 1px solid var(--c-accent);
  outline-offset: 6px;
}
```

**This template has NO background decoration layer, NO watermarks, NO gradients.** The black frame + white space IS the design.

---

### Academic Template Selection Guide

| Scenario | Template | Rationale |
|----------|----------|-----------|
| arXiv preprint, technical report | **08** (Vertical Anchor) | Left-aligned, data-dense feel |
| IEEE/ACM paper, English thesis | **09** (Symmetric) | Classic bilateral symmetry |
| Chinese thesis, journal with keywords | **10** (Journal) | CJK-optimized, keyword block |
| **Thesis proposal, institutional cover, government doc** | **11** (Institutional) | **White bg, black border frame, structured field slots** |
| Light/formal academic (white bg) | **01-07** (standard templates) | Use standard cover system |


Covers support a **background decoration layer** rendered behind all foreground content (Layer 1). This layer adds subtle depth through supergraphics, typographic watermarks, and blueprint hairlines.

See → `typesetting/cover-backgrounds.md` - complete specification with modules, recipes, and constraint matrix.

**Quick reference:**
- **Recipe A (Minimalist Modern)**: Deep-space arc only - safest, max whitespace
- **Recipe B (Engineering Academic)**: Coordinate cross + vertical spine text - precision, engineering feel
- **Recipe C (Stable & Authoritative)**: Angle slash + bottom bleed text - heavy, authoritative

**Background layer is OPTIONAL.** Not every cover needs one. Templates 01-07 already define their own Layer 1 backgrounds - use the recipes only when a template's built-in background is insufficient.

---

# PART 6: CHANGELOG

| Version | Date | Changes |
|---------|------|---------|
| V1.0 | - | Initial 7 layouts (Diagonal Tension, Vertical Axis, etc.) |
| V2.0 | 2026-04-03 | Complete rewrite. Absolute Anchor Grid; Z-index layers; Typography Weight System; 7 new templates with percentage coordinates; Code-level safety. |
| **V2.1** | **2026-04-03** | **Summary Block upgrade.** Added mandatory Summary/Description drawer to all 7 templates (anti-void iron rule). Introduced base spacing unit `U = W * 0.05`. Refined Hero Title range to 45-65pt. Added S3.4 Hard Width Boundary Enforcement + S3.5 Mandatory Summary Auto-Generation. Template 01: added Summary drawer at Y=0.45*H. Template 02: added Summary at Y=0.45*H + refined watermark to 180pt. Template 03: added Summary at Y=0.40*H with W*0.55 width guard. Template 04: Summary included in center-calculated block. Template 05: lower-right group expanded with Summary + 3pt accent line. Template 06: Zone C explicitly designated for substantial summary text. Template 07: sidebar width changed to `0.1*W` (~80pt), content uses relative vertical centering. |
| **V2.2** | **2026-04-07** | **Intent system unification + Template 11.** Part 2 Template Selection Guide migrated from "Document Tone" to Intent × Document Type matrix (aligned with `visual_framework.md` 5-intent system and `creative.md` Intent Mapping Table). Added Template 11 (Institutional) - white bg + black border frame + structured field slots for thesis proposals, dissertations, and institutional documents. Academic Template Selection Guide updated. |
| **V3.0** | **2026-04-07** | **Color unification + Layout balance overhaul.** (1) All template CSS variables renamed to `--c-` prefix (`--c-bg`, `--c-accent`, `--c-text`, `--c-muted`) for palette system alignment. Hardcoded hex values replaced with CSS variables. Palette tables marked as fallback defaults with `palette.cascade` as canonical source. (2) Added S3.7 Line-Length Alignment - vertical/horizontal lines must match text span. (3) Added S3.8 Vertical Balance - adaptive centering for sparse content, CJK title size compensation (50-80pt vs Latin 45-65pt), anchor points shifted down (title H*0.30, summary H*0.50, meta H*0.70). (4) Output cleanliness rules - no version numbers, draft labels, or process artifacts in final PDF. |
