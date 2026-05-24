# Design System — PPT Creative Guidelines

This file provides **creative principles and a color system** for PPT design. Components and layouts should be inventive and diverse — these guidelines exist to ensure visual coherence, not to constrain creativity.

**Philosophy: Every slide is a design opportunity.** No two slides should look the same. Vary layouts, card treatments, backgrounds, typography weight, and whitespace aggressively. The audience should feel visual momentum, not repetition.

---

## 1. Color Scale System (Mandatory)

The color system is the **only hard constraint**. All colors used in the deck must derive from the chosen theme.

### 1.1 Color Scale Generation

Eight levels are derived from a single Primary color:

| Level | Lightness | Usage Examples |
|-------|-----------|---------------|
| `primary-100` | Darkest (L≈8%) | Dark backgrounds, dramatic pages |
| `primary-90` | Dark (L≈15%) | Title bars, dark blocks, overlays |
| `primary-80` | **Main color** | Headings, primary UI elements |
| `primary-60` | Medium dark (L≈45%) | Subheadings, icons, secondary elements |
| `primary-40` | Medium light (L≈60%) | Muted text, dividers, subtle accents |
| `primary-20` | Light (L≈80%) | Borders, light decorations, tints |
| `primary-10` | Very light (L≈90%) | Card fills, surface backgrounds |
| `primary-5` | Near white (L≈96%) | Page tints, hover states, subtle surfaces |

**Generation method**: From the Primary HSL (H, S, L) — keep H constant; dark levels S×1.1, light levels S×0.3; assign L per the table.

### 1.2 Semantic Colors

| Token | Value | Purpose |
|-------|-------|---------|
| `background` | `#FFFFFF` | Default page base |
| `surface` | `primary-5` | Tinted content area base |
| `surface-card` | `#FFFFFF` | Card base (floats above surface) |
| `text-primary` | `primary-80` | Primary text |
| `text-secondary` | `primary-60` | Secondary text |
| `text-muted` | `primary-40` | Annotations, footnotes |
| `on-dark` | `#FFFFFF` | Text on dark backgrounds |
| `on-dark-secondary` | `rgba(255,255,255,0.7)` | Secondary text on dark backgrounds |
| `border` | `primary-10` | Dividers, borders |
| `accent` | Defined by theme | Highlight color |
| `on-accent` | `#FFFFFF` | Text on accent color |

### 1.3 Color Rules

- **All grays must derive from the primary color scale** — no arbitrary `#666`, `#999`, `#DDD`
- **Text on dark areas must use `on-dark`** (#FFFFFF) — never use light primary tints on dark backgrounds
- **Accent color area**: ≤15% on standard pages; up to 40% on emphasis/transition pages
- **Saturation limit**: Accent colors should have HSL saturation ≤ 65% for most themes. High-energy themes (Deep Mineral, Ember, Mono) may use S ≤ 75% to preserve their intended vibrancy. Colors above S=80% look garish on projected slides. The theme presets already comply; if creating custom colors, check saturation.
- **Large color blocks** (>15% of page area): reduce saturation further (S ≤ 50%) to avoid visual fatigue. Use higher saturation only for small accents (bars, icons, small badges).

---

## 2. Creative Principles (Guidelines, Not Rules)

These are high-level design intentions. Interpret them freely.

### 2.1 Visual Diversity

- **No two adjacent slides should share the same layout, background treatment, or card style.** This is the single most important principle.
- Aim for **5+ distinct layout structures** across a deck (split columns, grids, single-focus, full-bleed, asymmetric, timeline, staggered, etc.)
- Aim for **3+ card styles** (shadow float, outline, solid fill, gradient, frosted glass, tag-style, image-backed, borderless, etc.)
- Aim for **3+ background treatments** (pure white, tinted surface, dark full-bleed, photo + mask, gradient, diagonal split, corner decoration, color band, etc.)

### 2.1.1 Deck-Wide Consistency Rules

**What MUST stay uniform across the entire deck** (changing these = unprofessional):
- Body text font and font size (e.g., always 15pt Microsoft YaHei)
- Page left/right margins (e.g., always 48pt)
- Primary color palette (same theme throughout)
- Accent color (same accent variant throughout, or intentional A/B mixing as documented)

**What SHOULD vary across slides** (sameness = monotony):
- Layout structure (grid, split, list, focus, timeline, etc.)
- Background treatment (white, surface, dark, photo, gradient)
- Card style (shadow, outline, solid, accent-bar, dark card)
- Title bar style (dark bar, accent bar, underline, inline title, vertical band)
- Decorative elements (accent bars, shapes, circles, dot patterns)

**The boundary is clear: base typography and spacing = consistent; visual composition = diverse.**

### 2.2 Intentional vs Accidental Whitespace

Not all whitespace is bad. The key is **intent**.

**Intentional whitespace (good — preserve it):**
- KPI / big-number pages: Large number centered with breathing room → dramatic focus
- Quote / pullout pages: Single sentence with generous margins → elegant emphasis
- Photo + caption pages: Image dominates, text is minimal → visual storytelling
- Title / divider pages: Bold text with open space → transition signal

**Accidental whitespace (bad — fix it):**
- Bullet list with 3 items occupying only 1/3 of the slide → expand or add visuals
- Card grid with small cards clustered at the top, bottom 60% empty → enlarge cards, increase padding
- Text block in the upper-left corner with nothing else on the page → redistribute content

**Rule of thumb**: If the whitespace makes the slide feel "unfinished" or "forgot to add content," it's accidental. If it makes the slide feel "clean" or "focused," it's intentional.

### 2.3 Visual Rhythm

- Alternate between high-density and low-density pages
- Insert a "rhythm breaker" every 3-4 slides — a page that looks dramatically different (full-bleed dark, photo overlay, large single quote, bold color block)
- Cover and closing page should echo each other in color or mood
- **Vary the page header treatment** — don't use the same dark title bar on every slide. Alternatives:
  - Accent-colored bar (e.g., `background:${accent}`)
  - Transparent header with large colored text and a bottom divider line
  - No header bar at all — use oversized inline heading text
  - Left-side vertical color band instead of top horizontal bar
  - Gradient header (dark-to-transparent from top)

### 2.4 Typography

- **Hierarchy is king**: Headings should be visually distinct from body text through size, weight, and/or color
- Body text should be readable at projection distance — don't go below ~13pt
- Let type breathe — generous line-height makes everything feel more designed
- KPI / hero numbers should be oversized and bold to create a focal point
- Prefer cutting content over shrinking fonts — only when a slide is already well-filled and overflow is imminent; never cut content preemptively

### 2.5 Spacing & Layout

- Use consistent margins across the deck (recommended: 48pt left/right, 40pt top, ≥36pt bottom safe zone)
- Content should be **vertically balanced** — not crammed at the top with empty space below
- Give elements room — tight packing reads as "cheap"
- Fixed widths for multi-column cards prevent uneven stretching

### 2.6 Decoration & Visual Interest

- Every content slide should have **at least one non-text visual element** — accent bar, color block, photo, shape, large number, chart
- Decorative elements (circles, bars, geometric shapes, dot patterns) add polish and visual rhythm
- Decoration should support the content, never compete with it

### 2.7 Imagery

- Real photographs dramatically elevate quality — use them on covers, dividers, and visual pages
- Background images need a semi-transparent mask overlay for text readability
- Image tone (warm/cool) should match the theme atmosphere

---

## 3. Technical Constraints (html2pptx Engine)

These are **rendering engine limitations**, not design choices. Violating them causes broken output.

| Constraint | Reason |
|-----------|--------|
| Slide canvas: 720×405pt (16:9) | Fixed PPTX dimensions |
| `background-image` only works on `<body>`, not on `<div>` | html2pptx engine limitation |
| Don't use `flex-wrap` for multi-row grids — use separate flex containers per row | html2pptx drops wrapped rows |
| Don't use negative margins | Causes text stacking in PPT (elements become independent text boxes) |
| `font-family` must include a CJK font name | Required for correct font mapping |
| Images must be local file paths, not URLs | html2pptx reads from filesystem |
| Content that exceeds 405pt height will overflow — split into multiple slides instead | No scroll in PPT |
| Multi-column equal-width cards: use explicit `width` + `flex-shrink:0`, not `flex:1` | Prevents uneven stretching |
| Titles and short labels: use `white-space:nowrap` | Prevents unexpected line breaks in conversion |

---

## 4. Quick Reference

### Recommended Font Size Range

| Role | Suggested Size | Weight |
|------|---------------|--------|
| Display / decorative number (ghost) | 80–100pt | Bold |
| Hero focal metric (single-stat page) | 56–88pt | Bold |
| Hero / cover title | 40–48pt | Bold |
| KPI dashboard numbers | 36–48pt | Bold |
| Section title | 28–34pt | Bold |
| Page title (in title bar) | 26–30pt | Bold |
| Card / sub heading | 20–24pt | Bold |
| Body text | 14–16pt | Normal |
| Annotations / footnotes | 12–13pt | Normal |
| Captions / tiny labels | 10–12pt | Normal |
| Tag / chip label | 10–11pt | Normal |

**Ghost number technique**: Large decorative numbers (80–100pt) at very low opacity (`color:rgba(255,255,255,0.05)` on dark backgrounds, or `color:rgba(0,0,0,0.04)` on light) add visual depth without competing with readable content. Use as background layer behind chapter numbers, inside TOC cards, or behind section titles.

### Common Spacing Values

| Usage | Suggested |
|-------|-----------|
| Page left/right margins | 48pt |
| Page top margin | 40pt |
| Bottom safe zone | ≥36pt |
| Between heading and body | 8-16pt |
| Between cards | 12-20pt |
| Card internal padding | 16-24pt |

### Width Reference (720pt total)

| Layout | Approximate Column Widths |
|--------|--------------------------|
| Full width (with margins) | ~624pt |
| Two columns | ~296pt each (16pt gap) |
| Three columns | ~192pt each (16pt×2 gaps) |
| Four columns | ~140pt each (16pt×3 gaps) |

### Height Reference (405pt total)

| Component | Approximate Height |
|-----------|--------------------|
| Title bar | ~56pt |
| Single text line (15pt body) | ~23pt |
| KPI number line (40pt) | ~40pt |
| Card padding (top+bottom) | ~32-48pt |

---

**Remember: This system gives you a coherent color palette and a few engine-level constraints. Everything else — layout invention, typography expression, decorative flair, card styling, background artistry — is yours to create freely. Make every slide count.**
