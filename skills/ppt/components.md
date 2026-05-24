# Components — Pre-built HTML Component Reference Library

Each component is a **starting point, not a straitjacket**. Use them as inspiration and building blocks.

**Three ways to use this library**:
1. **Direct use**: Copy a component template, replace `${...}` variables and text content (safest, guaranteed to convert correctly)
2. **Remix**: Start from a template, then freely modify spacing, font sizes, card styles, background treatments, decorative elements, and layout proportions (encouraged — this creates variety)
3. **Free creation**: Design entirely new layouts from scratch based on html2pptx.md technical constraints (most flexible — must test conversion)

**The goal is visual diversity. If the deck looks monotonous, you haven't remixed enough.**

---

## Usage Rules

1. **Variable replacement**: All `${variableName}` must be replaced with actual values (colors, font names)
2. **Everything is modifiable**: Spacing, font sizes, padding, margins, gap, border-radius, shadows, card styles, background colors, layout proportions — change anything to serve the content and create variety
3. **Only engine constraints are sacred**: Don't use flex-wrap, negative margins, DIV background-image, or CSS gradients (see html2pptx.md Technical Constraints)
4. **Colors must come from the theme scale** — this is the one design-system rule that's enforced
5. **Adjacent slides should look distinctly different** — vary layout, background, card treatment, spacing density
6. **Fill the slide**: Content should use the available space well. Avoid large empty areas — if there's too much whitespace, add more content, use larger type, increase spacing, or add visual elements

---

## Card Style Cookbook (Quick Reference)

Every component template uses shadow-float cards by default. **Swap in these alternatives** to create variety across slides:

| Style | Name | CSS |
|-------|------|-----|
| A | Shadow Float | `background:${surface-card}; border-radius:10pt; box-shadow:0 2pt 8pt rgba(0,0,0,0.08); padding:20pt;` |
| B | Outline | `background:transparent; border:1.5pt solid ${primary-20}; border-radius:10pt; padding:20pt;` |
| C | Solid Fill | `background:${primary-10}; border-radius:10pt; padding:20pt; border:none; box-shadow:none;` |
| D | Left Accent Bar | `background:${surface-card}; border-left:4pt solid ${accent}; border-radius:0 10pt 10pt 0; padding:16pt 20pt; box-shadow:none;` |
| E | Solid Fill + Bottom Accent | `background:${surface}; border-radius:10pt; border-bottom:3pt solid ${accent}; padding:20pt; box-shadow:none;` |
| F | Dark Card | `background:${primary-90}; border-radius:10pt; padding:20pt; color:${on-dark}; box-shadow:none;` |

**Rule: No 2 adjacent slides should use the same card style.**

---

## Title Bar Variants (Quick Reference)

Components use a dark title bar by default. **Swap in these alternatives**:

| Variant | Description |
|---------|-------------|
| Default | `height:56pt; background:${primary-90}` with white bold title at `font-size:22pt` |
| Accent Color | Same structure but `background:${accent}` |
| Transparent + Underline | `padding:28pt 48pt 12pt 48pt; border-bottom:2pt solid ${primary-20};` title at `font-size:28pt; color:${primary-80}` |
| No Header Bar (Inline Title) | No separate header div; title is part of content area at `font-size:32pt; color:${primary-80}` |
| Left Vertical Band | `display:flex; height:56pt;` with `width:6pt; background:${accent}` left strip + rest is `background:${surface}` |

**Rule: All content slides must use the same title bar style — pick one variant and apply it consistently across the entire deck.**

---

## Dark Background Content Templates

Use these dark variants for **rhythm-breaking slides** (recommended every 3-4 pages).

### content-dark-bullets — Dark Background Bullet List

> `list | medium | dark | outline-light`

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-90};font-family:'Microsoft YaHei',sans-serif;">
  <div style="height:4pt;background:${accent};"></div>
  <div style="padding:32pt 48pt 0 48pt;">
    <span style="font-size:28pt;font-weight:bold;color:${on-dark};white-space:nowrap;">Page Title</span>
    <div style="width:40pt;height:3pt;background:${accent};margin-top:8pt;"></div>
  </div>
  <div style="padding:20pt 48pt 36pt 48pt;display:flex;flex-direction:column;gap:14pt;">
    <div style="background:rgba(255,255,255,0.08);border:1pt solid rgba(255,255,255,0.15);border-radius:8pt;padding:16pt 20pt;display:flex;align-items:flex-start;gap:12pt;">
      <div style="width:28pt;height:28pt;background:${accent};border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;">
        <span style="font-size:14pt;font-weight:bold;color:#FFFFFF;">1</span>
      </div>
      <div style="flex:1; min-width:0;">
        <span style="font-size:16pt;font-weight:bold;color:${on-dark};">Point Title</span><br/>
        <span style="font-size:13pt;color:${on-dark-secondary};line-height:1.5;">Description text</span>
      </div>
    </div>
    <!-- Repeat numbered items 2, 3, 4 with same structure -->
  </div>
</body>
```

### content-dark-kpi — Dark Background KPI Dashboard

> `grid | low | dark | solid-dark`

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-100};font-family:'Microsoft YaHei',sans-serif;">
  <div style="padding:36pt 48pt 0 48pt;text-align:center;">
    <span style="font-size:26pt;font-weight:bold;color:${on-dark};">Dashboard Title</span>
  </div>
  <div style="padding:24pt 48pt 36pt 48pt;display:flex;gap:16pt;justify-content:center;">
    <div style="width:192pt;background:rgba(255,255,255,0.06);border:1pt solid rgba(255,255,255,0.12);border-radius:10pt;padding:24pt 16pt;text-align:center;">
      <span style="font-size:38pt;font-weight:bold;color:${accent};white-space:nowrap;">85%</span><br/>
      <span style="font-size:13pt;color:${on-dark-secondary};margin-top:8pt;">Metric Label</span>
    </div>
    <div style="width:192pt;background:rgba(255,255,255,0.06);border:1pt solid rgba(255,255,255,0.12);border-radius:10pt;padding:24pt 16pt;text-align:center;">
      <span style="font-size:38pt;font-weight:bold;color:${accent};white-space:nowrap;">2.4M</span><br/>
      <span style="font-size:13pt;color:${on-dark-secondary};margin-top:8pt;">Metric Label</span>
    </div>
    <div style="width:192pt;background:rgba(255,255,255,0.06);border:1pt solid rgba(255,255,255,0.12);border-radius:10pt;padding:24pt 16pt;text-align:center;">
      <span style="font-size:38pt;font-weight:bold;color:${accent};white-space:nowrap;">+32%</span><br/>
      <span style="font-size:13pt;color:${on-dark-secondary};margin-top:8pt;">Metric Label</span>
    </div>
  </div>
</body>
```

### content-dark-split — Dark Background Left-Right Split

> `split | medium | dark | none`

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-90};font-family:'Microsoft YaHei',sans-serif;">
  <div style="display:flex;height:405pt;">
    <div style="width:260pt;background:${accent};padding:48pt 32pt;display:flex;flex-direction:column;justify-content:center;">
      <span style="font-size:42pt;font-weight:bold;color:#FFFFFF;line-height:1.15;">Key<br/>Insight</span>
      <div style="width:32pt;height:3pt;background:rgba(255,255,255,0.5);margin-top:16pt;"></div>
    </div>
    <div style="flex:1;padding:40pt 36pt;display:flex;flex-direction:column;justify-content:center;gap:16pt;">
      <div style="display:flex;align-items:flex-start;gap:10pt;">
        <div style="width:4pt;height:40pt;background:${accent};flex-shrink:0;border-radius:2pt;margin-top:4pt;"></div>
        <div style="flex:1; min-width:0;">
          <span style="font-size:16pt;font-weight:bold;color:${on-dark};">Sub-point One</span><br/>
          <span style="font-size:13pt;color:${on-dark-secondary};line-height:1.5;">Detailed explanation</span>
        </div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:10pt;">
        <div style="width:4pt;height:40pt;background:${accent};flex-shrink:0;border-radius:2pt;margin-top:4pt;"></div>
        <div style="flex:1; min-width:0;">
          <span style="font-size:16pt;font-weight:bold;color:${on-dark};">Sub-point Two</span><br/>
          <span style="font-size:13pt;color:${on-dark-secondary};line-height:1.5;">Another detailed explanation</span>
        </div>
      </div>
    </div>
  </div>
</body>
```

---

## Component Metadata Tags

Each component is tagged with metadata to help you plan slide variety:
- **Layout type**: `split` | `grid` | `list` | `focus` | `full-bleed` | `timeline` | `centered`
- **Density**: `high` | `medium` | `low`
- **Background**: `light` | `dark` | `image` | `any`
- **Default card style**: `shadow` | `outline` | `solid` | `tag` | `none` | `any`

---

## Table of Contents

### Cover
- [cover-center](#cover-center) — Centered cover `centered | low | light | none`
- [cover-split](#cover-split) — Left-right split cover `split | low | light | none`
- [cover-bottom-bar](#cover-bottom-bar) — Bottom info bar cover `centered | low | light | none`
- [cover-photo-mask](#cover-photo-mask) — Background image + mask cover `centered | low | image | none`
- [cover-dark-hero](#cover-dark-hero) — Dark hero cover, no photo needed `centered | low | dark | none`

### TOC Pages
- [toc-sidebar-list](#toc-sidebar-list) — Sidebar + numbered list `split | medium | light | none`
- [toc-card-grid](#toc-card-grid) — Card grid TOC `grid | medium | light | shadow`
- [toc-timeline](#toc-timeline) — Horizontal timeline TOC `timeline | medium | light | none`
- [toc-big-number](#toc-big-number) — Large number background TOC `list | medium | light | none`

### Content Pages
- [content-header-bullets](#content-header-bullets) — Header bar + bullet list `list | medium | light | none`
- [content-split-text-visual](#content-split-text-visual) — Text/image split (swap columns for image-left variant) `split | medium | light | none`
- [content-split-equal](#content-split-equal) — Equal 50/50 split `split | medium | light | none`
- [content-sidebar-stat](#content-sidebar-stat) — Sidebar large number + right content `split | medium | dark+light | none`
- [content-kpi-row](#content-kpi-row) — KPI large numbers in a row `grid | medium | light | shadow`
- [content-kpi-vertical](#content-kpi-vertical) — KPI numbers vertical stack `focus | medium | light | outline`
- [content-comparison](#content-comparison) — A vs B (or A vs B vs C) comparison `grid | medium | light | tag`
- [content-timeline](#content-timeline) — Horizontal process/steps `timeline | medium | light | none`
- [content-timeline-vertical](#content-timeline-vertical) — Vertical timeline `timeline | medium | light | none`
- [content-icon-grid](#content-icon-grid) — 2×3 icon grid `grid | high | light | none`
- [content-2x2-grid](#content-2x2-grid) — Four-quadrant grid (shadow) `grid | high | light | shadow`
- [content-2x2-grid-outline](#content-2x2-grid-outline) — Four-quadrant grid (outline) `grid | high | light | outline`
- [content-2x2-grid-solid](#content-2x2-grid-solid) — Four-quadrant grid (solid dark) `grid | high | dark | solid`
- [content-full-bleed](#content-full-bleed) — Full-bleed color block `full-bleed | low | dark | none`
- [content-left-accent-bar](#content-left-accent-bar) — Left accent bar emphasis `split | medium | light | none`
- [content-stagger-list](#content-stagger-list) — Staggered list `list | medium | light | none`
- [content-big-number-focus](#content-big-number-focus) — Large number focus `split | low | dark+light | none`
- [content-three-column](#content-three-column) — Three-column with icons `grid | medium | light | none`
- [content-three-card](#content-three-card) — Three card columns `grid | medium | light | shadow`
- [content-split-photo](#content-split-photo) — Left text + right photo `split | medium | light | none`
- [content-photo-cards](#content-photo-cards) — Three photo cards `grid | medium | light | shadow`
- [content-stat-overlay](#content-stat-overlay) — Hero stat on photo `full-bleed | low | image | none`
- [content-photo-overlay](#content-photo-overlay) — Full photo + floating card `full-bleed | low | image | frosted`
- [content-chart-focus](#content-chart-focus) — Chart-focused page `split | medium | light | none`
- [content-chart-bar](#content-chart-bar) — Bar chart + key metric cards `split | medium | light | none`
- [content-chart-pie](#content-chart-pie) — Pie chart + legend & insight `split | medium | light | none`
- [content-chart-line](#content-chart-line) — Wide line chart + trend stats `list | medium | light | none`
- [content-band-top](#content-band-top) — Top color band + content `list | medium | light | none`
- [content-table](#content-table) — Structured data table `data | medium | light | none`
- [content-table-comparison](#content-table-comparison) — Feature comparison matrix `data | medium | light | tag`

### High-Impact Accent Components
- [chapter-divider-bold](#chapter-divider-bold) — Accent panel + dark chapter divider `split | low | dark | none`
- [content-hero-stat](#content-hero-stat) — Single-focus large metric page `centered | low | dark | none`
- [content-asymmetric](#content-asymmetric) — Asymmetric dark panel (38%) + content (62%) `split | medium | dark+light | none`
- [quote-emphasis](#quote-emphasis) — Full-slide pull quote on dark background `centered | low | dark | none`
- [content-dark-three-card](#content-dark-three-card) — Dark background three-card rhythm breaker `grid | medium | dark | solid-dark`

### Data Visualization Pages (see [`data-viz-components.md`](data-viz-components.md))
- content-horizontal-bars — Horizontal bar comparison
- content-stacked-bars — Stacked progress bars
- content-data-table — Structured data table
- content-quadrant-matrix — 2×2 quadrant matrix
- content-funnel — Sales/conversion funnel
- content-before-after — Before vs after
- content-dashboard — Data dashboard / KPI grid
- content-pyramid — Pyramid / layered hierarchy

### Transition Pages
- [divider-bold-center](#divider-bold-center) — Centered bold transition `centered | low | light | none`
- [divider-split](#divider-split) — Split background transition `split | low | dark+light | none`
- [divider-photo-mask](#divider-photo-mask) — Background image + mask transition `centered | low | image | none`
- [divider-gradient](#divider-gradient) — Gradient background transition `centered | low | dark | none`

### Closing Pages
- [closing-takeaways](#closing-takeaways) — Key takeaways summary `grid | medium | light | tag`
- [closing-thankyou](#closing-thankyou) — Thank you page `centered | low | dark | none`

---

## Cover Components

<a id="cover-center"></a>
### cover-center — Centered Cover

Centered symmetrical layout. Accent dot + title + accent line divider + subtitle.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${surface};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="height:4pt;background:${accent};"></div>
  <div style="flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:0 80pt;">
    <div style="width:10pt;height:10pt;border-radius:50%;background:${accent};margin:0 0 24pt 0;"></div>
    <h1 style="font-size:40pt;font-weight:bold;color:${primary-80};margin:0;line-height:1.15;text-align:center;max-width:560pt;">Presentation Title Here</h1>
    <div style="width:40pt;height:3pt;background:${accent};margin:20pt 0;"></div>
    <p style="font-size:18pt;color:${primary-60};margin:0 0 28pt 0;line-height:1.4;text-align:center;max-width:440pt;">Subtitle or one-line overview</p>
    <p style="font-size:13pt;color:${primary-40};margin:0;line-height:1.5;text-align:center;white-space:nowrap;">Presenter: John Smith · December 2024</p>
  </div>
</body>
```

<a id="cover-split"></a>
### cover-split — Left-Right Split Cover

Left dark block (40%) with top anchor label + bottom vertical bar. Right text with accent divider.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:row;">
  <!-- Left: dark panel -->
  <div style="width:288pt;height:405pt;background:${primary-90};display:flex;flex-direction:column;justify-content:space-between;padding:36pt 32pt 40pt 36pt;">
    <div>
      <div style="width:24pt;height:3pt;background:${accent};margin:0 0 10pt 0;"></div>
      <span style="font-size:11pt;color:${accent};letter-spacing:2pt;line-height:1;white-space:nowrap;">ANNUAL REPORT 2024</span>
    </div>
    <div style="display:flex;align-items:flex-end;gap:12pt;">
      <div style="width:3pt;height:44pt;background:${accent};"></div>
      <span style="font-size:13pt;color:${on-dark-secondary};line-height:1.5;">Department · Category</span>
    </div>
  </div>
  <!-- Right: title area -->
  <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:0 48pt 0 40pt;">
    <h1 style="font-size:36pt;font-weight:bold;color:${primary-80};margin:0 0 16pt 0;line-height:1.2;">Presentation Main Title</h1>
    <div style="width:40pt;height:3pt;background:${accent};margin:0 0 20pt 0;"></div>
    <p style="font-size:17pt;color:${primary-60};margin:0 0 28pt 0;line-height:1.4;">Subtitle description text</p>
    <p style="font-size:13pt;color:${primary-40};margin:0;line-height:1.5;white-space:nowrap;">Presenter: John Smith · Product Department</p>
  </div>
</body>
```

<a id="cover-bottom-bar"></a>
### cover-bottom-bar — Bottom Info Bar Cover

Full-width centered title + bottom dark info bar.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:0 48pt;">
    <h1 style="font-size:34pt; font-weight:bold; color:${primary-80}; margin:0 0 12pt 0; line-height:1.15; text-align:center; max-width:580pt;">Presentation Main Title</h1>
    <p style="font-size:18pt; font-weight:bold; color:${primary-60}; margin:0; line-height:1.3; text-align:center; max-width:480pt;">Subtitle description text</p>
  </div>
  <div style="width:720pt; height:48pt; background:${primary-90};
              display:flex; align-items:center; justify-content:center; gap:24pt;">
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0; line-height:1;">Presenter: John Smith</p>
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0; line-height:1;">December 2024</p>
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0; line-height:1;">Product Department</p>
  </div>
</body>
```

<a id="cover-photo-mask"></a>
### cover-photo-mask — Background Image + Mask Cover

Full-screen image + dark mask + centered title. **Download image first**: `curl -L "https://source.unsplash.com/1920x1080/?keyword" -o cover-bg.jpg`

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-image:url('cover-bg.jpg'); background-size:cover; background-position:center;
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="position:absolute; top:0; left:0; width:720pt; height:405pt;
              background-color:rgba(26,51,64,0.75);"></div>
  <div style="position:relative; z-index:1; flex:1; display:flex; flex-direction:column;
              justify-content:center; align-items:center;">
    <div style="width:40pt; height:3pt; background:${accent}; margin:0 0 24pt 0;"></div>
    <h1 style="font-size:34pt; font-weight:bold; color:${on-dark}; margin:0 0 12pt 0; line-height:1.15; text-align:center; max-width:580pt;">Presentation Title Here</h1>
    <p style="font-size:18pt; color:${on-dark-secondary}; margin:0 0 32pt 0; line-height:1.3; text-align:center; max-width:480pt;">Subtitle or one-line overview</p>
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0; line-height:1.5; text-align:center;">Presenter · Date</p>
  </div>
</body>
```
**Mask color rule**: Replace `rgba(26,51,64,0.75)` RGB with theme's primary-90 value.

<a id="cover-dark-hero"></a>
### cover-dark-hero — Dark Background Hero Cover (No Photo Needed)

> `centered | low | dark | none` — ideal fallback when Unsplash fails, strong minimalist opening.

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-100};font-family:'Microsoft YaHei',sans-serif;">
  <div style="height:4pt;background:${accent};"></div>
  <div style="height:401pt;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:0 80pt;">
    <div style="width:10pt;height:10pt;border-radius:50%;background:${accent};margin-bottom:24pt;"></div>
    <span style="font-size:40pt;font-weight:bold;color:${on-dark};text-align:center;line-height:1.2;">Presentation Title</span>
    <div style="width:48pt;height:3pt;background:${accent};margin:20pt 0;"></div>
    <span style="font-size:18pt;color:${on-dark-secondary};text-align:center;line-height:1.4;">Subtitle or one-line description</span>
    <span style="font-size:13pt;color:${on-dark-secondary};margin-top:28pt;">Presenter · Date</span>
  </div>
</body>
```

---

## TOC Components

<a id="toc-sidebar-list"></a>
### toc-sidebar-list — Sidebar + Numbered List

Left dark sidebar (30%) + right numbered chapter list.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:row;">
  <div style="width:216pt; height:405pt; background:${primary-90};
              display:flex; flex-direction:column; justify-content:center; padding:0 32pt;">
    <p style="font-size:13pt; color:${accent}; margin:0 0 8pt 0; line-height:1; text-transform:uppercase; letter-spacing:2pt;">CONTENTS</p>
    <h2 style="font-size:28pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.2;">Contents</h2>
  </div>
  <div style="width:504pt; height:405pt;
              display:flex; flex-direction:column; justify-content:center; padding:0 48pt;">
    <div style="display:flex; align-items:baseline; padding:16pt 0; border-bottom:1pt solid ${primary-10};">
      <p style="font-size:22pt; font-weight:bold; color:${accent}; margin:0; line-height:1; width:48pt;">01</p>
      <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.3;">Chapter One Title</p>
    </div>
    <div style="display:flex; align-items:baseline; padding:16pt 0; border-bottom:1pt solid ${primary-10};">
      <p style="font-size:22pt; font-weight:bold; color:${accent}; margin:0; line-height:1; width:48pt;">02</p>
      <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.3;">Chapter Two Title</p>
    </div>
    <!-- Continue for chapters 03, 04... -->
  </div>
</body>
```

<a id="toc-card-grid"></a>
### toc-card-grid — Card Grid TOC

One numbered card per chapter, 3-4 columns.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${surface};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="padding:40pt 48pt 24pt 48pt;">
    <h2 style="font-size:28pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.2;">Contents</h2>
  </div>
  <div style="display:flex; gap:16pt; padding:0 48pt; flex:1;">
    <div style="width:140pt; flex-shrink:0; background:${surface-card}; border-radius:10pt;
                padding:24pt 16pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); display:flex; flex-direction:column;">
      <p style="font-size:32pt; font-weight:bold; color:${accent}; margin:0 0 12pt 0; line-height:1;">01</p>
      <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3;">Chapter Title</p>
      <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Brief description</p>
    </div>
    <div style="width:140pt; flex-shrink:0; background:${surface-card}; border-radius:10pt;
                padding:24pt 16pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); display:flex; flex-direction:column;">
      <p style="font-size:32pt; font-weight:bold; color:${accent}; margin:0 0 12pt 0; line-height:1;">02</p>
      <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3;">Chapter Title</p>
      <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Brief description</p>
    </div>
    <!-- Continue cards 03, 04... -->
  </div>
  <div style="height:36pt;"></div>
</body>
```

<a id="toc-timeline"></a>
### toc-timeline — Horizontal Timeline TOC

Horizontal nodes + connecting lines + chapter titles. For sequential content.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="padding:40pt 48pt 32pt 48pt;">
    <h2 style="font-size:28pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.2;">Contents</h2>
  </div>
  <div style="flex:1; padding:0 48pt; display:flex; flex-direction:column; justify-content:center;">
    <div style="width:524pt; height:2pt; background:${primary-10}; margin:0 auto 0 50pt;"></div>
    <div style="display:flex; justify-content:space-between; padding:0 24pt; margin-top:8pt;">
      <div style="display:flex; flex-direction:column; align-items:center; width:120pt;">
        <div style="width:28pt; height:28pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:13pt; font-weight:bold; color:#FFFFFF; margin:0; line-height:1;">01</p>
        </div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:12pt 0 4pt 0; line-height:1.3; text-align:center;">Chapter Title</p>
        <p style="font-size:11pt; color:${primary-40}; margin:0; line-height:1.4; text-align:center;">Brief description</p>
      </div>
      <div style="display:flex; flex-direction:column; align-items:center; width:120pt;">
        <div style="width:28pt; height:28pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:13pt; font-weight:bold; color:#FFFFFF; margin:0; line-height:1;">02</p>
        </div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:12pt 0 4pt 0; line-height:1.3; text-align:center;">Chapter Title</p>
        <p style="font-size:11pt; color:${primary-40}; margin:0; line-height:1.4; text-align:center;">Brief description</p>
      </div>
      <!-- Continue nodes 03, 04... -->
    </div>
  </div>
  <div style="height:36pt;"></div>
</body>
```

<a id="toc-big-number"></a>
### toc-big-number — Large Number Background TOC

Extra-large semi-transparent numbers as background, chapter titles float on top.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="padding:40pt 48pt 16pt 48pt;">
    <h2 style="font-size:28pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.2;">Contents</h2>
  </div>
  <div style="flex:1; padding:0 48pt; display:flex; flex-direction:column; justify-content:center; gap:8pt;">
    <div style="display:flex; align-items:center; padding:12pt 24pt;
                background:${surface}; border-radius:10pt; position:relative; overflow:hidden;">
      <p style="font-size:56pt; font-weight:bold; color:rgba(27,42,74,0.06); margin:0; line-height:1; position:absolute; right:16pt; top:-4pt;">01</p>
      <div style="width:3pt; height:32pt; background:${accent}; margin:0 16pt 0 0; border-radius:2pt;"></div>
      <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.3; flex:1; min-width:0;">Chapter One Title</p>
    </div>
    <div style="display:flex; align-items:center; padding:12pt 24pt;
                background:${surface}; border-radius:10pt; position:relative; overflow:hidden;">
      <p style="font-size:56pt; font-weight:bold; color:rgba(27,42,74,0.06); margin:0; line-height:1; position:absolute; right:16pt; top:-4pt;">02</p>
      <div style="width:3pt; height:32pt; background:${accent}; margin:0 16pt 0 0; border-radius:2pt;"></div>
      <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.3; flex:1; min-width:0;">Chapter Two Title</p>
    </div>
    <!-- Continue rows 03, 04... -->
  </div>
  <div style="height:36pt;"></div>
</body>
```

---

## Content Page Components

<a id="content-header-bullets"></a>
### content-header-bullets — Header Bar + Bullet List

Dark header bar + bullet point list with left accent bars.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Page Title</h2>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt; gap:12pt;">
    <div style="display:flex; align-items:flex-start; gap:12pt;">
      <div style="width:3pt; min-height:36pt; background:${accent}; border-radius:2pt; margin-top:2pt;"></div>
      <div style="flex:1; min-width:0;">
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.5;">Key Point Title One</p>
        <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5;">Description text for the key point</p>
      </div>
    </div>
    <div style="display:flex; align-items:flex-start; gap:12pt;">
      <div style="width:3pt; min-height:36pt; background:${accent}; border-radius:2pt; margin-top:2pt;"></div>
      <div style="flex:1; min-width:0;">
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.5;">Key Point Title Two</p>
        <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5;">Description text for the key point</p>
      </div>
    </div>
    <!-- Continue points 3, 4, 5... -->
  </div>
</body>
```

<a id="content-split-text-visual"></a>
### content-split-text-visual — Text / Image Split

Left text (40%) + right image/chart (60%). **For image-left layout, swap the two columns** (put the image div first at `width:360pt`, text div second at `width:240pt`).

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="width:720pt;height:56pt;background:${primary-90};display:flex;align-items:center;padding:0 48pt;">
    <h2 style="font-size:22pt;font-weight:bold;color:${on-dark};margin:0;line-height:1.25;">Page Title</h2>
  </div>
  <div style="flex:1;display:flex;padding:0 48pt;gap:24pt;align-items:center;">
    <div style="width:240pt;display:flex;flex-direction:column;justify-content:center;">
      <p style="font-size:18pt;font-weight:bold;color:${primary-80};margin:0 0 12pt 0;line-height:1.3;">Sub-heading</p>
      <ul style="font-size:15pt;color:${primary-60};margin:0;padding-left:20pt;line-height:23pt;">
        <li>First key point</li>
        <li>Second key point</li>
        <li>Third key point</li>
      </ul>
      <p style="font-size:11pt;color:${primary-40};margin:16pt 0 0 0;line-height:1.4;">Source: 2024 Annual Report</p>
    </div>
    <div style="width:360pt;background:${surface};border-radius:10pt;display:flex;align-items:center;justify-content:center;min-height:200pt;">
      <img src="content-img.jpg" style="width:360pt;height:240pt;object-fit:cover;border-radius:10pt;display:block;" />
    </div>
  </div>
</body>
```

<a id="content-split-equal"></a>
### content-split-equal — Equal 50/50 Split

Two equal columns with a vertical divider.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Page Title</h2>
  </div>
  <div style="flex:1; display:flex; padding:0 48pt; gap:16pt; align-items:center;">
    <div style="width:296pt; flex-shrink:0;">
      <div style="width:40pt; height:3pt; background:${accent}; margin:0 0 12pt 0;"></div>
      <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3;">Left Column Title</p>
      <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5;">Left column content text.</p>
    </div>
    <div style="width:1pt; height:180pt; background:${primary-10};"></div>
    <div style="width:296pt; flex-shrink:0;">
      <div style="width:40pt; height:3pt; background:${accent}; margin:0 0 12pt 0;"></div>
      <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3;">Right Column Title</p>
      <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5;">Right column content text.</p>
    </div>
  </div>
</body>
```

<a id="content-sidebar-stat"></a>
### content-sidebar-stat — Sidebar Large Number + Right Content

Left narrow sidebar with KPIs, right side with detailed content.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:row;">
  <div style="width:200pt; height:405pt; background:${primary-90};
              display:flex; flex-direction:column; justify-content:center; align-items:center; padding:0 24pt;">
    <p style="font-size:40pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1; text-align:center;">86%</p>
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0; line-height:1.4; text-align:center;">User Satisfaction</p>
    <div style="width:32pt; height:1.5pt; background:${accent}; margin:24pt 0;"></div>
    <p style="font-size:40pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1; text-align:center;">2.4×</p>
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0; line-height:1.4; text-align:center;">Efficiency Improvement</p>
  </div>
  <div style="width:520pt; display:flex; flex-direction:column;">
    <div style="padding:40pt 48pt 16pt 40pt;">
      <h2 style="font-size:22pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.25;">Core Business Results</h2>
    </div>
    <div style="padding:0 48pt 0 40pt; flex:1;">
      <ul style="font-size:15pt; color:${primary-60}; margin:0; padding-left:20pt; line-height:23pt;">
        <li>User satisfaction rose from 72% to 86%</li>
        <li>Operational efficiency improved 2.4x YoY</li>
        <li>Client renewal rate reached 95%</li>
        <li>New customer acquisition cost reduced by 35%</li>
      </ul>
    </div>
  </div>
</body>
```

<a id="content-kpi-row"></a>
### content-kpi-row — KPI Large Numbers in a Row

2-4 key data indicators displayed horizontally, cards with shadow elevation.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${surface};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Core Business Metrics</h2>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt;">
    <div style="display:flex; gap:16pt; margin-bottom:16pt;">
      <div style="width:192pt; flex-shrink:0; background:${surface-card}; border-radius:10pt;
                  padding:24pt 16pt; text-align:center; box-shadow:0 3pt 10pt rgba(0,0,0,0.08);">
        <p style="font-size:40pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1;">86%</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.4;">User Retention Rate</p>
      </div>
      <div style="width:192pt; flex-shrink:0; background:${surface-card}; border-radius:10pt;
                  padding:24pt 16pt; text-align:center; box-shadow:0 3pt 10pt rgba(0,0,0,0.08);">
        <p style="font-size:40pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1;">2.4×</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.4;">Customer Acquisition Efficiency</p>
      </div>
      <!-- Continue KPI card 3 (and 4)... -->
    </div>
    <p style="font-size:11pt; color:${primary-40}; margin:0; line-height:1.4;">Data source: 2024 Annual User Survey | Period: Jan-Dec 2024</p>
  </div>
</body>
```

<a id="content-kpi-vertical"></a>
### content-kpi-vertical — KPI Numbers Vertical Stack

> `focus | medium | light | outline`

Left dark sidebar title + right KPI stack with outline cards.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:row;">
  <div style="width:240pt; height:405pt; background:${primary-90};
              display:flex; flex-direction:column; justify-content:center; padding:0 32pt;">
    <div style="width:32pt; height:3pt; background:${accent}; margin:0 0 16pt 0;"></div>
    <h2 style="font-size:28pt; font-weight:bold; color:${on-dark}; margin:0 0 8pt 0; line-height:1.2;">Key Metrics</h2>
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0; line-height:1.5;">Performance highlights for 2024</p>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 40pt; gap:16pt;">
    <div style="display:flex; align-items:center; gap:24pt; padding:16pt 24pt;
                border:1.5pt solid ${primary-20}; border-radius:10pt;">
      <p style="font-size:40pt; font-weight:bold; color:${accent}; margin:0; line-height:1; white-space:nowrap;">92%</p>
      <div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Customer Satisfaction</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Up 12 points from prior year</p>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:24pt; padding:16pt 24pt;
                border:1.5pt solid ${primary-20}; border-radius:10pt;">
      <p style="font-size:40pt; font-weight:bold; color:${accent}; margin:0; line-height:1; white-space:nowrap;">3.1×</p>
      <div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Revenue Growth</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Fastest in company history</p>
      </div>
    </div>
    <!-- Continue KPI item 3... -->
  </div>
</body>
```

<a id="content-comparison"></a>
### content-comparison — A vs B (or A vs B vs C) Comparison

Comparison cards with top accent border. **For 2-way**: two cards at `width:296pt`. **For 3-way**: three cards at `width:192pt;flex-shrink:0` with border colors `${accent}` / `${primary-60}` / `${primary-40}`.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="width:720pt;height:56pt;background:${primary-90};display:flex;align-items:center;padding:0 48pt;">
    <h2 style="font-size:22pt;font-weight:bold;color:${on-dark};margin:0;line-height:1.25;">Plan Comparison</h2>
  </div>
  <div style="flex:1;display:flex;justify-content:center;align-items:center;padding:0 48pt;gap:16pt;">
    <div style="width:296pt;height:240pt;background:${surface};border-radius:10pt;padding:24pt;border-top:3pt solid ${accent};">
      <p style="font-size:18pt;font-weight:bold;color:${primary-80};margin:0 0 16pt 0;line-height:1.3;">Plan A</p>
      <ul style="font-size:15pt;color:${primary-60};margin:0;padding-left:20pt;line-height:23pt;">
        <li>Advantage one</li>
        <li>Advantage two</li>
        <li>Advantage three</li>
      </ul>
    </div>
    <div style="width:296pt;height:240pt;background:${surface};border-radius:10pt;padding:24pt;border-top:3pt solid ${primary-40};">
      <p style="font-size:18pt;font-weight:bold;color:${primary-80};margin:0 0 16pt 0;line-height:1.3;">Plan B</p>
      <ul style="font-size:15pt;color:${primary-60};margin:0;padding-left:20pt;line-height:23pt;">
        <li>Feature one</li>
        <li>Feature two</li>
        <li>Feature three</li>
      </ul>
    </div>
    <!-- For 3-way: add third card at width:192pt, border-top:3pt solid ${primary-40}; reduce other cards to width:192pt -->
  </div>
</body>
```

<a id="content-timeline"></a>
### content-timeline — Horizontal Process/Steps

Horizontal step nodes + descriptions. 3-5 steps.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Implementation Roadmap</h2>
  </div>
  <div style="flex:1; padding:32pt 48pt 36pt 48pt; display:flex; flex-direction:column; justify-content:center;">
    <div style="width:524pt; height:2pt; background:${primary-10}; margin:0 auto 0 48pt;"></div>
    <div style="display:flex; justify-content:space-between; padding:0 16pt; margin-top:8pt;">
      <div style="width:140pt; display:flex; flex-direction:column; align-items:center;">
        <div style="width:32pt; height:32pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:15pt; font-weight:bold; color:#FFFFFF; margin:0; line-height:1;">1</p>
        </div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:12pt 0 4pt 0; line-height:1.3; text-align:center;">Requirements Analysis</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5; text-align:center;">Research user needs</p>
      </div>
      <div style="width:140pt; display:flex; flex-direction:column; align-items:center;">
        <div style="width:32pt; height:32pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:15pt; font-weight:bold; color:#FFFFFF; margin:0; line-height:1;">2</p>
        </div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:12pt 0 4pt 0; line-height:1.3; text-align:center;">Solution Design</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5; text-align:center;">Implementation plan</p>
      </div>
      <!-- Continue steps 3, 4... -->
    </div>
  </div>
</body>
```

<a id="content-timeline-vertical"></a>
### content-timeline-vertical — Vertical Timeline

> `timeline | medium | light | none` — Vertical flow with numbered nodes and connecting lines.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="padding:32pt 48pt 16pt 48pt;">
    <h2 style="font-size:28pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.2;">Timeline</h2>
    <div style="width:40pt; height:3pt; background:${accent};"></div>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt;">
    <div style="display:flex; align-items:flex-start; gap:16pt; margin-bottom:20pt;">
      <div style="display:flex; flex-direction:column; align-items:center; flex-shrink:0;">
        <div style="width:28pt; height:28pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:13pt; font-weight:bold; color:#FFFFFF; margin:0; line-height:1;">1</p>
        </div>
        <div style="width:2pt; height:24pt; background:${primary-10};"></div>
      </div>
      <div style="padding-top:4pt;">
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Phase One — Discovery</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Research and stakeholder interviews</p>
      </div>
    </div>
    <div style="display:flex; align-items:flex-start; gap:16pt; margin-bottom:20pt;">
      <div style="display:flex; flex-direction:column; align-items:center; flex-shrink:0;">
        <div style="width:28pt; height:28pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:13pt; font-weight:bold; color:#FFFFFF; margin:0; line-height:1;">2</p>
        </div>
        <div style="width:2pt; height:24pt; background:${primary-10};"></div>
      </div>
      <div style="padding-top:4pt;">
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Phase Two — Design</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Prototyping and iterative design</p>
      </div>
    </div>
    <!-- Continue phases 3, 4... -->
  </div>
</body>
```

<a id="content-icon-grid"></a>
### content-icon-grid — 2×3 Icon Grid

Grid display of 6 features. **Two independent flex rows, no flex-wrap**. Icon positions use colored circle placeholders.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Core Features</h2>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt;">
    <div style="display:flex; gap:16pt; margin-bottom:24pt;">
      <div style="width:192pt; display:flex; flex-direction:column; align-items:flex-start;">
        <div style="width:36pt; height:36pt; border-radius:8pt; background:${primary-10}; display:flex; align-items:center; justify-content:center; margin-bottom:8pt;">
          <p style="font-size:18pt; color:${accent}; margin:0; line-height:1;">⚡</p>
        </div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.5;">Feature Name</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Brief description</p>
      </div>
      <div style="width:192pt; display:flex; flex-direction:column; align-items:flex-start;">
        <div style="width:36pt; height:36pt; border-radius:8pt; background:${primary-10}; display:flex; align-items:center; justify-content:center; margin-bottom:8pt;">
          <p style="font-size:18pt; color:${accent}; margin:0; line-height:1;">📊</p>
        </div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.5;">Feature Name</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Brief description</p>
      </div>
      <div style="width:192pt; display:flex; flex-direction:column; align-items:flex-start;">
        <div style="width:36pt; height:36pt; border-radius:8pt; background:${primary-10}; display:flex; align-items:center; justify-content:center; margin-bottom:8pt;">
          <p style="font-size:18pt; color:${accent}; margin:0; line-height:1;">🔒</p>
        </div>
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.5;">Feature Name</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Brief description</p>
      </div>
    </div>
    <!-- Row 2: repeat same structure with 3 more icon items -->
  </div>
</body>
```

<a id="content-2x2-grid"></a>
### content-2x2-grid — Four-Quadrant Grid (Shadow Cards)

Four equal-sized cards. **Two independent flex rows, no flex-wrap**.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Four Core Pillars</h2>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt;">
    <div style="display:flex; gap:16pt; margin-bottom:16pt;">
      <div style="width:296pt; height:113pt; background:${surface}; border-radius:10pt; padding:16pt 20pt;">
        <p style="font-size:18pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1.3;">Pillar One</p>
        <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5;">Description content</p>
      </div>
      <div style="width:296pt; height:113pt; background:${surface}; border-radius:10pt; padding:16pt 20pt;">
        <p style="font-size:18pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1.3;">Pillar Two</p>
        <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5;">Description content</p>
      </div>
    </div>
    <!-- Row 2: same structure for Pillars Three and Four -->
  </div>
</body>
```

<a id="content-2x2-grid-outline"></a>
### content-2x2-grid-outline — Four-Quadrant Grid (Outline Cards)

> `grid | high | light | outline` — Same layout as content-2x2-grid but card style is outline.

**Diff from shadow version**: Replace card div `background:${surface}; border-radius:10pt;` with:
```
border:1.5pt solid ${primary-20}; border-radius:10pt; padding:16pt 20pt;
```
Remove any `box-shadow`. All other structure identical.

<a id="content-2x2-grid-solid"></a>
### content-2x2-grid-solid — Four-Quadrant Grid (Solid Dark Cards)

> `grid | high | dark | solid` — Dark background + solid primary-colored cards.

**Diff from shadow version**: Background is `${primary-90}`, cards use `background:${primary-80}`, text colors switch to `${on-dark}` / `${on-dark-secondary}`. Title uses `${on-dark}` with accent underline divider. No box-shadow.

<a id="content-full-bleed"></a>
### content-full-bleed — Full-Bleed Color Block Emphasis Page

Full dark background + decorative element + key insight. For visual rhythm variation.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${primary-90};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column; justify-content:center; align-items:center;">
  <div style="width:48pt; height:48pt; border-radius:50%; background:${accent};
              display:flex; align-items:center; justify-content:center; margin:0 0 24pt 0;">
    <p style="font-size:22pt; font-weight:bold; color:${primary-90}; margin:0; line-height:1;">★</p>
  </div>
  <h2 style="font-size:28pt; font-weight:bold; color:${on-dark}; margin:0 0 16pt 0; line-height:1.3; text-align:center; max-width:520pt;">Core Insight or Key Quote</h2>
  <div style="width:40pt; height:3pt; background:${accent}; margin:0 0 16pt 0;"></div>
  <p style="font-size:15pt; color:${on-dark-secondary}; margin:0; line-height:1.5; text-align:center; max-width:440pt;">Supporting text to explain or expand on the core insight</p>
</body>
```

<a id="content-left-accent-bar"></a>
### content-left-accent-bar — Left Accent Bar Emphasis Page

Left thick accent bar + left color block + right content with numbered points.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:row;">
  <div style="width:8pt; height:405pt; background:${accent};"></div>
  <div style="width:200pt; height:405pt; background:${primary-5};
              display:flex; flex-direction:column; justify-content:center; padding:0 24pt;">
    <p style="font-size:44pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1;">!</p>
    <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.3;">Key Conclusion</p>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt 0 32pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${primary-80}; margin:0 0 16pt 0; line-height:1.25;">Conclusion Title</h2>
    <div style="display:flex; flex-direction:column; gap:12pt;">
      <div style="display:flex; align-items:flex-start; gap:12pt;">
        <div style="width:24pt; height:24pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2pt;">
          <p style="font-size:11pt; font-weight:bold; color:${primary-90}; margin:0; line-height:1;">1</p>
        </div>
        <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5; flex:1; min-width:0;">Detailed description of the first conclusion point</p>
      </div>
      <div style="display:flex; align-items:flex-start; gap:12pt;">
        <div style="width:24pt; height:24pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2pt;">
          <p style="font-size:11pt; font-weight:bold; color:${primary-90}; margin:0; line-height:1;">2</p>
        </div>
        <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5; flex:1; min-width:0;">Detailed description of the second conclusion point</p>
      </div>
      <!-- Continue points 3, 4... -->
    </div>
  </div>
</body>
```

<a id="content-stagger-list"></a>
### content-stagger-list — Staggered List

Each item has a differently colored number block. For processes, rankings, priority lists.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${surface};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="padding:32pt 48pt 16pt 48pt;">
    <h2 style="font-size:28pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.2;">List Title</h2>
    <div style="width:40pt; height:3pt; background:${accent};"></div>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt;">
    <div style="display:flex; align-items:center; gap:16pt; margin-bottom:16pt;">
      <div style="width:48pt; height:48pt; border-radius:10pt; background:${primary-80}; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
        <p style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">01</p>
      </div>
      <div style="flex:1; min-width:0;">
        <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Step One Title</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Description text</p>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:16pt; margin-bottom:16pt;">
      <div style="width:48pt; height:48pt; border-radius:10pt; background:${primary-60}; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
        <p style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">02</p>
      </div>
      <div style="flex:1; min-width:0;">
        <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Step Two Title</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Description text</p>
      </div>
    </div>
    <!-- Continue items 03 (${primary-40}), 04 (${accent})... varying bg colors -->
  </div>
</body>
```

<a id="content-big-number-focus"></a>
### content-big-number-focus — Large Number Focus Page

Oversized number + explanatory text. For data highlights, milestones.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${primary-5};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:row;">
  <div style="width:360pt; height:405pt; background:${primary-90};
              display:flex; flex-direction:column; justify-content:center; align-items:center;">
    <p style="font-size:44pt; font-weight:bold; color:${accent}; margin:0 0 8pt 0; line-height:1;">15 Million</p>
    <div style="width:40pt; height:3pt; background:${accent}; margin:0 0 12pt 0;"></div>
    <p style="font-size:15pt; color:${on-dark-secondary}; margin:0; line-height:1.5; text-align:center;">Number Meaning Label</p>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 40pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${primary-80}; margin:0 0 16pt 0; line-height:1.25;">The Story Behind the Number</h2>
    <p style="font-size:15pt; color:${primary-60}; margin:0 0 16pt 0; line-height:1.6;">Detailed interpretation and background explanation</p>
    <div style="display:flex; gap:24pt;">
      <div>
        <p style="font-size:22pt; font-weight:bold; color:${accent}; margin:0; line-height:1;">85%</p>
        <p style="font-size:11pt; color:${primary-40}; margin:4pt 0 0 0; line-height:1.3;">Related Metric A</p>
      </div>
      <div>
        <p style="font-size:22pt; font-weight:bold; color:${accent}; margin:0; line-height:1;">3.2x</p>
        <p style="font-size:11pt; color:${primary-40}; margin:4pt 0 0 0; line-height:1.3;">Related Metric B</p>
      </div>
    </div>
  </div>
</body>
```

<a id="content-three-column"></a>
### content-three-column — Three-Column Equal-Width Content

Three columns with circular icons + vertical dividers. For "three advantages", "three phases", etc.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="padding:32pt 48pt 0 48pt;">
    <h2 style="font-size:28pt;font-weight:bold;color:${primary-80};margin:0 0 4pt 0;line-height:1.2;">Three-Column Title</h2>
    <div style="width:40pt;height:3pt;background:${accent};"></div>
  </div>
  <div style="flex:1;display:flex;justify-content:center;align-items:center;padding:0 48pt;gap:24pt;">
    <div style="width:184pt;display:flex;flex-direction:column;align-items:center;text-align:center;">
      <div style="width:56pt;height:56pt;border-radius:50%;background:${primary-10};border:3pt solid ${accent};display:flex;align-items:center;justify-content:center;margin-bottom:16pt;">
        <p style="font-size:22pt;font-weight:bold;color:${accent};margin:0;line-height:1;">A</p>
      </div>
      <p style="font-size:18pt;font-weight:bold;color:${primary-80};margin:0 0 8pt 0;line-height:1.3;">Column One Title</p>
      <p style="font-size:13pt;color:${primary-40};margin:0;line-height:1.5;">Brief description text</p>
    </div>
    <div style="width:1pt;height:140pt;background:${primary-10};"></div>
    <!-- Column B: same structure, letter "B", "Column Two Title" -->
    <div style="width:1pt;height:140pt;background:${primary-10};"></div>
    <!-- Column C: same structure, letter "C", "Column Three Title" -->
  </div>
</body>
```

<a id="content-three-card"></a>
### content-three-card — Three Card Columns

> `grid | medium | light | shadow` — Three equal shadow cards with icon boxes. Different from three-column (which has circle icons + dividers).

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${surface};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Three Key Aspects</h2>
  </div>
  <div style="flex:1; display:flex; justify-content:center; align-items:center; padding:0 48pt; gap:16pt;">
    <div style="width:192pt; flex-shrink:0; background:${surface-card}; border-radius:10pt; padding:24pt 16pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08);">
      <div style="width:36pt; height:36pt; border-radius:8pt; background:${primary-10}; display:flex; align-items:center; justify-content:center; margin-bottom:12pt;">
        <p style="font-size:18pt; color:${accent}; margin:0; line-height:1;">A</p>
      </div>
      <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3;">Card Title One</p>
      <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Descriptive text</p>
    </div>
    <div style="width:192pt; flex-shrink:0; background:${surface-card}; border-radius:10pt; padding:24pt 16pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08);">
      <div style="width:36pt; height:36pt; border-radius:8pt; background:${primary-10}; display:flex; align-items:center; justify-content:center; margin-bottom:12pt;">
        <p style="font-size:18pt; color:${accent}; margin:0; line-height:1;">B</p>
      </div>
      <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3;">Card Title Two</p>
      <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Descriptive text</p>
    </div>
    <!-- Card 3 same structure -->
  </div>
</body>
```

<a id="content-split-photo"></a>
### content-split-photo — Left Text + Right Photo

Left: title + text with accent bar. Right: photograph with rounded corners. **Requires downloaded image.**

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h1 style="font-size:28pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.2;">Page Title Here</h1>
  </div>
  <div style="flex:1; display:flex; flex-direction:row; justify-content:center;
              padding:0 48pt; gap:24pt; align-items:center;">
    <div style="width:296pt; flex-shrink:0; display:flex; flex-direction:column; justify-content:center;">
      <div style="width:40pt; height:3pt; background:${accent}; margin:0 0 12pt 0;"></div>
      <h2 style="font-size:22pt; font-weight:bold; color:${primary-80}; margin:0 0 12pt 0; line-height:1.3;">Sub-heading Text</h2>
      <p style="font-size:15pt; color:${primary-60}; margin:0 0 10pt 0; line-height:1.5;">First paragraph explaining a key concept.</p>
      <p style="font-size:15pt; color:${primary-60}; margin:0 0 10pt 0; line-height:1.5;">Second paragraph with additional evidence.</p>
      <p style="font-size:15pt; color:${primary-60}; margin:0; line-height:1.5;">Third paragraph with concluding insight.</p>
    </div>
    <div style="width:296pt; flex-shrink:0; display:flex; align-items:center; justify-content:center;">
      <div style="border-radius:10pt; overflow:hidden; box-shadow:0 4pt 16pt rgba(0,0,0,0.12);">
        <img src="content-img.jpg" style="width:296pt; height:220pt; object-fit:cover; display:block;" />
      </div>
    </div>
  </div>
</body>
```

<a id="content-photo-cards"></a>
### content-photo-cards — Three Photo Cards

Three cards, each with photo header, title, and description. **Requires 3 downloaded images.**

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${primary-5};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h1 style="font-size:28pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.2;">Page Title Here</h1>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; justify-content:center;">
    <div style="display:flex; gap:16pt; padding:0 48pt;">
      <div style="width:192pt; flex-shrink:0; background:${background}; border-radius:10pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); overflow:hidden;">
        <img src="card1.jpg" style="width:192pt; height:120pt; object-fit:cover; display:block;" />
        <div style="padding:16pt;">
          <h3 style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3; white-space:nowrap;">Card Title One</h3>
          <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Brief description</p>
        </div>
      </div>
      <div style="width:192pt; flex-shrink:0; background:${background}; border-radius:10pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); overflow:hidden;">
        <img src="card2.jpg" style="width:192pt; height:120pt; object-fit:cover; display:block;" />
        <div style="padding:16pt;">
          <h3 style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3; white-space:nowrap;">Card Title Two</h3>
          <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Brief description</p>
        </div>
      </div>
      <!-- Card 3 same structure with card3.jpg -->
    </div>
  </div>
</body>
```

<a id="content-stat-overlay"></a>
### content-stat-overlay — Hero Stat on Photo Background

Full-screen photo + dark mask + large centered statistic. **Requires downloaded image.**

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-image:url('stat-bg.jpg'); background-size:cover; background-position:center;
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="position:absolute; top:0; left:0; width:720pt; height:405pt;
              background-color:rgba(26,51,64,0.80);"></div>
  <div style="position:relative; z-index:1; flex:1;
              display:flex; flex-direction:column; justify-content:center; align-items:center;">
    <p style="font-size:13pt; color:${on-dark-secondary}; margin:0 0 8pt 0; line-height:1; text-transform:uppercase; letter-spacing:2pt;">KEY METRIC</p>
    <h1 style="font-size:72pt; font-weight:bold; color:${on-dark}; margin:0 0 4pt 0; line-height:1; white-space:nowrap;">2,500+</h1>
    <p style="font-size:22pt; font-weight:bold; color:${accent}; margin:0 0 24pt 0; line-height:1; white-space:nowrap;">Active Users</p>
    <p style="font-size:15pt; color:${on-dark-secondary}; margin:0; line-height:1.5; text-align:center; max-width:480pt;">Brief description providing context for this statistic</p>
    <div style="width:40pt; height:3pt; background:${accent}; margin:24pt 0 0 0;"></div>
  </div>
</body>
```
**Mask color rule**: Replace `rgba(26,51,64,0.80)` RGB with theme's primary-90 value.

<a id="content-photo-overlay"></a>
### content-photo-overlay — Full Photo + Floating Text Card

> `full-bleed | low | image | frosted` — Full-screen photo + semi-transparent floating card.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-image:url('content-bg.jpg'); background-size:cover; background-position:center;
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="position:absolute; top:0; left:0; width:720pt; height:405pt;
              background-color:rgba(0,0,0,0.3);"></div>
  <div style="position:relative; z-index:1; flex:1; display:flex; align-items:center; justify-content:flex-end; padding:0 48pt;">
    <div style="width:320pt; background:rgba(255,255,255,0.85); border-radius:10pt; padding:32pt 24pt; border:1pt solid rgba(255,255,255,0.5);">
      <div style="width:32pt; height:3pt; background:${accent}; margin:0 0 16pt 0;"></div>
      <h2 style="font-size:22pt; font-weight:bold; color:${primary-80}; margin:0 0 12pt 0; line-height:1.25;">Section Title</h2>
      <p style="font-size:15pt; color:${primary-60}; margin:0 0 12pt 0; line-height:1.5;">Key insight that complements the background image.</p>
      <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1.5;">Supporting detail or data point</p>
    </div>
  </div>
</body>
```

<a id="content-chart-focus"></a>
### content-chart-focus — Chart-Focused Page

> `split | medium | light | none` — Chart takes 70%, narrow text sidebar for interpretation.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">Data Analysis</h2>
  </div>
  <div style="flex:1; display:flex; padding:0 48pt; gap:24pt; align-items:center;">
    <div class="placeholder" id="chart-area" style="width:400pt; flex-shrink:0; background:${surface}; border-radius:10pt; display:flex; align-items:center; justify-content:center; min-height:260pt;"><p style="font-size:13pt; color:${primary-40}; margin:0;">Chart Area</p></div>
    <div style="width:176pt; display:flex; flex-direction:column; justify-content:center;">
      <div style="width:32pt; height:3pt; background:${accent}; margin:0 0 12pt 0;"></div>
      <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0 0 8pt 0; line-height:1.3;">Key Insight</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0 0 16pt 0; line-height:1.5;">Chart interpretation</p>
      <div style="padding:12pt; background:${primary-5}; border-radius:8pt;">
        <p style="font-size:22pt; font-weight:bold; color:${accent}; margin:0 0 4pt 0; line-height:1; white-space:nowrap;">+42%</p>
        <p style="font-size:11pt; color:${primary-40}; margin:0; line-height:1.4;">Year-over-year growth</p>
      </div>
    </div>
  </div>
</body>
```

<a id="content-chart-bar"></a>
### content-chart-bar — Bar Chart + Key Metric Cards

> `split | medium | light | none` — Chart placeholder takes left ~67%, right sidebar shows 3 KPI cards. Column widths: chart 420pt + gap 20pt + sidebar 184pt = 624pt.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="width:720pt;height:56pt;background:${primary-90};display:flex;align-items:center;padding:0 48pt;">
    <h2 style="font-size:22pt;font-weight:bold;color:${on-dark};margin:0;line-height:1.25;">Revenue by Quarter</h2>
  </div>
  <div style="flex:1;display:flex;padding:16pt 48pt;gap:20pt;align-items:center;">
    <div class="placeholder" id="chart-area" style="width:420pt;flex-shrink:0;background:${surface};border-radius:10pt;align-self:stretch;display:flex;align-items:center;justify-content:center;">
      <p style="font-size:13pt;color:${primary-40};margin:0;">Bar Chart</p>
    </div>
    <div style="width:184pt;flex-shrink:0;display:flex;flex-direction:column;justify-content:center;gap:12pt;">
      <div style="padding:12pt;background:${surface};border-radius:8pt;border-left:3pt solid ${accent};">
        <p style="font-size:11pt;color:${primary-40};margin:0 0 4pt 0;line-height:1.3;">Q4（最高）</p>
        <p style="font-size:20pt;font-weight:bold;color:${accent};margin:0;line-height:1;white-space:nowrap;">$2.1M</p>
      </div>
      <div style="padding:12pt;background:${surface};border-radius:8pt;border-left:3pt solid ${primary-40};">
        <p style="font-size:11pt;color:${primary-40};margin:0 0 4pt 0;line-height:1.3;">全年总计</p>
        <p style="font-size:20pt;font-weight:bold;color:${primary-80};margin:0;line-height:1;white-space:nowrap;">$6.6M</p>
      </div>
      <div style="padding:12pt;background:${surface};border-radius:8pt;border-left:3pt solid ${primary-40};">
        <p style="font-size:11pt;color:${primary-40};margin:0 0 4pt 0;line-height:1.3;">同比增长</p>
        <p style="font-size:20pt;font-weight:bold;color:${primary-80};margin:0;line-height:1;white-space:nowrap;">+32%</p>
      </div>
    </div>
  </div>
</body>
```

<a id="content-chart-pie"></a>
### content-chart-pie — Pie Chart + Legend & Insight

> `split | medium | light | none` — Square chart placeholder on left (260×260pt), legend + insight on right (332pt). Legend items use colored dot + label + percentage.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="width:720pt;height:56pt;background:${primary-90};display:flex;align-items:center;padding:0 48pt;">
    <h2 style="font-size:22pt;font-weight:bold;color:${on-dark};margin:0;line-height:1.25;">Market Share Distribution</h2>
  </div>
  <div style="flex:1;display:flex;padding:0 48pt;gap:32pt;align-items:center;">
    <div class="placeholder" id="chart-area" style="width:260pt;height:260pt;flex-shrink:0;background:${surface};border-radius:10pt;display:flex;align-items:center;justify-content:center;">
      <p style="font-size:13pt;color:${primary-40};margin:0;">Pie Chart</p>
    </div>
    <div style="width:332pt;flex-shrink:0;display:flex;flex-direction:column;justify-content:center;gap:12pt;">
      <div style="display:flex;align-items:center;gap:10pt;">
        <div style="width:11pt;height:11pt;border-radius:50%;background:${accent};flex-shrink:0;"></div>
        <p style="font-size:13pt;color:${primary-80};margin:0;flex:1;line-height:1.3;">Category A</p>
        <p style="font-size:15pt;font-weight:bold;color:${accent};margin:0;white-space:nowrap;">42%</p>
      </div>
      <div style="display:flex;align-items:center;gap:10pt;">
        <div style="width:11pt;height:11pt;border-radius:50%;background:${primary-60};flex-shrink:0;"></div>
        <p style="font-size:13pt;color:${primary-80};margin:0;flex:1;line-height:1.3;">Category B</p>
        <p style="font-size:15pt;font-weight:bold;color:${primary-60};margin:0;white-space:nowrap;">28%</p>
      </div>
      <div style="display:flex;align-items:center;gap:10pt;">
        <div style="width:11pt;height:11pt;border-radius:50%;background:${primary-40};flex-shrink:0;"></div>
        <p style="font-size:13pt;color:${primary-80};margin:0;flex:1;line-height:1.3;">Category C</p>
        <p style="font-size:15pt;font-weight:bold;color:${primary-40};margin:0;white-space:nowrap;">18%</p>
      </div>
      <div style="display:flex;align-items:center;gap:10pt;">
        <div style="width:11pt;height:11pt;border-radius:50%;background:${primary-20};flex-shrink:0;"></div>
        <p style="font-size:13pt;color:${primary-80};margin:0;flex:1;line-height:1.3;">Others</p>
        <p style="font-size:15pt;font-weight:bold;color:${primary-20};margin:0;white-space:nowrap;">12%</p>
      </div>
      <div style="height:1pt;background:${primary-10};"></div>
      <p style="font-size:12pt;color:${primary-40};margin:0;line-height:1.5;">Key insight explaining the most important takeaway from the distribution.</p>
    </div>
  </div>
</body>
```

<a id="content-chart-line"></a>
### content-chart-line — Wide Line Chart + Trend Stats

> `list | medium | light | none` — Accent stripe header (no dark bar). Wide chart placeholder stretches to fill vertical space, 4 stat cards pinned to the bottom. Chart area: 624pt wide × ~220pt tall.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="width:720pt;height:8pt;background:${accent};flex-shrink:0;"></div>
  <div style="padding:16pt 48pt 8pt 48pt;display:flex;align-items:baseline;justify-content:space-between;flex-shrink:0;">
    <h2 style="font-size:24pt;font-weight:bold;color:${primary-80};margin:0;line-height:1.25;">Growth Trend</h2>
    <p style="font-size:12pt;color:${primary-40};margin:0;white-space:nowrap;">2021 — 2025</p>
  </div>
  <div class="placeholder" id="chart-area" style="flex:1;margin:0 48pt 12pt 48pt;background:${surface};border-radius:10pt;display:flex;align-items:center;justify-content:center;">
    <p style="font-size:13pt;color:${primary-40};margin:0;">Line Chart</p>
  </div>
  <div style="display:flex;padding:0 48pt 16pt 48pt;gap:12pt;flex-shrink:0;">
    <div style="flex:1;padding:10pt 12pt;background:${surface};border-radius:8pt;">
      <p style="font-size:11pt;color:${primary-40};margin:0 0 3pt 0;white-space:nowrap;">起始值</p>
      <p style="font-size:16pt;font-weight:bold;color:${primary-80};margin:0;line-height:1;white-space:nowrap;">$1.2M</p>
    </div>
    <div style="flex:1;padding:10pt 12pt;background:${accent};border-radius:8pt;">
      <p style="font-size:11pt;color:rgba(255,255,255,0.75);margin:0 0 3pt 0;white-space:nowrap;">当前值</p>
      <p style="font-size:16pt;font-weight:bold;color:#FFFFFF;margin:0;line-height:1;white-space:nowrap;">$6.6M</p>
    </div>
    <div style="flex:1;padding:10pt 12pt;background:${surface};border-radius:8pt;">
      <p style="font-size:11pt;color:${primary-40};margin:0 0 3pt 0;white-space:nowrap;">CAGR</p>
      <p style="font-size:16pt;font-weight:bold;color:${accent};margin:0;line-height:1;white-space:nowrap;">+41%</p>
    </div>
    <div style="flex:1;padding:10pt 12pt;background:${surface};border-radius:8pt;">
      <p style="font-size:11pt;color:${primary-40};margin:0 0 3pt 0;white-space:nowrap;">峰值时间</p>
      <p style="font-size:16pt;font-weight:bold;color:${primary-80};margin:0;line-height:1;white-space:nowrap;">Q3 2025</p>
    </div>
  </div>
</body>
```

<a id="content-band-top"></a>
### content-band-top — Top Color Band + Content Below

> `list | medium | light | none`

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="width:720pt;height:8pt;background:${accent};"></div>
  <div style="padding:24pt 48pt 16pt 48pt;">
    <h2 style="font-size:28pt;font-weight:bold;color:${primary-80};margin:0 0 4pt 0;line-height:1.2;">Page Title</h2>
    <p style="font-size:13pt;color:${primary-40};margin:0;line-height:1.5;">Subtitle</p>
  </div>
  <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:0 48pt;gap:12pt;">
    <div style="display:flex;align-items:flex-start;gap:12pt;">
      <div style="width:3pt;min-height:36pt;background:${accent};border-radius:2pt;margin-top:2pt;"></div>
      <div style="flex:1; min-width:0;"><p style="font-size:15pt;font-weight:bold;color:${primary-80};margin:0 0 4pt 0;line-height:1.5;">Point One</p><p style="font-size:15pt;color:${primary-60};margin:0;line-height:1.5;">Explanation</p></div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12pt;">
      <div style="width:3pt;min-height:36pt;background:${accent};border-radius:2pt;margin-top:2pt;"></div>
      <div style="flex:1; min-width:0;"><p style="font-size:15pt;font-weight:bold;color:${primary-80};margin:0 0 4pt 0;line-height:1.5;">Point Two</p><p style="font-size:15pt;color:${primary-60};margin:0;line-height:1.5;">Explanation</p></div>
    </div>
    <!-- More points... -->
  </div>
</body>
```

<a id="content-table"></a>
### content-table — Structured Data Table with Zebra Rows

> `data | medium | light | none` — **No `<table>` tags — rows are flex divs.** Column widths sum to 624pt.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="height:56pt;background:${primary-90};display:flex;align-items:center;padding:0 48pt;"><h2 style="font-size:22pt;font-weight:bold;color:${on-dark};margin:0;line-height:1.25;">Table Title</h2></div>
  <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:0 48pt;">
    <div style="display:flex;background:${primary-80};border-radius:8pt 8pt 0 0;padding:10pt 16pt;">
      <p style="font-size:13pt;font-weight:bold;color:${on-dark};margin:0;width:180pt;">项目</p>
      <p style="font-size:13pt;font-weight:bold;color:${on-dark};margin:0;width:111pt;text-align:right;">Q1</p>
      <p style="font-size:13pt;font-weight:bold;color:${on-dark};margin:0;width:111pt;text-align:right;">Q2</p>
      <p style="font-size:13pt;font-weight:bold;color:${on-dark};margin:0;width:111pt;text-align:right;">Q3</p>
      <p style="font-size:13pt;font-weight:bold;color:${on-dark};margin:0;width:111pt;text-align:right;">Q4</p>
    </div>
    <div style="display:flex;background:${surface};padding:10pt 16pt;border-left:1pt solid ${primary-10};border-right:1pt solid ${primary-10};">
      <p style="font-size:13pt;color:${primary-80};margin:0;width:180pt;">Engineering</p>
      <p style="font-size:13pt;color:${primary-60};margin:0;width:111pt;text-align:right;">$1.2M</p>
      <p style="font-size:13pt;color:${primary-60};margin:0;width:111pt;text-align:right;">$1.5M</p>
      <p style="font-size:13pt;color:${primary-60};margin:0;width:111pt;text-align:right;">$1.8M</p>
      <p style="font-size:13pt;font-weight:bold;color:${accent};margin:0;width:111pt;text-align:right;">$2.1M</p>
    </div>
    <!-- More zebra rows alternating ${surface}/${background} -->
    <div style="display:flex;background:${primary-10};padding:10pt 16pt;border-radius:0 0 8pt 8pt;">
      <p style="font-size:13pt;font-weight:bold;color:${primary-80};margin:0;width:180pt;">Total</p>
      <p style="font-size:13pt;font-weight:bold;color:${primary-80};margin:0;width:111pt;text-align:right;">$4.0M</p>
      <p style="font-size:13pt;font-weight:bold;color:${accent};margin:0;width:334pt;text-align:right;">$6.2M</p>
    </div>
  </div>
</body>
```

<a id="content-table-comparison"></a>
### content-table-comparison — Feature Comparison Matrix

> `data | medium | light | tag` — Rows = options, columns = criteria, last column = tag. Tag colors: `${accent}` bg = recommended, `${primary-10}` bg = neutral/pass.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="height:56pt;background:${primary-90};display:flex;align-items:center;padding:0 48pt;"><h2 style="font-size:22pt;font-weight:bold;color:${on-dark};margin:0;line-height:1.25;">方案对比</h2></div>
  <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:0 48pt;">
    <div style="display:flex;background:${primary-80};border-radius:8pt 8pt 0 0;padding:10pt 16pt;">
      <p style="font-size:12pt;font-weight:bold;color:${on-dark};margin:0;width:140pt;">方案</p>
      <p style="font-size:12pt;font-weight:bold;color:${on-dark};margin:0;width:120pt;text-align:center;">成本</p>
      <p style="font-size:12pt;font-weight:bold;color:${on-dark};margin:0;width:120pt;text-align:center;">实施周期</p>
      <p style="font-size:12pt;font-weight:bold;color:${on-dark};margin:0;width:120pt;text-align:center;">可扩展性</p>
      <p style="font-size:12pt;font-weight:bold;color:${on-dark};margin:0;width:124pt;text-align:center;">综合评级</p>
    </div>
    <div style="display:flex;align-items:center;background:${surface};padding:10pt 16pt;border-left:1pt solid ${primary-10};border-right:1pt solid ${primary-10};">
      <p style="font-size:13pt;font-weight:bold;color:${primary-80};margin:0;width:140pt;">方案 A</p>
      <p style="font-size:13pt;color:${primary-60};margin:0;width:120pt;text-align:center;">低</p>
      <p style="font-size:13pt;color:${primary-60};margin:0;width:120pt;text-align:center;">3 个月</p>
      <p style="font-size:13pt;color:${primary-60};margin:0;width:120pt;text-align:center;">高</p>
      <div style="width:124pt;display:flex;justify-content:center;"><div style="background:${accent};border-radius:4pt;padding:3pt 10pt;"><p style="font-size:12pt;font-weight:bold;color:#FFFFFF;margin:0;text-align:center;">推荐</p></div></div>
    </div>
    <!-- More rows with different tags -->
  </div>
</body>
```

---

## High-Impact Accent Components

Use these for visual punch: chapter openings, key metric reveals, strong statements, and rhythm breaks.

<a id="chapter-divider-bold"></a>
### chapter-divider-bold — Accent Panel + Dark Chapter Divider

> `split | low | dark | none` — Full-bleed dark slide with bold chapter number. Use at the opening of each major section.

Left 40%: accent-color panel with oversized chapter number. Right 60%: dark background with chapter title and one-line overview. Produces strong structural signal between sections.

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-100};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:row;">
  <!-- Left: accent number panel -->
  <div style="width:288pt;height:405pt;background:${accent};display:flex;flex-direction:column;justify-content:center;align-items:center;">
    <span style="font-size:96pt;font-weight:bold;color:rgba(255,255,255,0.90);line-height:1;white-space:nowrap;">02</span>
    <span style="font-size:11pt;color:rgba(255,255,255,0.65);letter-spacing:3pt;margin-top:8pt;white-space:nowrap;">CHAPTER</span>
  </div>
  <!-- Right: chapter info -->
  <div style="flex:1;height:405pt;display:flex;flex-direction:column;justify-content:center;padding:0 48pt;">
    <div style="width:36pt;height:3pt;background:${accent};margin:0 0 24pt 0;"></div>
    <span style="font-size:30pt;font-weight:bold;color:#FFFFFF;line-height:1.25;">Chapter Title Here</span>
    <span style="font-size:15pt;color:rgba(255,255,255,0.60);margin-top:16pt;line-height:1.6;">One-line description of this chapter's content</span>
  </div>
</body>
```

<a id="content-hero-stat"></a>
### content-hero-stat — Single-Focus Large Metric Page

> `centered | low | dark | none` — One dominant number, optionally with 2 supporting context metrics below. Use for single powerful KPI reveals, survey results, or market data.

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-90};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="height:4pt;background:${accent};"></div>
  <div style="flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:0 80pt;">
    <span style="font-size:13pt;color:${accent};letter-spacing:2pt;text-align:center;line-height:1;white-space:nowrap;">KEY METRIC</span>
    <span style="font-size:88pt;font-weight:bold;color:#FFFFFF;line-height:1;text-align:center;margin-top:12pt;white-space:nowrap;">85%</span>
    <span style="font-size:18pt;color:rgba(255,255,255,0.70);margin-top:16pt;text-align:center;line-height:1.5;max-width:460pt;">Description of what this metric means</span>
    <!-- Optional: 2 supporting context metrics -->
    <div style="display:flex;gap:32pt;margin-top:32pt;padding:14pt 28pt;background:rgba(255,255,255,0.06);border-radius:8pt;">
      <div style="text-align:center;">
        <span style="font-size:22pt;font-weight:bold;color:${accent};line-height:1;display:block;white-space:nowrap;">+12%</span>
        <span style="font-size:11pt;color:rgba(255,255,255,0.50);margin-top:4pt;display:block;white-space:nowrap;">vs last year</span>
      </div>
      <div style="text-align:center;">
        <span style="font-size:22pt;font-weight:bold;color:${accent};line-height:1;display:block;white-space:nowrap;">#3</span>
        <span style="font-size:11pt;color:rgba(255,255,255,0.50);margin-top:4pt;display:block;white-space:nowrap;">Industry rank</span>
      </div>
    </div>
  </div>
</body>
```

<a id="content-asymmetric"></a>
### content-asymmetric — Asymmetric Dark Panel + Content

> `split | medium | dark+light | none` — Left 38% dark panel with section label and framing tagline; right 62% light area with 3 numbered key points. Breaks the visual monotony of symmetric layouts.

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:row;">
  <!-- Left: dark framing panel -->
  <div style="width:274pt;height:405pt;background:${primary-90};display:flex;flex-direction:column;justify-content:center;padding:0 32pt;">
    <span style="font-size:11pt;color:${accent};letter-spacing:2pt;line-height:1;margin:0 0 16pt 0;white-space:nowrap;">SECTION</span>
    <span style="font-size:32pt;font-weight:bold;color:#FFFFFF;line-height:1.2;margin:0 0 20pt 0;">Core<br/>Concept</span>
    <div style="width:32pt;height:3pt;background:${accent};margin:0 0 16pt 0;"></div>
    <span style="font-size:13pt;color:rgba(255,255,255,0.60);line-height:1.6;">Brief framing of this content area</span>
  </div>
  <!-- Right: content with numbered points -->
  <div style="flex:1;height:405pt;display:flex;flex-direction:column;justify-content:center;padding:0 40pt;gap:16pt;">
    <div style="display:flex;align-items:flex-start;gap:14pt;">
      <div style="width:28pt;height:28pt;background:${accent};border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;">
        <span style="font-size:13pt;font-weight:bold;color:#FFFFFF;line-height:1;">1</span>
      </div>
      <div style="flex:1;min-width:0;">
        <span style="font-size:16pt;font-weight:bold;color:${primary-80};line-height:1.3;display:block;">Point Title One</span>
        <span style="font-size:13pt;color:${primary-60};line-height:1.5;display:block;margin-top:4pt;">Description text supporting this point</span>
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:14pt;">
      <div style="width:28pt;height:28pt;background:${accent};border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;">
        <span style="font-size:13pt;font-weight:bold;color:#FFFFFF;line-height:1;">2</span>
      </div>
      <div style="flex:1;min-width:0;">
        <span style="font-size:16pt;font-weight:bold;color:${primary-80};line-height:1.3;display:block;">Point Title Two</span>
        <span style="font-size:13pt;color:${primary-60};line-height:1.5;display:block;margin-top:4pt;">Description text supporting this point</span>
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:14pt;">
      <div style="width:28pt;height:28pt;background:${accent};border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;">
        <span style="font-size:13pt;font-weight:bold;color:#FFFFFF;line-height:1;">3</span>
      </div>
      <div style="flex:1;min-width:0;">
        <span style="font-size:16pt;font-weight:bold;color:${primary-80};line-height:1.3;display:block;">Point Title Three</span>
        <span style="font-size:13pt;color:${primary-60};line-height:1.5;display:block;margin-top:4pt;">Description text supporting this point</span>
      </div>
    </div>
  </div>
</body>
```

<a id="quote-emphasis"></a>
### quote-emphasis — Full-Slide Pull Quote

> `centered | low | dark | none` — Full dark slide with oversized opening quotation mark, large quote text, and attribution. Use for powerful statements, expert opinions, or memorable data points framed as quotes. **For a light-background variant**: replace `background-color:${primary-90}` with `${surface}`, `color:#FFFFFF` with `${primary-80}`, and `rgba(255,255,255,0.55)` with `${primary-40}`.

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-90};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;justify-content:center;">
  <div style="padding:0 80pt;">
    <span style="font-size:80pt;font-weight:bold;color:${accent};line-height:0.8;display:block;margin:0 0 4pt 0;">"</span>
    <span style="font-size:24pt;color:#FFFFFF;line-height:1.55;display:block;margin:0 0 28pt 0;max-width:520pt;">Quote text goes here — one or two lines of impactful, memorable prose.</span>
    <div style="width:48pt;height:3pt;background:${accent};margin:0 0 14pt 0;"></div>
    <span style="font-size:13pt;color:rgba(255,255,255,0.55);line-height:1.5;">— Person Name, Title · Year</span>
  </div>
</body>
```

<a id="content-dark-three-card"></a>
### content-dark-three-card — Dark Background Three-Card Rhythm Breaker

> `grid | medium | dark | solid-dark` — Same role as a regular three-card content page but on a dark background. Use every 3–4 slides to break white-page monotony. Inline title (no title bar) creates visual distinction from light-background pages.

```html
<body style="margin:0;padding:0;width:720pt;height:405pt;overflow:hidden;background-color:${primary-90};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <!-- Inline title with horizontal rule — no title bar -->
  <div style="padding:28pt 48pt 20pt 48pt;display:flex;align-items:center;gap:16pt;">
    <span style="font-size:26pt;font-weight:bold;color:#FFFFFF;line-height:1.2;white-space:nowrap;">Page Title</span>
    <div style="flex:1;height:1pt;background:rgba(255,255,255,0.12);"></div>
  </div>
  <!-- Three semi-transparent cards -->
  <div style="flex:1;display:flex;gap:14pt;padding:0 48pt 32pt 48pt;align-items:stretch;">
    <div style="flex:1;background:rgba(255,255,255,0.06);border:1pt solid rgba(255,255,255,0.10);border-radius:10pt;padding:20pt 18pt;display:flex;flex-direction:column;">
      <div style="width:32pt;height:3pt;background:${accent};margin:0 0 14pt 0;"></div>
      <span style="font-size:16pt;font-weight:bold;color:#FFFFFF;line-height:1.3;margin:0 0 8pt 0;display:block;">Card Title One</span>
      <span style="font-size:13pt;color:rgba(255,255,255,0.60);line-height:1.6;display:block;">Card description text, one or two sentences.</span>
    </div>
    <div style="flex:1;background:rgba(255,255,255,0.06);border:1pt solid rgba(255,255,255,0.10);border-radius:10pt;padding:20pt 18pt;display:flex;flex-direction:column;">
      <div style="width:32pt;height:3pt;background:${accent};margin:0 0 14pt 0;"></div>
      <span style="font-size:16pt;font-weight:bold;color:#FFFFFF;line-height:1.3;margin:0 0 8pt 0;display:block;">Card Title Two</span>
      <span style="font-size:13pt;color:rgba(255,255,255,0.60);line-height:1.6;display:block;">Card description text, one or two sentences.</span>
    </div>
    <div style="flex:1;background:rgba(255,255,255,0.06);border:1pt solid rgba(255,255,255,0.10);border-radius:10pt;padding:20pt 18pt;display:flex;flex-direction:column;">
      <div style="width:32pt;height:3pt;background:${accent};margin:0 0 14pt 0;"></div>
      <span style="font-size:16pt;font-weight:bold;color:#FFFFFF;line-height:1.3;margin:0 0 8pt 0;display:block;">Card Title Three</span>
      <span style="font-size:13pt;color:rgba(255,255,255,0.60);line-height:1.6;display:block;">Card description text, one or two sentences.</span>
    </div>
  </div>
</body>
```

---

## Transition Page Components

<a id="divider-bold-center"></a>
### divider-bold-center — Centered Bold Text Transition

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;justify-content:center;align-items:center;">
  <div style="width:40pt;height:3pt;background:${accent};margin:0 0 24pt 0;"></div>
  <p style="font-size:44pt;font-weight:bold;color:${primary-20};margin:0 0 8pt 0;line-height:1;">02</p>
  <h2 style="font-size:28pt;font-weight:bold;color:${primary-80};margin:0 0 12pt 0;line-height:1.2;text-align:center;">Chapter Title</h2>
  <p style="font-size:15pt;color:${primary-40};margin:0;line-height:1.5;text-align:center;max-width:400pt;">One-line overview</p>
</body>
```

<a id="divider-split"></a>
### divider-split — Split Background Transition

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${background};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:row;">
  <div style="width:324pt;height:405pt;background:${primary-90};display:flex;flex-direction:column;justify-content:center;align-items:center;"><p style="font-size:80pt;font-weight:bold;color:rgba(255,255,255,0.08);margin:0;line-height:1;">02</p></div>
  <div style="width:396pt;height:405pt;display:flex;flex-direction:column;justify-content:center;padding:0 48pt;">
    <div style="width:32pt;height:3pt;background:${accent};margin:0 0 16pt 0;"></div>
    <h2 style="font-size:28pt;font-weight:bold;color:${primary-80};margin:0 0 12pt 0;line-height:1.2;">Chapter Title</h2>
    <p style="font-size:15pt;color:${primary-40};margin:0;line-height:1.5;max-width:280pt;">One-line overview</p>
  </div>
</body>
```

<a id="divider-photo-mask"></a>
### divider-photo-mask — Background Image + Mask Transition

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-image:url('section-bg.jpg');background-size:cover;background-position:center;font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="position:absolute;top:0;left:0;width:720pt;height:405pt;background-color:rgba(26,51,64,0.7);"></div>
  <div style="position:relative;z-index:1;flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;">
    <div style="width:40pt;height:3pt;background:${accent};margin:0 0 24pt 0;"></div>
    <p style="font-size:44pt;font-weight:bold;color:${accent};margin:0 0 8pt 0;line-height:1;">02</p>
    <h2 style="font-size:28pt;font-weight:bold;color:${on-dark};margin:0 0 12pt 0;line-height:1.2;text-align:center;">Chapter Title</h2>
    <p style="font-size:15pt;color:${on-dark-secondary};margin:0;line-height:1.5;text-align:center;max-width:400pt;">One-line overview</p>
  </div>
</body>
```

<a id="divider-gradient"></a>
### divider-gradient — Gradient Background Transition

> Requires pre-generating gradient PNG via sharp/SVG.

```javascript
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080"><defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:${primary-100}"/><stop offset="100%" style="stop-color:${primary-80}"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#g)"/></svg>`;
await sharp(Buffer.from(svg)).png().toFile('gradient-bg.png');
```

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-image:url('gradient-bg.png');background-size:cover;font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;justify-content:center;align-items:center;">
  <div style="width:40pt;height:3pt;background:${accent};margin:0 0 24pt 0;"></div>
  <p style="font-size:44pt;font-weight:bold;color:${accent};margin:0 0 8pt 0;line-height:1;">03</p>
  <h2 style="font-size:28pt;font-weight:bold;color:${on-dark};margin:0 0 12pt 0;line-height:1.2;text-align:center;">Chapter Title</h2>
  <p style="font-size:15pt;color:${on-dark-secondary};margin:0;line-height:1.5;text-align:center;max-width:400pt;">One-line overview</p>
</body>
```

---

## Closing Page Components

<a id="closing-takeaways"></a>
### closing-takeaways — Key Takeaways Summary

Header bar + 3 key takeaway cards with top accent borders.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${surface};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;">
  <div style="width:720pt;height:56pt;background:${primary-90};display:flex;align-items:center;padding:0 48pt;"><h2 style="font-size:22pt;font-weight:bold;color:${on-dark};margin:0;line-height:1.25;">Key Conclusions</h2></div>
  <div style="flex:1;display:flex;justify-content:center;align-items:center;padding:0 48pt;gap:16pt;">
    <div style="width:192pt;flex-shrink:0;background:${surface-card};border-radius:10pt;padding:24pt 16pt;box-shadow:0 3pt 10pt rgba(0,0,0,0.08);border-top:3pt solid ${accent};">
      <p style="font-size:22pt;font-weight:bold;color:${accent};margin:0 0 8pt 0;line-height:1;">01</p>
      <p style="font-size:15pt;font-weight:bold;color:${primary-80};margin:0 0 8pt 0;line-height:1.5;">Takeaway Point One</p>
      <p style="font-size:13pt;color:${primary-40};margin:0;line-height:1.5;">Brief note</p>
    </div>
    <div style="width:192pt;flex-shrink:0;background:${surface-card};border-radius:10pt;padding:24pt 16pt;box-shadow:0 3pt 10pt rgba(0,0,0,0.08);border-top:3pt solid ${accent};">
      <p style="font-size:22pt;font-weight:bold;color:${accent};margin:0 0 8pt 0;line-height:1;">02</p>
      <p style="font-size:15pt;font-weight:bold;color:${primary-80};margin:0 0 8pt 0;line-height:1.5;">Takeaway Point Two</p>
      <p style="font-size:13pt;color:${primary-40};margin:0;line-height:1.5;">Brief note</p>
    </div>
    <!-- Card 3 same structure -->
  </div>
</body>
```

<a id="closing-thankyou"></a>
### closing-thankyou — Thank You Page

Dark background + large thank-you + contact info. Echoes cover.

```html
<body style="width:720pt;height:405pt;margin:0;padding:0;overflow:hidden;background-color:${primary-90};font-family:'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;justify-content:center;align-items:center;">
  <h1 style="font-size:34pt;font-weight:bold;color:${on-dark};margin:0 0 8pt 0;line-height:1.15;text-align:center;">Thank You</h1>
  <div style="width:40pt;height:3pt;background:${accent};margin:0 0 24pt 0;"></div>
  <p style="font-size:15pt;color:${on-dark-secondary};margin:0 0 4pt 0;line-height:1.5;text-align:center;">John Smith | Product Department</p>
  <p style="font-size:13pt;color:${on-dark-secondary};margin:0;line-height:1.5;text-align:center;">john.smith@company.com</p>
</body>
```