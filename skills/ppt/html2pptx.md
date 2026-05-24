# HTML to PowerPoint Guide

Convert HTML slides into PowerPoint presentations using the `html2pptx.js` library with accurate positioning.

---

## Creating HTML Slides

Each HTML slide must include the correct body dimensions:

### Layout Dimensions

- **16:9** (default): `width: 720pt; height: 405pt`
- **4:3**: `width: 720pt; height: 540pt`
- **16:10**: `width: 720pt; height: 450pt`

**CRITICAL — Prevent Overflow**:
- `<body>` must set exact dimensions via inline style: `width: 720pt; height: 405pt; margin: 0; padding: 0; overflow: hidden;`
- All content must fit within these boundaries. If content overflows, split into multiple slides
- Maintain at least **36pt bottom margin** inside the body for safe spacing

**Fill the slide**: Content should occupy the available space well. Avoid designs where text clusters in the top third with the bottom two-thirds empty. Use generous font sizes, spacing, and visual elements to fill the canvas.

### Supported Elements

- `<p>`, `<h1>`-`<h6>` — Styled text
- `<ul>`, `<ol>` — Lists (never use manual bullet symbols like bullet, -, *)
- `<b>`, `<strong>`, `<i>`, `<em>`, `<u>` — Inline formatting
- `<span>` — Inline formatting with CSS styles
- `<br>` — Line breaks
- `<div>` with bg/border — Becomes shapes
- `<img>` — Images
- `class="placeholder"` — Reserves space for charts (returns `{ id, x, y, w, h }`)

### Key Text Rules

**All text must be inside `<p>`, `<h1>`-`<h6>`, `<ul>`, or `<ol>` tags:**
- ✅ `<div><p>Text here</p></div>`
- ❌ `<div>Text here</div>` — **Silently ignored in PowerPoint**
- ❌ `<span>Text here</span>` — **Silently ignored in PowerPoint**

**Never use manual bullet symbols (bullet, -, *, etc.)** — use `<ul>` or `<ol>` instead.

**v3 Smart Font Mapping (pass-through + safe fallback):**

html2pptx.js v3 no longer maps all fonts to the same one. Strategy:
1. **PPT-safe fonts** (Corbel, Arial, SimHei, Palatino Linotype, etc. 40+) → **Pass through directly**
2. **macOS-exclusive fonts** (PingFang SC, Hiragino Sans) → Mapped to cross-platform equivalents
3. **Web fonts** (Roboto, Montserrat, Inter, etc. 40+) → Mapped to visually closest PPT-safe font
4. **CSS generic names** (sans-serif/serif) → Corbel / Times New Roman
5. **Unknown fonts** → CJK falls back to fontConfig.cjk; Latin passes through directly

**Write PPT font names directly in HTML**:
```css
font-family: "SimHei", "Microsoft YaHei", sans-serif;  /* CJK heading */
font-family: "Microsoft YaHei", sans-serif;  /* CJK body */
font-family: "Gill Sans MT", "Century Gothic", sans-serif;  /* English heading */
```

**fontConfig is still available** (as CJK/Latin ultimate fallback, optional):
```javascript
const fontConfig = { cjk: 'SimHei', latin: 'Gill Sans MT' };
const result = await html2pptx('slide.html', pptx, { fontConfig });
```

### Styles

- Body must use `display: flex; flex-direction: column;` — without it, multiple direct children stack horizontally
- **Do not use `flex-wrap`** — multi-row layouts must use separate flex containers (html2pptx renderer may lose wrapped content)
- `<span>` supports: `font-weight`, `font-style`, `text-decoration`, `color`, `font-size`, `letter-spacing`; color accepts `rgba()` for transparency
- `<span>` does NOT support: `margin`, `padding`
- `text-transform: uppercase / lowercase / capitalize` works on all text elements and `<span>`
- **Rotated text**: `transform: rotate(-30deg)` or `writing-mode: vertical-rl / vertical-lr`
- Use hex colors with `#` prefix in CSS; use `text-align` for alignment hints

### Shape Styles (DIV elements only)

Backgrounds, borders, and shadows **only work on `<div>`**, not on text elements.

- **Background**: `background-color` on `<div>` only
- **Border**: uniform (`border: 2px solid #333`) or partial (`border-left`, `border-right`, etc.)
- **Border radius**: `border-radius: 8pt` for rounded corners; `50%` for circle; percentages relative to smaller dimension
- **Box shadow**: outer shadows only — `box-shadow: 2px 2px 8px rgba(0,0,0,0.3)`; inset shadows ignored

### Typography Guidelines

Choose font sizes that create clear visual hierarchy. Refer to `design-system.md` for suggested ranges.

**Minimum font size**: Don't go below ~11pt for any text. Prefer ≥13pt for body text.
**Hierarchy principle**: Headings should be noticeably larger and bolder than body text.

### Spacing Guidelines

Use consistent spacing throughout the deck. Refer to `design-system.md` for suggested values. Key principles:
- Page margins: ~48pt left/right, ~40pt top, ≥36pt bottom safe zone
- Be generous with whitespace — but fill the slide; avoid large empty areas

### Color Rules

**All colors must come from the current theme's color scale.** Arbitrary grays unrelated to the primary color are forbidden.

After selecting a theme from `themes.md`, use that theme's complete color scale.

### Image Rules (MANDATORY for decks with 6+ slides)

Every deck with 6+ slides must include real photographs. Images create visual richness and professional quality.

**Image sourcing priority**:
1. **Unsplash** (free, high quality) — use theme's **Image Keywords** from `themes.md`:
   ```bash
   curl -L "https://source.unsplash.com/1920x1080/?keyword1,keyword2" -o cover-bg.jpg
   ```
2. **User-provided images** — local files
3. **Gradient fallback** — if Unsplash fails, generate gradient PNG via Sharp

**Image usage in HTML**:
- **Page background**: `<body style="background-image:url('bg.jpg'); background-size:cover;">`
- **Inline image**: `<img src="photo.jpg" style="width:296pt; height:220pt; object-fit:cover;">`
- **DIV background-image is NOT supported** — only body background-image works

**Mask overlay for photo backgrounds**:
```html
<!-- Background photo on body -->
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-image:url('cover-bg.jpg'); background-size:cover;
             font-family:'PingFang SC','Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">
  <!-- Semi-transparent mask layer (use theme's Mask Color from themes.md) -->
  <div style="position:absolute; top:0; left:0; width:720pt; height:405pt;
              background-color:rgba(18,32,64,0.75);"></div>
  <!-- Content layer above mask -->
  <div style="position:relative; z-index:1; flex:1; display:flex; flex-direction:column;
              justify-content:center; align-items:center;">
    <h1 style="font-size:34pt; font-weight:bold; color:#FFFFFF;">Title Here</h1>
  </div>
</body>
```

**Mask opacity guide**:
- Dark mask (cover/divider): opacity 0.70–0.85 — text clearly readable
- Light mask (content): opacity 0.60–0.80 — retains image visibility

### Icons & Gradients

- **CRITICAL: Never use CSS gradients** — pre-rasterize as PNG with Sharp
- **Icons**: rasterize react-icons SVG to PNG; **Gradients**: rasterize SVG to PNG background

```javascript
// Icon to PNG
const { FaHome } = require('react-icons/fa');
const svgString = ReactDOMServer.renderToStaticMarkup(
  React.createElement(FaHome, { color: '#4472C4', size: '256' })
);
await sharp(Buffer.from(svgString)).png().toFile('home-icon.png');
// In HTML: <img src="home-icon.png" style="width:40pt;height:40pt;">

// Gradient to PNG
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="562.5">
  <defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" style="stop-color:#COLOR1"/>
    <stop offset="100%" style="stop-color:#COLOR2"/>
  </linearGradient></defs>
  <rect width="100%" height="100%" fill="url(#g)"/>
</svg>`;
await sharp(Buffer.from(svg)).png().toFile('gradient-bg.png');
// In HTML: <body style="background-image: url('gradient-bg.png');">
```

---

## Using the html2pptx Library

### Dependencies

Globally installed: `pptxgenjs`, `playwright`, `sharp`

### Basic Usage

```javascript
const pptxgen = require('pptxgenjs');
const html2pptx = require('./html2pptx');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_16x9';  // Must match HTML body dimensions

// Optional: custom font configuration (from themes.md)
const fontConfig = { cjk: 'Microsoft YaHei', latin: 'Gill Sans MT' };

const { slide, placeholders, warnings } = await html2pptx('slide1.html', pptx, { fontConfig });

// If warnings is non-empty, fix the HTML and re-run
if (warnings.length > 0) {
    console.error('Fix overflow issues before saving:', warnings);
    process.exit(1);
}

if (placeholders.length > 0) {
    slide.addChart(pptx.charts.LINE, chartData, placeholders[0]);
}

await pptx.writeFile('output.pptx');
```

### API Reference

```javascript
await html2pptx(htmlFile, pres, options)
```

**Parameters:**
- `htmlFile` (string): Path to HTML file
- `pres` (pptxgen): PptxGenJS instance with layout set
- `options.tmpDir` (string): Temp dir (default: `process.env.TMPDIR || '/tmp'`)
- `options.slide` (object): Existing slide to reuse
- `options.fontConfig` (object): Font mapping config `{ cjk: 'Microsoft YaHei', latin: 'Corbel' }`

**Returns:**
```javascript
{
    slide: pptxgenSlide,
    placeholders: [{ id, x, y, w, h }, ...],
    warnings: string[]   // Overflow and layout suggestions; empty if none
}
```

### Validation

**Blocking errors** (conversion aborted):
1. CSS gradients — must be pre-rasterized as PNG
2. Backgrounds/borders/shadows on text elements (`<p>`, `<h1>`-`<h6>`, etc.)
3. Unwrapped text directly in `<div>`
4. Manual bullet symbols in text elements
5. Font size below 11pt

**Non-blocking warnings** (conversion succeeds, returned in `warnings`):
1. Element out of bounds — extends beyond slide edges
2. Vertical imbalance — content clusters in top 55%
3. Text overlap — text elements overlap each other
4. Character density — exceeds 350 CJK / 550 Latin chars
5. Body-level overflow

**Blocking issues** (overflow, font < 11pt) must be fixed. **Non-blocking warnings** are suggestions — use your judgment.

### Visual Quality Check

After generating all slides, do a quick quality scan:
- Is there visual variety across the deck? (different layouts, backgrounds, card styles)
- Do real photographs appear? (at least on cover + 1 other slide for 6+ slide decks)
- Is there at least one dramatic "rhythm breaker" page?
- Does every slide have a clear visual focal point?

### Working with Placeholders

```javascript
const { slide, placeholders } = await html2pptx('slide.html', pptx);
slide.addChart(pptx.charts.BAR, data, placeholders[0]);

// By ID:
const chartArea = placeholders.find(p => p.id === 'chart-area');
slide.addChart(pptx.charts.LINE, data, chartArea);
```

---

## Using PptxGenJS

### Critical Rules

**NEVER use `#` prefix** with hex colors in PptxGenJS — causes file corruption.
- ✅ `color: "FF0000"`, `fill: { color: "0066CC" }`
- ❌ `color: "#FF0000"`

### Adding Images

```javascript
const imgWidth = 1860, imgHeight = 1519;
const h = 3, w = h * (imgWidth / imgHeight);
slide.addImage({ path: "chart.png", x: (10 - w) / 2, y: 1.5, w, h });
```

### Adding Shapes

```javascript
slide.addShape(pptx.shapes.RECTANGLE, {
    x: 1, y: 1, w: 3, h: 2,
    fill: { color: "4472C4" },
    line: { color: "2E5DA8", width: 2 },
    rectRadius: 0.1  // rounded corners (ROUNDED_RECTANGLE only)
});
```

### Adding Charts

**Time Series Granularity:** `< 30 days` daily | `30-365 days` monthly | `> 365 days` yearly.

#### Bar Chart

```javascript
slide.addChart(pptx.charts.BAR, [{
    name: "Sales 2024",
    labels: ["Q1", "Q2", "Q3", "Q4"],
    values: [4500, 5500, 6200, 7100]
}], {
    ...placeholders[0],
    barDir: 'col',
    showTitle: true, title: 'Quarterly Sales',
    showLegend: false,
    showCatAxisTitle: true, catAxisTitle: 'Quarter',
    showValAxisTitle: true, valAxisTitle: 'Sales ($000s)',
    valAxisMinVal: 0, valAxisMaxVal: 8000, valAxisMajorUnit: 2000,
    chartColors: ["4472C4"]
});
```

#### Line Chart

```javascript
slide.addChart(pptx.charts.LINE, [{
    name: "Temperature",
    labels: ["Jan", "Feb", "Mar", "Apr"],
    values: [32, 35, 42, 55]
}], {
    x: 1, y: 1, w: 8, h: 4,
    lineSize: 4, lineSmooth: true,
    showCatAxisTitle: true, catAxisTitle: 'Month',
    showValAxisTitle: true, valAxisTitle: 'Temp (F)',
    valAxisMinVal: 0, valAxisMaxVal: 60, valAxisMajorUnit: 20,
    chartColors: ["4472C4"]
});
```

#### Pie Chart

```javascript
slide.addChart(pptx.charts.PIE, [{
    name: "Market Share",
    labels: ["Product A", "Product B", "Other"],
    values: [35, 45, 20]
}], {
    x: 2, y: 1, w: 6, h: 4,
    showPercent: true, showLegend: true, legendPos: 'r',
    chartColors: ["4472C4", "ED7D31", "A5A5A5"]
});
```

#### Scatter Chart

```javascript
slide.addChart(pptx.charts.SCATTER, [
    { name: 'X-Axis', values: [10, 15, 20, 12, 18] },
    { name: 'Series 1', values: [20, 25, 30] },
    { name: 'Series 2', values: [18, 22] }
], {
    x: 1, y: 1, w: 8, h: 4,
    lineSize: 0, lineDataSymbol: 'circle', lineDataSymbolSize: 6,
    showCatAxisTitle: true, catAxisTitle: 'X',
    showValAxisTitle: true, valAxisTitle: 'Y',
    chartColors: ["4472C4", "ED7D31"]
});
```

#### Multiple Series

```javascript
slide.addChart(pptx.charts.LINE, [
    { name: "Product A", labels: ["Q1","Q2","Q3","Q4"], values: [10,20,30,40] },
    { name: "Product B", labels: ["Q1","Q2","Q3","Q4"], values: [15,25,20,35] }
], { x: 1, y: 1, w: 8, h: 4, showCatAxisTitle: true, catAxisTitle: 'Quarter',
     showValAxisTitle: true, valAxisTitle: 'Revenue ($M)' });
```

**Chart colors:** no `#` prefix; align with slide palette; strong contrast between adjacent series.

### Adding Tables

```javascript
const tableData = [
    [
        { text: "Product", options: { fill: { color: "4472C4" }, color: "FFFFFF", bold: true } },
        { text: "Revenue", options: { fill: { color: "4472C4" }, color: "FFFFFF", bold: true } },
        { text: "Growth",  options: { fill: { color: "4472C4" }, color: "FFFFFF", bold: true } }
    ],
    ["Product A", "$50M", "+15%"],
    ["Product B", "$35M", "+22%"]
];
slide.addTable(tableData, {
    x: 1, y: 1.5, w: 8, h: 2.5,
    colW: [3, 2.5, 2.5], rowH: [0.5, 0.6, 0.6],
    border: { pt: 1, color: "CCCCCC" },
    align: "center", valign: "middle", fontSize: 14
});
```

**Table options:** `x, y, w, h` | `colW` | `rowH` | `border: { pt, color }` | `fill` | `align` | `valign` | `fontSize` | `autoPage`
