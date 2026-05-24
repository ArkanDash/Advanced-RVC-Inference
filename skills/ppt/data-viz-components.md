# Data Visualization Components — Supplement to components.md

These components use **pure HTML+CSS** to render charts and data visualizations without relying on PptxGenJS chart API. They convert directly through html2pptx into shapes.

**When to use**: Prefer data visualization components whenever the content contains numbers, comparisons, percentages, or structured data. Every PPT with 6+ slides **MUST** include at least 1-2 data visualization pages.

**Selection principle**: Concrete data → prioritize data viz components (bars/tables/funnels/donuts) over plain text lists. Alternate between text pages and data pages for visual rhythm.

For PptxGenJS native charts (BAR/LINE/PIE/DOUGHNUT), use `content-chart-focus` with placeholder + `slide.addChart()` in compile.js.

---

<a id="content-horizontal-bars"></a>
### content-horizontal-bars — Horizontal Bar Comparison

`data | medium | light | none`

Multi-item value comparison. Best for: market share, KPI benchmarks, quarterly comparisons.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Market Share Comparison
    </h2>
  </div>

  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt; gap:20pt;">

    <!-- Bar item: repeat for each data point -->
    <div style="display:flex; align-items:center; gap:12pt;">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:100pt; line-height:1.3; white-space:nowrap;">Product A</p>
      <div style="flex:1; height:24pt; background:${primary-10}; border-radius:4pt;">
        <div style="height:24pt; width:70%; background:${primary-80}; border-radius:4pt;"></div>
      </div>
      <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0; width:48pt; text-align:right; line-height:1;">70%</p>
    </div>

    <div style="display:flex; align-items:center; gap:12pt;">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:100pt; line-height:1.3; white-space:nowrap;">Product B</p>
      <div style="flex:1; height:24pt; background:${primary-10}; border-radius:4pt;">
        <div style="height:24pt; width:52%; background:${primary-60}; border-radius:4pt;"></div>
      </div>
      <p style="font-size:15pt; font-weight:bold; color:${primary-60}; margin:0; width:48pt; text-align:right; line-height:1;">52%</p>
    </div>

    <div style="display:flex; align-items:center; gap:12pt;">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:100pt; line-height:1.3; white-space:nowrap;">Product C</p>
      <div style="flex:1; height:24pt; background:${primary-10}; border-radius:4pt;">
        <div style="height:24pt; width:38%; background:${primary-40}; border-radius:4pt;"></div>
      </div>
      <p style="font-size:15pt; font-weight:bold; color:${primary-40}; margin:0; width:48pt; text-align:right; line-height:1;">38%</p>
    </div>

    <div style="display:flex; align-items:center; gap:12pt;">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:100pt; line-height:1.3; white-space:nowrap;">Product D</p>
      <div style="flex:1; height:24pt; background:${primary-10}; border-radius:4pt;">
        <div style="height:24pt; width:22%; background:${primary-20}; border-radius:4pt;"></div>
      </div>
      <p style="font-size:15pt; font-weight:bold; color:${primary-20}; margin:0; width:48pt; text-align:right; line-height:1;">22%</p>
    </div>

  </div>
</body>
```

**Variation**: Use `${accent}` for the highlighted/top bar to draw attention.

---

<a id="content-stacked-bars"></a>
### content-stacked-bars — Stacked Progress Bars

`data | medium | light | shadow`

Multi-dimension proportional data. Best for: budget allocation, resource distribution, composition analysis.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Budget Allocation by Department
    </h2>
  </div>

  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt; gap:24pt;">

    <div>
      <div style="display:flex; justify-content:space-between; margin-bottom:8pt;">
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.3;">Q1 2024</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1;">$2.4M Total</p>
      </div>
      <div style="display:flex; height:32pt; border-radius:6pt; overflow:hidden;">
        <div style="width:35%; background:${primary-80};"></div>
        <div style="width:25%; background:${primary-60};"></div>
        <div style="width:20%; background:${primary-40};"></div>
        <div style="width:12%; background:${accent};"></div>
        <div style="width:8%; background:${primary-20};"></div>
      </div>
      <div style="display:flex; gap:16pt; margin-top:8pt;">
        <div style="display:flex; align-items:center; gap:4pt;">
          <div style="width:8pt; height:8pt; background:${primary-80}; border-radius:2pt;"></div>
          <p style="font-size:11pt; color:${primary-60}; margin:0; line-height:1;">R&D 35%</p>
        </div>
        <div style="display:flex; align-items:center; gap:4pt;">
          <div style="width:8pt; height:8pt; background:${primary-60}; border-radius:2pt;"></div>
          <p style="font-size:11pt; color:${primary-60}; margin:0; line-height:1;">Marketing 25%</p>
        </div>
        <div style="display:flex; align-items:center; gap:4pt;">
          <div style="width:8pt; height:8pt; background:${primary-40}; border-radius:2pt;"></div>
          <p style="font-size:11pt; color:${primary-60}; margin:0; line-height:1;">Sales 20%</p>
        </div>
        <div style="display:flex; align-items:center; gap:4pt;">
          <div style="width:8pt; height:8pt; background:${accent}; border-radius:2pt;"></div>
          <p style="font-size:11pt; color:${primary-60}; margin:0; line-height:1;">Ops 12%</p>
        </div>
      </div>
    </div>

    <div>
      <div style="display:flex; justify-content:space-between; margin-bottom:8pt;">
        <p style="font-size:15pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.3;">Q2 2024</p>
        <p style="font-size:13pt; color:${primary-40}; margin:0; line-height:1;">$2.8M Total</p>
      </div>
      <div style="display:flex; height:32pt; border-radius:6pt; overflow:hidden;">
        <div style="width:40%; background:${primary-80};"></div>
        <div style="width:22%; background:${primary-60};"></div>
        <div style="width:18%; background:${primary-40};"></div>
        <div style="width:10%; background:${accent};"></div>
        <div style="width:10%; background:${primary-20};"></div>
      </div>
    </div>

  </div>
</body>
```

---

<a id="content-data-table"></a>
### content-data-table — Structured Data Table

`data | medium | light | none`

Multi-row multi-column structured data. Best for: quarterly reports, feature comparisons, competitive analysis.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Quarterly Performance Summary
    </h2>
  </div>

  <div style="flex:1; display:flex; flex-direction:column; justify-content:center; padding:0 48pt;">

    <div style="display:flex; background:${primary-80}; border-radius:8pt 8pt 0 0; padding:12pt 16pt;">
      <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; width:120pt; line-height:1.3;">Department</p>
      <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; width:120pt; line-height:1.3; text-align:right;">Q1</p>
      <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; width:120pt; line-height:1.3; text-align:right;">Q2</p>
      <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; width:120pt; line-height:1.3; text-align:right;">Q3</p>
      <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; width:112pt; line-height:1.3; text-align:right;">Q4</p>
    </div>

    <div style="display:flex; background:${surface}; padding:10pt 16pt; border-left:1pt solid ${primary-10}; border-right:1pt solid ${primary-10};">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:120pt; line-height:1.3;">Engineering</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$1.2M</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$1.5M</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$1.8M</p>
      <p style="font-size:13pt; color:${accent}; margin:0; width:112pt; line-height:1.3; text-align:right; font-weight:bold;">$2.1M</p>
    </div>

    <div style="display:flex; background:${background}; padding:10pt 16pt; border-left:1pt solid ${primary-10}; border-right:1pt solid ${primary-10};">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:120pt; line-height:1.3;">Marketing</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$800K</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$920K</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$1.1M</p>
      <p style="font-size:13pt; color:${accent}; margin:0; width:112pt; line-height:1.3; text-align:right; font-weight:bold;">$1.3M</p>
    </div>

    <div style="display:flex; background:${surface}; padding:10pt 16pt; border-left:1pt solid ${primary-10}; border-right:1pt solid ${primary-10};">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:120pt; line-height:1.3;">Sales</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$2.0M</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$2.3M</p>
      <p style="font-size:13pt; color:${primary-60}; margin:0; width:120pt; line-height:1.3; text-align:right;">$2.5M</p>
      <p style="font-size:13pt; color:${accent}; margin:0; width:112pt; line-height:1.3; text-align:right; font-weight:bold;">$2.8M</p>
    </div>

    <div style="display:flex; background:${primary-10}; padding:10pt 16pt; border-radius:0 0 8pt 8pt;">
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:120pt; line-height:1.3; font-weight:bold;">Total</p>
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:120pt; line-height:1.3; text-align:right; font-weight:bold;">$4.0M</p>
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:120pt; line-height:1.3; text-align:right; font-weight:bold;">$4.7M</p>
      <p style="font-size:13pt; color:${primary-80}; margin:0; width:120pt; line-height:1.3; text-align:right; font-weight:bold;">$5.4M</p>
      <p style="font-size:13pt; color:${accent}; margin:0; width:112pt; line-height:1.3; text-align:right; font-weight:bold;">$6.2M</p>
    </div>

  </div>
</body>
```

---

<a id="content-quadrant-matrix"></a>
### content-quadrant-matrix — 2x2 Quadrant Matrix

`data | medium | light | none`

Two-axis classification framework. Best for: BCG matrix, priority matrix, SWOT, evaluation frameworks.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Priority Matrix
    </h2>
  </div>

  <div style="flex:1; display:flex; padding:8pt 48pt 0 48pt;">

    <div style="width:24pt; display:flex; align-items:center; justify-content:center;">
      <p style="font-size:9pt; color:${primary-40}; margin:0; writing-mode:vertical-rl; transform:rotate(180deg); text-align:center; line-height:1.2;">Impact</p>
    </div>

    <div style="flex:1; display:flex; flex-direction:column;">

      <div style="display:flex; gap:16pt; margin-bottom:16pt;">
        <div style="width:296pt; flex-shrink:0; border-radius:10pt; padding:16pt 20pt; background:${primary-80};">
          <p style="font-size:18pt; font-weight:bold; color:${on-dark}; margin:0 0 6pt 0; line-height:1.25;">Do First</p>
          <p style="font-size:12pt; color:${on-dark-secondary}; margin:0; line-height:1.5;">High Impact + High Effort. Critical items requiring attention.</p>
        </div>
        <div style="width:296pt; flex-shrink:0; border-radius:10pt; padding:16pt 20pt; background:${surface}; border:1.5pt solid ${primary-20};">
          <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 6pt 0; line-height:1.25;">Quick Wins</p>
          <p style="font-size:12pt; color:${primary-60}; margin:0; line-height:1.5;">High Impact + Low Effort. Prioritize for maximum ROI.</p>
        </div>
      </div>

      <div style="display:flex; gap:16pt;">
        <div style="width:296pt; flex-shrink:0; border-radius:10pt; padding:16pt 20pt; border:1.5pt solid ${primary-20};">
          <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0 0 6pt 0; line-height:1.25;">Schedule</p>
          <p style="font-size:12pt; color:${primary-60}; margin:0; line-height:1.5;">Low Impact + High Effort. Plan for later.</p>
        </div>
        <div style="width:296pt; flex-shrink:0; border-radius:10pt; padding:16pt 20pt; background:${primary-90};">
          <p style="font-size:18pt; font-weight:bold; color:${on-dark}; margin:0 0 6pt 0; line-height:1.25;">Eliminate</p>
          <p style="font-size:12pt; color:${on-dark-secondary}; margin:0; line-height:1.5;">Low Impact + Low Effort. Drop these tasks.</p>
        </div>
      </div>

      <div style="margin-top:6pt;">
        <p style="font-size:9pt; font-weight:bold; color:${primary-40}; margin:0; text-align:center;">LOW Effort &#8594; HIGH</p>
      </div>

    </div>
  </div>
</body>
```

---

<a id="content-funnel"></a>
### content-funnel — Sales/Conversion Funnel

`data | medium | light | none`

Multi-stage progressive filtering. Best for: sales pipeline, conversion funnel, qualification process.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Sales Conversion Funnel
    </h2>
  </div>

  <div style="flex:1; display:flex; align-items:center; padding:0 48pt; gap:40pt;">

    <div style="width:320pt; display:flex; flex-direction:column; align-items:center; gap:4pt;">
      <div style="width:280pt; height:36pt; background:${primary-80}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Leads: 10,000</p>
      </div>
      <div style="width:230pt; height:36pt; background:${primary-60}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Qualified: 6,500</p>
      </div>
      <div style="width:180pt; height:36pt; background:${primary-40}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Proposals: 3,200</p>
      </div>
      <div style="width:130pt; height:36pt; background:${accent}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Negotiation: 1,800</p>
      </div>
      <div style="width:80pt; height:36pt; background:${primary-80}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Closed: 850</p>
      </div>
    </div>

    <div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:16pt;">
      <div style="background:${surface}; border-radius:8pt; padding:16pt; border-left:3pt solid ${accent};">
        <p style="font-size:32pt; font-weight:bold; color:${accent}; margin:0; line-height:1; white-space:nowrap;">8.5%</p>
        <p style="font-size:13pt; color:${primary-60}; margin:4pt 0 0 0; line-height:1.4;">Overall Conversion Rate</p>
      </div>
      <div style="background:${surface}; border-radius:8pt; padding:16pt;">
        <p style="font-size:32pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1; white-space:nowrap;">46%</p>
        <p style="font-size:13pt; color:${primary-60}; margin:4pt 0 0 0; line-height:1.4;">Qualified to Proposal Rate</p>
      </div>
    </div>

  </div>
</body>
```

---

<a id="content-before-after"></a>
### content-before-after — Before vs After Comparison

`data | medium | light | none`

Side-by-side comparison panels. Best for: optimization results, before/after, pros/cons.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Before vs After Optimization
    </h2>
  </div>

  <div style="flex:1; display:flex; padding:16pt 48pt 0 48pt; gap:16pt;">

    <!-- Before -->
    <div style="width:296pt; flex-shrink:0; background:${surface}; border-radius:10pt; padding:20pt 24pt;">
      <div style="display:flex; align-items:center; gap:8pt; margin-bottom:16pt;">
        <div style="width:28pt; height:28pt; border-radius:50%; background:${primary-40}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:13pt; color:${on-dark}; margin:0; line-height:1; font-weight:bold;">B</p>
        </div>
        <p style="font-size:18pt; font-weight:bold; color:${primary-40}; margin:0; line-height:1.25;">Before</p>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8pt 0; border-bottom:1pt solid ${primary-10};">
        <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Load Time</p>
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.5;">4.2s</p>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8pt 0; border-bottom:1pt solid ${primary-10};">
        <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Bounce Rate</p>
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.5;">38%</p>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8pt 0;">
        <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Conversion</p>
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.5;">2.1%</p>
      </div>
    </div>

    <!-- After (accent border to highlight) -->
    <div style="width:296pt; flex-shrink:0; background:${surface}; border-radius:10pt; padding:20pt 24pt; border:2pt solid ${accent};">
      <div style="display:flex; align-items:center; gap:8pt; margin-bottom:16pt;">
        <div style="width:28pt; height:28pt; border-radius:50%; background:${accent}; display:flex; align-items:center; justify-content:center;">
          <p style="font-size:13pt; color:${on-dark}; margin:0; line-height:1; font-weight:bold;">A</p>
        </div>
        <p style="font-size:18pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1.25;">After</p>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8pt 0; border-bottom:1pt solid ${primary-10};">
        <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Load Time</p>
        <p style="font-size:13pt; font-weight:bold; color:${accent}; margin:0; line-height:1.5;">1.8s</p>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8pt 0; border-bottom:1pt solid ${primary-10};">
        <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Bounce Rate</p>
        <p style="font-size:13pt; font-weight:bold; color:${accent}; margin:0; line-height:1.5;">22%</p>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8pt 0;">
        <p style="font-size:13pt; color:${primary-60}; margin:0; line-height:1.5;">Conversion</p>
        <p style="font-size:13pt; font-weight:bold; color:${accent}; margin:0; line-height:1.5;">4.7%</p>
      </div>
    </div>

  </div>
</body>
```

---

<a id="content-dashboard"></a>
### content-dashboard — Data Dashboard / KPI Grid

`data | medium | light | shadow`

Multi-metric overview. Best for: dashboards, weekly/monthly reports, KPI summaries.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${surface};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Monthly Performance Dashboard
    </h2>
  </div>

  <!-- Row 1: 3 large stats -->
  <div style="display:flex; gap:16pt; padding:16pt 48pt 0 48pt;">
    <div style="width:192pt; flex-shrink:0; background:${surface-card}; border-radius:10pt; padding:20pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); text-align:center;">
      <p style="font-size:40pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1; white-space:nowrap;">$4.2M</p>
      <p style="font-size:13pt; color:${primary-40}; margin:8pt 0 0 0; line-height:1.4;">Revenue</p>
      <p style="font-size:11pt; color:${accent}; margin:8pt 0 0 0; line-height:1; font-weight:bold;">+12.3% vs last month</p>
    </div>
    <div style="width:192pt; flex-shrink:0; background:${surface-card}; border-radius:10pt; padding:20pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); text-align:center;">
      <p style="font-size:40pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1; white-space:nowrap;">2,847</p>
      <p style="font-size:13pt; color:${primary-40}; margin:8pt 0 0 0; line-height:1.4;">Active Users</p>
      <p style="font-size:11pt; color:${accent}; margin:8pt 0 0 0; line-height:1; font-weight:bold;">+8.7% vs last month</p>
    </div>
    <div style="width:192pt; flex-shrink:0; background:${surface-card}; border-radius:10pt; padding:20pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); text-align:center;">
      <p style="font-size:40pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1; white-space:nowrap;">94.2%</p>
      <p style="font-size:13pt; color:${primary-40}; margin:8pt 0 0 0; line-height:1.4;">Uptime</p>
      <p style="font-size:11pt; color:${accent}; margin:8pt 0 0 0; line-height:1; font-weight:bold;">+0.3% vs last month</p>
    </div>
  </div>

  <!-- Row 2: 4 smaller stats -->
  <div style="display:flex; gap:16pt; padding:16pt 48pt;">
    <div style="width:140pt; flex-shrink:0; background:${surface-card}; border-radius:8pt; padding:14pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); text-align:center;">
      <p style="font-size:32pt; font-weight:bold; color:${primary-60}; margin:0; line-height:1; white-space:nowrap;">156</p>
      <p style="font-size:11pt; color:${primary-40}; margin:6pt 0 0 0; line-height:1.4;">New Signups</p>
    </div>
    <div style="width:140pt; flex-shrink:0; background:${surface-card}; border-radius:8pt; padding:14pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); text-align:center;">
      <p style="font-size:32pt; font-weight:bold; color:${primary-60}; margin:0; line-height:1; white-space:nowrap;">$148</p>
      <p style="font-size:11pt; color:${primary-40}; margin:6pt 0 0 0; line-height:1.4;">Avg Revenue/User</p>
    </div>
    <div style="width:140pt; flex-shrink:0; background:${surface-card}; border-radius:8pt; padding:14pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); text-align:center;">
      <p style="font-size:32pt; font-weight:bold; color:${primary-60}; margin:0; line-height:1; white-space:nowrap;">4.6</p>
      <p style="font-size:11pt; color:${primary-40}; margin:6pt 0 0 0; line-height:1.4;">NPS Score</p>
    </div>
    <div style="width:140pt; flex-shrink:0; background:${surface-card}; border-radius:8pt; padding:14pt; box-shadow:0 3pt 10pt rgba(0,0,0,0.08); text-align:center;">
      <p style="font-size:32pt; font-weight:bold; color:${primary-60}; margin:0; line-height:1; white-space:nowrap;">23ms</p>
      <p style="font-size:11pt; color:${primary-40}; margin:6pt 0 0 0; line-height:1.4;">Avg Response</p>
    </div>
  </div>
</body>
```

---

<a id="content-pyramid"></a>
### content-pyramid — Pyramid / Layered Hierarchy

`data | medium | light | none`

Layered hierarchy. Best for: Maslow pyramid, org levels, tech stack layers, priority hierarchy.

```html
<body style="width:720pt; height:405pt; margin:0; padding:0; overflow:hidden;
             background-color:${background};
             font-family:'Microsoft YaHei',sans-serif;
             display:flex; flex-direction:column;">

  <div style="width:720pt; height:56pt; background:${primary-90};
              display:flex; align-items:center; padding:0 48pt;">
    <h2 style="font-size:22pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1.25;">
      Strategy Implementation Pyramid
    </h2>
  </div>

  <div style="flex:1; display:flex; align-items:center; padding:0 48pt; gap:32pt;">

    <div style="width:360pt; display:flex; flex-direction:column; align-items:center; gap:6pt;">
      <div style="width:140pt; height:40pt; background:${primary-80}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Vision &amp; Mission</p>
      </div>
      <div style="width:200pt; height:40pt; background:${primary-60}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Strategic Objectives</p>
      </div>
      <div style="width:260pt; height:40pt; background:${primary-40}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${on-dark}; margin:0; line-height:1;">Tactical Plans &amp; Programs</p>
      </div>
      <div style="width:320pt; height:40pt; background:${primary-20}; border-radius:6pt; display:flex; align-items:center; justify-content:center;">
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0; line-height:1;">Operations &amp; Daily Execution</p>
      </div>
    </div>

    <div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:20pt;">
      <div style="border-left:2pt solid ${primary-80}; padding-left:12pt;">
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Top Level</p>
        <p style="font-size:12pt; color:${primary-60}; margin:0; line-height:1.4;">Long-term direction, 5-10 year horizon</p>
      </div>
      <div style="border-left:2pt solid ${primary-60}; padding[left:12pt;">
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Strategy</p>
        <p style="font-size:12pt; color:${primary-60}; margin:0; line-height:1.4;">3-5 year measurable targets</p>
      </div>
      <div style="border-left:2pt solid ${primary-40}; padding-left:12pt;">
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Tactics</p>
        <p style="font-size:12pt; color:${primary-60}; margin:0; line-height:1.4;">1-2 year initiatives and projects</p>
      </div>
      <div style="border-left:2pt solid ${primary-20}; padding-left:12pt;">
        <p style="font-size:13pt; font-weight:bold; color:${primary-80}; margin:0 0 4pt 0; line-height:1.3;">Operations</p>
        <p style="font-size:12pt; color:${primary-60}; margin:0; line-height:1.4;">Daily tasks and short-term goals</p>
      </div>
    </div>

  </div>
</body>
```
