/**
 * html2pptx v3 - Convert HTML slide to pptxgenjs slide with positioned elements
 *
 * v3 Changes (2026-03-31):
 *   - Smart font mapping: PPT-safe fonts pass through, macOS/web fonts auto-mapped
 *   - Element boundary checking: warns when elements exceed slide bounds
 *   - z-index sorting: elements rendered in correct visual layer order
 *   - Adaptive width compensation: short text, numbers, headings get scaled compensation
 *   - Height compensation: all text elements get 12% height buffer
 *   - white-space: nowrap support: prevents line-breaking in short labels
 *   - Minimum font size validation: blocks text < 11pt
 *   - Per-slide character count warning
 *   - Vertical balance detection: warns when content clusters in top portion
 *   - Text overlap detection: warns when text elements overlap each other
 *
 * USAGE:
 *   const pptx = new pptxgen();
 *   pptx.layout = 'LAYOUT_16x9';
 *   const { slide, placeholders, warnings } = await html2pptx('slide.html', pptx, { fontConfig });
 *   // If warnings is non-empty → fix HTML and re-run
 *   await pptx.writeFile('output.pptx');
 *
 *   await pptx.writeFile('output.pptx');
 *
 * FEATURES:
 *   - Converts HTML to PowerPoint with accurate positioning
 *   - Supports text, images, shapes, and bullet lists
 *   - Extracts placeholder elements (class="placeholder") with positions
 *   - Handles CSS gradients, borders, and margins
 *
 * VALIDATION:
 *   - Uses body width/height from HTML for viewport sizing
 *   - Throws error if HTML dimensions don't match presentation layout
 *   - Throws error if content overflows body (with overflow details)
 *
 * RETURNS:
 *   { slide, placeholders } where placeholders is an array of { id, x, y, w, h }
 */

const { chromium } = require('playwright');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs');

const PT_PER_PX = 0.75;
const PX_PER_IN = 96;
const EMU_PER_IN = 914400;

// ── v3: Compensation factors ──
const COMPENSATION = {
  HEADING_WIDTH: 0.25, SINGLE_LINE_NARROW: 0.18, SINGLE_LINE_NORMAL: 0.10,
  MULTI_LINE: 0.05, SHORT_TEXT_EXTRA: 0.12, NUMERIC_TEXT_EXTRA: 0.08,
  NOWRAP_EXTRA: 0.15, MAX_WIDTH_FACTOR: 0.40,
  TEXT_HEIGHT: 0.08, LIST_HEIGHT: 0.06,
  MIN_FONT_SIZE_PT: 11, MAX_CHARS_CJK: 350, MAX_CHARS_LATIN: 550,
  VERTICAL_BALANCE_THRESHOLD: 0.55, OVERLAP_TOLERANCE_IN: 0.05, BOUNDS_TOLERANCE_IN: 0.02,
  AUTO_SHORT_TEXT_THRESHOLD: 15,  // chars — auto-detect short text even without explicit nowrap
};

// ── v3: Validation helpers (Node.js scope) ──
function extractTextContent(el) {
  if (typeof el.text === 'string') return el.text;
  if (Array.isArray(el.text)) return el.text.map(r => r.text || '').join('');
  if (Array.isArray(el.items)) return el.items.map(r => r.text || '').join('');
  return '';
}
function getElementLabel(el) {
  const t = extractTextContent(el);
  return t.substring(0, 40) + (t.length > 40 ? '...' : '') || `[${el.type}]`;
}
// Compute PPT-adjusted position for text/list elements (mirrors addElements logic, pre-clamp)
function getAdjustedTextPosition(el, slideWidthIn) {
  const MAX_TEXT_WIDTH_IN = 680 / 72;
  const widthFactor = calculateWidthCompensation(el, slideWidthIn);
  const widthIncrease = el.position.w * widthFactor;
  let adjustedX = el.position.x;
  let adjustedW = el.position.w;
  const align = el.style?.align;
  if (align === 'center') { adjustedX -= widthIncrease / 2; adjustedW += widthIncrease; }
  else if (align === 'right') { adjustedX -= widthIncrease; adjustedW += widthIncrease; }
  else { adjustedW += widthIncrease; }
  if (align === 'center' && /^h[1-6]$/.test(el.type)) {
    const centerX = adjustedX + adjustedW / 2;
    const margin = 0.3;
    const maxExpand = Math.min(centerX - margin, slideWidthIn - centerX - margin);
    if (maxExpand > adjustedW / 2) { adjustedX = centerX - maxExpand; adjustedW = maxExpand * 2; }
  }
  const hComp = el.type === 'list' ? COMPENSATION.LIST_HEIGHT : COMPENSATION.TEXT_HEIGHT;
  const finalW = Math.min(adjustedW, MAX_TEXT_WIDTH_IN);
  return { x: adjustedX, y: el.position.y, w: finalW, h: el.position.h * (1 + hComp) };
}
function checkElementBounds(slideData, sw, sh) {
  const w = []; const tol = COMPENSATION.BOUNDS_TOLERANCE_IN;
  const textTypes = new Set(['p','h1','h2','h3','h4','h5','h6','list']);
  for (const el of slideData.elements) {
    if (!el.position) continue;
    const p = textTypes.has(el.type) ? getAdjustedTextPosition(el, sw) : el.position;
    if (p.x < -tol) w.push(`⚠ BOUNDS: "${getElementLabel(el)}" extends ${(-p.x*72).toFixed(0)}pt beyond LEFT`);
    if (p.y < -tol) w.push(`⚠ BOUNDS: "${getElementLabel(el)}" extends ${(-p.y*72).toFixed(0)}pt beyond TOP`);
    if (p.x+p.w > sw+tol) w.push(`⚠ BOUNDS: "${getElementLabel(el)}" extends ${((p.x+p.w-sw)*72).toFixed(0)}pt beyond RIGHT`);
    if (p.y+p.h > sh+tol) w.push(`⚠ BOUNDS: "${getElementLabel(el)}" extends ${((p.y+p.h-sh)*72).toFixed(0)}pt beyond BOTTOM`);
  }
  return w;
}
function checkVerticalBalance(slideData, sh) {
  const ce = slideData.elements.filter(e => ['p','h1','h2','h3','h4','h5','h6','list'].includes(e.type));
  if (!ce.length) return [];
  const mb = Math.max(...ce.map(e => e.position.y + e.position.h));
  return mb/sh < COMPENSATION.VERTICAL_BALANCE_THRESHOLD
    ? [`⚠ LAYOUT: Content only in top ${(mb/sh*100).toFixed(0)}% — consider vertical centering`] : [];
}
function checkTextOverlaps(slideData) {
  const w = []; const te = slideData.elements.filter(e => ['p','h1','h2','h3','h4','h5','h6','list'].includes(e.type));
  for (let i=0;i<te.length;i++) for (let j=i+1;j<te.length;j++) {
    const a=te[i].position, b=te[j].position, tol=COMPENSATION.OVERLAP_TOLERANCE_IN;
    const ow=Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x);
    const oh=Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y);
    if(ow>tol&&oh>tol) w.push(`⚠ OVERLAP: "${getElementLabel(te[i])}" overlaps "${getElementLabel(te[j])}"`);
  }
  return w;
}
function checkMinFontSize(slideData) {
  const e = [];
  for (const el of slideData.elements) {
    if (!['p','h1','h2','h3','h4','h5','h6','list'].includes(el.type)) continue;
    const fs = el.style?.fontSize || 0;
    if (fs > 0 && fs < COMPENSATION.MIN_FONT_SIZE_PT)
      e.push(`Text "${getElementLabel(el)}" is ${fs.toFixed(1)}pt — min is ${COMPENSATION.MIN_FONT_SIZE_PT}pt`);
  }
  return e;
}
function checkCharCount(slideData) {
  let total=0, cjk=false;
  for (const el of slideData.elements) {
    const t=extractTextContent(el); total+=t.length;
    if(/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(t)) cjk=true;
  }
  const lim = cjk ? COMPENSATION.MAX_CHARS_CJK : COMPENSATION.MAX_CHARS_LATIN;
  return total>lim ? [`⚠ DENSITY: ${total} chars (limit ${lim}) — split into multiple slides`] : [];
}
function calculateWidthCompensation(el, slideWidthIn) {
  const isH = /^h[1-6]$/.test(el.type);
  const txt = extractTextContent(el);
  const lh = el.style?.lineSpacing || (el.style?.fontSize||16)*1.2;
  const single = el.position.h <= lh*1.5/72;
  let f = isH ? COMPENSATION.HEADING_WIDTH : single ? (el.position.w < slideWidthIn/3 ? COMPENSATION.SINGLE_LINE_NARROW : COMPENSATION.SINGLE_LINE_NORMAL) : COMPENSATION.MULTI_LINE;
  if (txt.length>0 && txt.length<10) f += COMPENSATION.SHORT_TEXT_EXTRA;
  if (/^[\d\s\.\,\-\/\+\%\$\#\@\!\?\:\;\(\)\[\]]+$/.test(txt)) f += COMPENSATION.NUMERIC_TEXT_EXTRA;
  if (el.noWrap) f += COMPENSATION.NOWRAP_EXTRA;
  // v4: Auto-detect short text that should not wrap even without explicit nowrap
  if (!el.noWrap && single && txt.length >= 10 && txt.length < COMPENSATION.AUTO_SHORT_TEXT_THRESHOLD) {
    f += COMPENSATION.SHORT_TEXT_EXTRA * 0.5; // Half bonus for auto-detected short text
  }
  // v4: Auto-detect card titles (>=18pt, single line, <20 chars) — treat like heading
  const fs_ = el.style?.fontSize || 16;
  if (!isH && single && fs_ >= 18 && txt.length < 20) {
    f = Math.max(f, COMPENSATION.HEADING_WIDTH * 0.8); // At least 80% of heading compensation
  }
  return Math.min(f, COMPENSATION.MAX_WIDTH_FACTOR);
}

// ── Emphasis font: apply to bold numeric text (post-extraction, Node.js scope) ──
// Matches text that is purely numeric/symbolic (KPI values, percentages, currency, etc.)
const NUMERIC_EMPHASIS_RE = /^[\d\s.,\-\/+%$#@!?:;()[\]]+$/;
function applyEmphasisFont(slideData, emphasisFont) {
  for (const el of slideData.elements) {
    if (!['p','h1','h2','h3','h4','h5','h6'].includes(el.type)) continue;
    if (typeof el.text === 'string') {
      // Plain text element — bold is on el.style
      if (el.style?.bold && NUMERIC_EMPHASIS_RE.test(el.text.trim())) {
        el.style.fontFace = emphasisFont;
      }
    } else if (Array.isArray(el.text)) {
      // Runs — bold may be per-run (inline formatting)
      for (const run of el.text) {
        if (run.options?.bold && NUMERIC_EMPHASIS_RE.test(run.text.trim())) {
          run.options.fontFace = emphasisFont;
        }
      }
    }
  }
}

// Helper: Fix image path if file extension doesn't match actual format
function fixImageExtension(imagePath, tmpDir) {
  try {
    const fd = fs.openSync(imagePath, 'r');
    const buf = Buffer.alloc(12);
    fs.readSync(fd, buf, 0, 12, 0);
    fs.closeSync(fd);

    let actualExt = null;
    if (buf[0] === 0xFF && buf[1] === 0xD8) actualExt = '.jpg';
    else if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4E && buf[3] === 0x47) actualExt = '.png';
    else if (buf[0] === 0x47 && buf[1] === 0x49 && buf[2] === 0x46) actualExt = '.gif';
    else if (buf[0] === 0x52 && buf[1] === 0x49 && buf[8] === 0x57 && buf[9] === 0x45) actualExt = '.webp';

    if (!actualExt) return imagePath;

    const currentExt = path.extname(imagePath).toLowerCase();
    if (currentExt === actualExt || (currentExt === '.jpeg' && actualExt === '.jpg') || (currentExt === '.jpg' && actualExt === '.jpeg')) {
      return imagePath;
    }

    // Extension mismatch: copy with correct extension
    const fixedPath = path.join(tmpDir, path.basename(imagePath, currentExt) + actualExt);
    fs.copyFileSync(imagePath, fixedPath);
    return fixedPath;
  } catch (e) {
    return imagePath;
  }
}

// Helper: Get body dimensions and check for overflow
async function getBodyDimensions(page) {
  const bodyDimensions = await page.evaluate(() => {
    const body = document.body;
    const style = window.getComputedStyle(body);

    return {
      width: parseFloat(style.width),
      height: parseFloat(style.height),
      scrollWidth: body.scrollWidth,
      scrollHeight: body.scrollHeight
    };
  });

  const errors = [];
  const widthOverflowPx = Math.max(0, bodyDimensions.scrollWidth - bodyDimensions.width - 1);
  const heightOverflowPx = Math.max(0, bodyDimensions.scrollHeight - bodyDimensions.height - 1);

  const widthOverflowPt = widthOverflowPx * PT_PER_PX;
  const heightOverflowPt = heightOverflowPx * PT_PER_PX;

  if (widthOverflowPt > 0 || heightOverflowPt > 0) {
    const directions = [];
    if (widthOverflowPt > 0) directions.push(`${widthOverflowPt.toFixed(1)}pt horizontally`);
    if (heightOverflowPt > 0) directions.push(`${heightOverflowPt.toFixed(1)}pt vertically`);
    const reminder = heightOverflowPt > 0 ? ' (Remember: leave 0.5" margin at bottom of slide)' : '';
    errors.push(`HTML content overflows body by ${directions.join(' and ')}${reminder}`);
  }

  return { ...bodyDimensions, errors };
}

// Helper: Validate dimensions match presentation layout
function validateDimensions(bodyDimensions, pres) {
  const errors = [];
  const widthInches = bodyDimensions.width / PX_PER_IN;
  const heightInches = bodyDimensions.height / PX_PER_IN;

  if (pres.presLayout) {
    const layoutWidth = pres.presLayout.width / EMU_PER_IN;
    const layoutHeight = pres.presLayout.height / EMU_PER_IN;

    if (Math.abs(layoutWidth - widthInches) > 0.1 || Math.abs(layoutHeight - heightInches) > 0.1) {
      errors.push(
        `HTML dimensions (${widthInches.toFixed(1)}" × ${heightInches.toFixed(1)}") ` +
        `don't match presentation layout (${layoutWidth.toFixed(1)}" × ${layoutHeight.toFixed(1)}")`
      );
    }
  }
  return errors;
}

function validateTextBoxPosition(slideData, bodyDimensions) {
  const errors = [];
  const slideHeightInches = bodyDimensions.height / PX_PER_IN;
  const minBottomMargin = 0.5; // 0.5 inches from bottom

  for (const el of slideData.elements) {
    // Check text elements (p, h1-h6, list)
    if (['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'list'].includes(el.type)) {
      const fontSize = el.style?.fontSize || 0;
      const bottomEdge = el.position.y + el.position.h;
      const distanceFromBottom = slideHeightInches - bottomEdge;

      if (fontSize > 12 && distanceFromBottom < minBottomMargin) {
        const getText = () => {
          if (typeof el.text === 'string') return el.text;
          if (Array.isArray(el.text)) return el.text.find(t => t.text)?.text || '';
          if (Array.isArray(el.items)) return el.items.find(item => item.text)?.text || '';
          return '';
        };
        const textPrefix = getText().substring(0, 50) + (getText().length > 50 ? '...' : '');

        errors.push(
          `Text box "${textPrefix}" ends too close to bottom edge ` +
          `(${distanceFromBottom.toFixed(2)}" from bottom, minimum ${minBottomMargin}" required)`
        );
      }
    }
  }

  return errors;
}

// Helper: Add background to slide
async function addBackground(slideData, targetSlide, pres, tmpDir) {
  if (slideData.background.type === 'image' && slideData.background.path) {
    let imagePath = slideData.background.path.startsWith('file://')
      ? slideData.background.path.replace('file://', '')
      : slideData.background.path;
    // PptxGenJS slide.background = { path } is unreliable for local files;
    // use addImage at (0,0) covering the full slide instead.
    const slideW = pres.presLayout ? pres.presLayout.width / EMU_PER_IN : 10;
    const slideH = pres.presLayout ? pres.presLayout.height / EMU_PER_IN : 5.625;
    targetSlide.addImage({
      path: fixImageExtension(imagePath, tmpDir),
      x: 0, y: 0, w: slideW, h: slideH
    });
  } else if (slideData.background.type === 'color' && slideData.background.value) {
    targetSlide.background = { color: slideData.background.value };
  }
}

// Helper: Add elements to slide
function addElements(slideData, targetSlide, pres, tmpDir) {
  const slideWidthIn = pres.presLayout ? pres.presLayout.width / EMU_PER_IN : 10;
  const slideHeightIn = pres.presLayout ? pres.presLayout.height / EMU_PER_IN : 5.625;
  const MAX_TEXT_WIDTH_IN = 680 / 72; // ~9.44in — cap after compensation to avoid overflow

  // v3: Sort by z-index for correct visual layering
  const sortedElements = [...slideData.elements].sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));

  for (const el of sortedElements) {
    if (el.type === 'image') {
      let imagePath = el.src.startsWith('file://') ? el.src.replace('file://', '') : el.src;
      targetSlide.addImage({
        path: fixImageExtension(imagePath, tmpDir),
        x: el.position.x, y: el.position.y, w: el.position.w, h: el.position.h
      });
    } else if (el.type === 'line') {
      targetSlide.addShape(pres.ShapeType.line, {
        x: el.x1, y: el.y1, w: el.x2 - el.x1, h: el.y2 - el.y1,
        line: { color: el.color, width: el.width }
      });
    } else if (el.type === 'shape') {
      const shapeOptions = {
        x: el.position.x, y: el.position.y, w: el.position.w, h: el.position.h,
        shape: el.shape.rectRadius > 0 ? pres.ShapeType.roundRect : pres.ShapeType.rect
      };
      if (el.shape.fill) {
        shapeOptions.fill = { color: el.shape.fill };
        if (el.shape.transparency != null) shapeOptions.fill.transparency = el.shape.transparency;
      }
      if (el.shape.line) shapeOptions.line = el.shape.line;
      if (el.shape.rectRadius > 0) shapeOptions.rectRadius = el.shape.rectRadius;
      if (el.shape.shadow) shapeOptions.shadow = el.shape.shadow;
      targetSlide.addText(el.text || '', shapeOptions);
    } else if (el.type === 'list') {
      // v3: Height compensation for lists
      let adjustedH = el.position.h * (1 + COMPENSATION.LIST_HEIGHT);
      // Clamp list height to slide bottom
      if (el.position.y + adjustedH > slideHeightIn) adjustedH = slideHeightIn - el.position.y;
      if (adjustedH < 0.1) adjustedH = 0.1;
      // Clamp list width to slide right edge
      let listX = el.position.x;
      let listW = el.position.w;
      if (listX + listW > slideWidthIn) listW = slideWidthIn - listX;
      if (listW < 0.1) listW = 0.1;
      if (listW > MAX_TEXT_WIDTH_IN) listW = MAX_TEXT_WIDTH_IN;
      const listOptions = {
        x: listX, y: el.position.y, w: listW, h: adjustedH,
        fontSize: el.style.fontSize, fontFace: el.style.fontFace, color: el.style.color,
        align: el.style.align, valign: 'top', charSpacing: el.style.charSpacing,
        lineSpacing: el.style.lineSpacing, paraSpaceBefore: el.style.paraSpaceBefore,
        paraSpaceAfter: el.style.paraSpaceAfter, margin: el.style.margin
      };
      if (el.style.margin) listOptions.margin = el.style.margin;
      targetSlide.addText(el.items, listOptions);
    } else {
      // ── Text elements (p, h1-h6) with v3 adaptive compensation ──
      const widthFactor = calculateWidthCompensation(el, slideWidthIn);
      const widthIncrease = el.position.w * widthFactor;

      let adjustedX = el.position.x;
      let adjustedW = el.position.w;
      const align = el.style.align;

      if (align === 'center') {
        adjustedX -= widthIncrease / 2;
        adjustedW += widthIncrease;
      } else if (align === 'right') {
        adjustedX -= widthIncrease;
        adjustedW += widthIncrease;
      } else {
        adjustedW += widthIncrease;
      }

      // v3: Height compensation for all text
      let adjustedH = el.position.h * (1 + COMPENSATION.TEXT_HEIGHT);
      // Clamp height to slide bottom
      if (el.position.y + adjustedH > slideHeightIn) adjustedH = slideHeightIn - el.position.y;
      if (adjustedH < 0.1) adjustedH = 0.1;

      // Centered headings: expand width symmetrically for PPT center alignment
      if (el.style.align === 'center' && /^h[1-6]$/.test(el.type)) {
        const centerX = adjustedX + adjustedW / 2;
        const margin = 0.3;
        const maxExpand = Math.min(centerX - margin, slideWidthIn - centerX - margin);
        if (maxExpand > adjustedW / 2) {
          adjustedX = centerX - maxExpand;
          adjustedW = maxExpand * 2;
        }
      }

      // Clamp to slide bounds (safety net)
      if (adjustedX < 0) { adjustedW += adjustedX; adjustedX = 0; }
      if (adjustedX + adjustedW > slideWidthIn) adjustedW = slideWidthIn - adjustedX;
      if (adjustedW < 0.1) adjustedW = 0.1;
      // Cap at 680pt to prevent over-expansion from width compensation
      if (adjustedW > MAX_TEXT_WIDTH_IN) adjustedW = MAX_TEXT_WIDTH_IN;

      const textOptions = {
        x: adjustedX, y: el.position.y, w: adjustedW, h: adjustedH,
        fontSize: el.style.fontSize, fontFace: el.style.fontFace, color: el.style.color,
        bold: el.style.bold, italic: el.style.italic, underline: el.style.underline,
        valign: 'top', charSpacing: el.style.charSpacing, lineSpacing: el.style.lineSpacing,
        paraSpaceBefore: el.style.paraSpaceBefore, paraSpaceAfter: el.style.paraSpaceAfter,
        inset: 0
      };
      if (el.style.align) textOptions.align = el.style.align;
      if (el.style.margin) textOptions.margin = el.style.margin;
      if (el.style.rotate !== undefined) textOptions.rotate = el.style.rotate;
      if (el.style.transparency !== null && el.style.transparency !== undefined) textOptions.transparency = el.style.transparency;
      if (el.noWrap) textOptions.wrap = false;

      targetSlide.addText(el.text, textOptions);
    }
  }
}

// Helper: Extract slide data from HTML page
async function extractSlideData(page) {
  return await page.evaluate(() => {
    const PT_PER_PX = 0.75;
    const PX_PER_IN = 96;

    // Fonts that are single-weight and should not have bold applied
    // (applying bold causes PowerPoint to use faux bold which makes text wider)
    const SINGLE_WEIGHT_FONTS = ['impact'];

    // Helper: Check if a font should skip bold formatting
    const shouldSkipBold = (fontFamily) => {
      if (!fontFamily) return false;
      const normalizedFont = fontFamily.toLowerCase().replace(/['"]/g, '').split(',')[0].trim();
      return SINGLE_WEIGHT_FONTS.includes(normalizedFont);
    };

    // Known CJK font name fragments (lowercase) — presence means the element targets CJK text
    const CJK_FONT_FRAGMENTS = [
      'yahei', '雅黑', 'simhei', '黑体', 'simsun', '宋体', 'kaiti', '楷体',
      'fangsong', '仿宋', 'pingfang', 'hiragino', 'noto sans cjk', 'noto sans sc',
      'noto sans tc', 'noto sans hk', 'source han sans', '思源黑体', '思源宋体',
      'wenquanyi', 'arial unicode', 'yugothic', 'meiryo', 'ms gothic', 'ms mincho',
      'malgun gothic', 'apple sd gothic', 'heiti', 'songti', 'wawati', 'weibei',
      'libian', 'xingkai', 'baoli', 'yuanti', 'dengxian', '等线', 'stxihei',
      'stheiti', 'stkaiti', 'stsong', 'stfangsong'
    ];

    // Detect if text content contains CJK characters
    const hasCJKChars = (text) => /[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(text);

    // v3: Smart font mapping — PPT-safe fonts pass through, others get mapped
    // fontConfig from caller is still respected as ultimate fallback for CJK/Latin defaults
    const _fc = window.__FONT_CONFIG__ || {};

    // macOS-only / web fonts → cross-platform PPT-safe equivalents
    const FONT_FALLBACK_MAP = {
      'pingfang sc': 'Microsoft YaHei', 'pingfang tc': 'Microsoft YaHei', 'pingfang hk': 'Microsoft YaHei',
      'hiragino sans': 'Microsoft YaHei', 'hiragino sans gb': 'Microsoft YaHei',
      'hiragino mincho pron': 'SimSun', 'hiragino maru gothic pro': 'Microsoft YaHei',
      'heiti sc': 'SimHei', 'heiti tc': 'SimHei', 'songti sc': 'SimSun',
      'stxihei': 'Microsoft YaHei', 'stheiti': 'SimHei', 'stkaiti': 'KaiTi',
      'stsong': 'SimSun', 'stfangsong': 'FangSong', 'apple sd gothic neo': 'Microsoft YaHei',
      'noto sans sc': 'Microsoft YaHei', 'noto sans tc': 'Microsoft YaHei',
      'noto sans cjk sc': 'Microsoft YaHei', 'noto serif sc': 'SimSun',
      'source han sans sc': 'Microsoft YaHei', 'source han serif sc': 'SimSun',
      'source han sans': 'Microsoft YaHei', 'source han serif': 'SimSun',
      '思源黑体': 'Microsoft YaHei', '思源宋体': 'SimSun',
      '阿里巴巴普惠体': 'Microsoft YaHei',
      'system-ui': 'Century Gothic', '-apple-system': 'Century Gothic', 'blinkmacsystemfont': 'Century Gothic',
      'segoe ui': 'Century Gothic', 'helvetica neue': 'Arial', 'helvetica': 'Arial',
      'sans-serif': 'Century Gothic', 'serif': 'Times New Roman', 'monospace': 'Courier New',
      'inter': 'Century Gothic', 'roboto': 'Arial', 'roboto slab': 'Rockwell',
      'open sans': 'Century Gothic', 'lato': 'Century Gothic', 'montserrat': 'Century Gothic',
      'poppins': 'Candara', 'raleway': 'Century Gothic', 'nunito': 'Candara',
      'nunito sans': 'Candara', 'source sans pro': 'Corbel', 'source sans 3': 'Corbel',
      'source serif pro': 'Georgia', 'work sans': 'Century Gothic', 'dm sans': 'Century Gothic',
      'space grotesk': 'Century Gothic', 'plus jakarta sans': 'Candara',
      'manrope': 'Corbel', 'fira sans': 'Corbel', 'playfair display': 'Georgia',
      'merriweather': 'Georgia', 'libre baskerville': 'Georgia',
      'pt sans': 'Corbel', 'pt serif': 'Constantia', 'ubuntu': 'Century Gothic',
    };

    // PPT-safe fonts — pass through directly without mapping
    const PPT_SAFE_FONTS = new Set([
      'microsoft yahei', '微软雅黑', 'simhei', '黑体', 'simsun', '宋体',
      'kaiti', '楷体', 'simkai', 'fangsong', '仿宋', 'dengxian', '等线',
      'calibri', 'arial', 'arial black', 'arial narrow',
      'times new roman', 'georgia', 'gill sans mt',
      'century gothic', 'palatino linotype', 'palatino',
      'trebuchet ms', 'garamond', 'rockwell', 'candara', 'corbel',
      'constantia', 'cambria', 'book antiqua', 'courier new',
      'verdana', 'tahoma', 'impact', 'comic sans ms',
      'lucida sans', 'franklin gothic medium', 'bodoni mt',
      'copperplate gothic', 'tw cen mt', 'century schoolbook',
    ]);

    const mapFontFace = (fontFamily, textContent = '') => {
      if (!fontFamily) {
        return hasCJKChars(textContent) ? (_fc.cjk || 'Microsoft YaHei') : (_fc.latin || 'Century Gothic');
      }
      const fonts = fontFamily.split(',').map(f => f.trim().replace(/['"]/g, ''));
      for (const font of fonts) {
        const lower = font.toLowerCase();
        const isCJKFont = CJK_FONT_FRAGMENTS.some(f => lower.includes(f));
        if (PPT_SAFE_FONTS.has(lower)) {
          // CJK font specified but text has no CJK characters → use Latin font to avoid
          // mixing Century Gothic and YaHei for English-only content
          if (isCJKFont && !hasCJKChars(textContent)) return _fc.latin || 'Century Gothic';
          return font;
        }
        if (FONT_FALLBACK_MAP[lower]) {
          const mapped = FONT_FALLBACK_MAP[lower];
          const mappedIsCJK = CJK_FONT_FRAGMENTS.some(f => mapped.toLowerCase().includes(f));
          // If the mapped font is a non-CJK font but the text has CJK chars, keep looking
          // (e.g. font-family:'Inter','Microsoft YaHei' with Chinese text should use YaHei, not Century Gothic)
          if (!mappedIsCJK && hasCJKChars(textContent)) continue;
          return mapped;
        }
        if (['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy'].includes(lower)) continue;
        if (isCJKFont || hasCJKChars(textContent)) return _fc.cjk || 'Microsoft YaHei';
        return font; // unknown non-CJK font: pass through
      }
      return hasCJKChars(textContent) ? (_fc.cjk || 'Microsoft YaHei') : (_fc.latin || 'Century Gothic');
    };

    // Unit conversion helpers
    const pxToInch = (px) => px / PX_PER_IN;
    const pxToPoints = (pxStr) => parseFloat(pxStr) * PT_PER_PX;
    const rgbToHex = (rgbStr) => {
      // Handle transparent backgrounds by defaulting to white
      if (rgbStr === 'rgba(0, 0, 0, 0)' || rgbStr === 'transparent') return 'FFFFFF';

      const match = rgbStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (!match) return 'FFFFFF';
      return match.slice(1).map(n => parseInt(n).toString(16).padStart(2, '0')).join('');
    };

    const extractAlpha = (rgbStr) => {
      const match = rgbStr.match(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/);
      if (!match || !match[4]) return null;
      const alpha = parseFloat(match[4]);
      return Math.round((1 - alpha) * 100);
    };

    const applyTextTransform = (text, textTransform) => {
      if (textTransform === 'uppercase') return text.toUpperCase();
      if (textTransform === 'lowercase') return text.toLowerCase();
      if (textTransform === 'capitalize') {
        return text.replace(/\b\w/g, c => c.toUpperCase());
      }
      return text;
    };

    // Extract rotation angle from CSS transform and writing-mode
    const getRotation = (transform, writingMode) => {
      let angle = 0;

      // Handle writing-mode first
      // PowerPoint: 90° = text rotated 90° clockwise (reads top to bottom, letters upright)
      // PowerPoint: 270° = text rotated 270° clockwise (reads bottom to top, letters upright)
      if (writingMode === 'vertical-rl') {
        // vertical-rl alone = text reads top to bottom = 90° in PowerPoint
        angle = 90;
      } else if (writingMode === 'vertical-lr') {
        // vertical-lr alone = text reads bottom to top = 270° in PowerPoint
        angle = 270;
      }

      // Then add any transform rotation
      if (transform && transform !== 'none') {
        // Try to match rotate() function
        const rotateMatch = transform.match(/rotate\((-?\d+(?:\.\d+)?)deg\)/);
        if (rotateMatch) {
          angle += parseFloat(rotateMatch[1]);
        } else {
          // Browser may compute as matrix - extract rotation from matrix
          const matrixMatch = transform.match(/matrix\(([^)]+)\)/);
          if (matrixMatch) {
            const values = matrixMatch[1].split(',').map(parseFloat);
            // matrix(a, b, c, d, e, f) where rotation = atan2(b, a)
            const matrixAngle = Math.atan2(values[1], values[0]) * (180 / Math.PI);
            angle += Math.round(matrixAngle);
          }
        }
      }

      // Normalize to 0-359 range
      angle = angle % 360;
      if (angle < 0) angle += 360;

      return angle === 0 ? null : angle;
    };

    // Get position/dimensions accounting for rotation
    const getPositionAndSize = (el, rect, rotation) => {
      if (rotation === null) {
        return { x: rect.left, y: rect.top, w: rect.width, h: rect.height };
      }

      // For 90° or 270° rotations, swap width and height
      // because PowerPoint applies rotation to the original (unrotated) box
      const isVertical = rotation === 90 || rotation === 270;

      if (isVertical) {
        // The browser shows us the rotated dimensions (tall box for vertical text)
        // But PowerPoint needs the pre-rotation dimensions (wide box that will be rotated)
        // So we swap: browser's height becomes PPT's width, browser's width becomes PPT's height
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        return {
          x: centerX - rect.height / 2,
          y: centerY - rect.width / 2,
          w: rect.height,
          h: rect.width
        };
      }

      // For other rotations, use element's offset dimensions
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      return {
        x: centerX - el.offsetWidth / 2,
        y: centerY - el.offsetHeight / 2,
        w: el.offsetWidth,
        h: el.offsetHeight
      };
    };

    // Parse CSS box-shadow into PptxGenJS shadow properties
    const parseBoxShadow = (boxShadow) => {
      if (!boxShadow || boxShadow === 'none') return null;

      // Browser computed style format: "rgba(0, 0, 0, 0.3) 2px 2px 8px 0px [inset]"
      // CSS format: "[inset] 2px 2px 8px 0px rgba(0, 0, 0, 0.3)"

      const insetMatch = boxShadow.match(/inset/);

      // IMPORTANT: PptxGenJS/PowerPoint doesn't properly support inset shadows
      // Only process outer shadows to avoid file corruption
      if (insetMatch) return null;

      // Extract color first (rgba or rgb at start)
      const colorMatch = boxShadow.match(/rgba?\([^)]+\)/);

      // Extract numeric values (handles both px and pt units)
      const parts = boxShadow.match(/([-\d.]+)(px|pt)/g);

      if (!parts || parts.length < 2) return null;

      const offsetX = parseFloat(parts[0]);
      const offsetY = parseFloat(parts[1]);
      const blur = parts.length > 2 ? parseFloat(parts[2]) : 0;

      // Calculate angle from offsets (in degrees, 0 = right, 90 = down)
      let angle = 0;
      if (offsetX !== 0 || offsetY !== 0) {
        angle = Math.atan2(offsetY, offsetX) * (180 / Math.PI);
        if (angle < 0) angle += 360;
      }

      // Calculate offset distance (hypotenuse)
      const offset = Math.sqrt(offsetX * offsetX + offsetY * offsetY) * PT_PER_PX;

      // Extract opacity from rgba
      let opacity = 0.5;
      if (colorMatch) {
        const opacityMatch = colorMatch[0].match(/[\d.]+\)$/);
        if (opacityMatch) {
          opacity = parseFloat(opacityMatch[0].replace(')', ''));
        }
      }

      return {
        type: 'outer',
        angle: Math.round(angle),
        blur: blur * 0.75, // Convert to points
        color: colorMatch ? rgbToHex(colorMatch[0]) : '000000',
        offset: offset,
        opacity
      };
    };

    // Parse inline formatting tags (<b>, <i>, <u>, <strong>, <em>, <span>) into text runs
    const parseInlineFormatting = (element, baseOptions = {}, runs = [], baseTextTransform = (x) => x) => {
      let prevNodeIsText = false;

      element.childNodes.forEach((node) => {
        let textTransform = baseTextTransform;

        const isText = node.nodeType === Node.TEXT_NODE || node.tagName === 'BR';
        if (isText) {
          const text = node.tagName === 'BR' ? '\n' : textTransform(node.textContent.replace(/\s+/g, ' '));
          const prevRun = runs[runs.length - 1];
          if (prevNodeIsText && prevRun) {
            prevRun.text += text;
          } else {
            runs.push({ text, options: { ...baseOptions } });
          }

        } else if (node.nodeType === Node.ELEMENT_NODE && node.textContent.trim()) {
          const options = { ...baseOptions };
          const computed = window.getComputedStyle(node);

          // Handle inline elements with computed styles
          if (node.tagName === 'SPAN' || node.tagName === 'B' || node.tagName === 'STRONG' || node.tagName === 'I' || node.tagName === 'EM' || node.tagName === 'U') {
            const isBold = computed.fontWeight === 'bold' || parseInt(computed.fontWeight) >= 600;
            if (isBold && !shouldSkipBold(computed.fontFamily)) options.bold = true;
            if (computed.fontStyle === 'italic') options.italic = true;
            if (computed.textDecoration && computed.textDecoration.includes('underline')) options.underline = true;
            if (computed.color && computed.color !== 'rgb(0, 0, 0)') {
              options.color = rgbToHex(computed.color);
              const transparency = extractAlpha(computed.color);
              if (transparency !== null) options.transparency = transparency;
            }
            if (computed.fontSize) options.fontSize = pxToPoints(computed.fontSize);
            if (computed.letterSpacing && computed.letterSpacing !== 'normal') options.charSpacing = pxToPoints(computed.letterSpacing);

            // Apply text-transform on the span element itself
            if (computed.textTransform && computed.textTransform !== 'none') {
              const transformStr = computed.textTransform;
              textTransform = (text) => applyTextTransform(text, transformStr);
            }

            // Validate: Check for margins on inline elements
            if (computed.marginLeft && parseFloat(computed.marginLeft) > 0) {
              errors.push(`Inline element <${node.tagName.toLowerCase()}> has margin-left which is not supported in PowerPoint. Remove margin from inline elements.`);
            }
            if (computed.marginRight && parseFloat(computed.marginRight) > 0) {
              errors.push(`Inline element <${node.tagName.toLowerCase()}> has margin-right which is not supported in PowerPoint. Remove margin from inline elements.`);
            }
            if (computed.marginTop && parseFloat(computed.marginTop) > 0) {
              errors.push(`Inline element <${node.tagName.toLowerCase()}> has margin-top which is not supported in PowerPoint. Remove margin from inline elements.`);
            }
            if (computed.marginBottom && parseFloat(computed.marginBottom) > 0) {
              errors.push(`Inline element <${node.tagName.toLowerCase()}> has margin-bottom which is not supported in PowerPoint. Remove margin from inline elements.`);
            }

            // Recursively process the child node. This will flatten nested spans into multiple runs.
            parseInlineFormatting(node, options, runs, textTransform);
          } else {
            // Unknown inline element (e.g. <a>, <code>, <sub>, <sup>, <mark>) —
            // extract its text content as plain text to avoid silent data loss.
            const text = textTransform(node.textContent.replace(/\s+/g, ' '));
            if (text.trim()) runs.push({ text, options: { ...baseOptions } });
          }
        }

        prevNodeIsText = isText;
      });

      // Trim leading space from first run and trailing space from last run
      if (runs.length > 0) {
        runs[0].text = runs[0].text.replace(/^\s+/, '');
        runs[runs.length - 1].text = runs[runs.length - 1].text.replace(/\s+$/, '');
      }

      return runs.filter(r => r.text.length > 0);
    };

    // Extract background from body (image or color)
    const body = document.body;
    const bodyStyle = window.getComputedStyle(body);
    const bgImage = bodyStyle.backgroundImage;
    const bgColor = bodyStyle.backgroundColor;

    // Collect validation errors
    const errors = [];

    // Validate: Check for CSS gradients
    if (bgImage && (bgImage.includes('linear-gradient') || bgImage.includes('radial-gradient'))) {
      errors.push(
        'CSS gradients are not supported. Use Sharp to rasterize gradients as PNG images first, ' +
        'then reference with background-image: url(\'gradient.png\')'
      );
    }

    let background;
    if (bgImage && bgImage !== 'none') {
      // Extract URL from url("...") or url(...)
      const urlMatch = bgImage.match(/url\(["']?([^"')]+)["']?\)/);
      if (urlMatch) {
        background = {
          type: 'image',
          path: urlMatch[1]
        };
      } else {
        background = {
          type: 'color',
          value: rgbToHex(bgColor)
        };
      }
    } else {
      background = {
        type: 'color',
        value: rgbToHex(bgColor)
      };
    }

    // Process all elements
    const elements = [];
    const placeholders = [];
    const textTags = ['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'UL', 'OL', 'LI'];
    // Block container elements treated identically to DIV (background → shape, unwrapped text → error)
    const blockContainerTags = new Set(['DIV', 'SECTION', 'HEADER', 'FOOTER', 'ARTICLE', 'ASIDE', 'MAIN', 'NAV', 'FIGURE']);
    // Inline elements that LLMs sometimes use as standalone block text containers
    const inlineAsBlockTags = new Set(['SPAN', 'STRONG', 'B', 'EM', 'I', 'A', 'LABEL', 'CODE', 'MARK', 'TIME', 'CITE', 'ABBR', 'S', 'U']);
    // Block elements that are text-leaf-ish (rarely contain P/H* children)
    const leafBlockTags = new Set(['FIGCAPTION', 'DT', 'DD', 'CAPTION', 'BLOCKQUOTE', 'TD', 'TH', 'SUMMARY']);
    const processed = new Set();

    // Switch every element to border-box BEFORE any getBoundingClientRect() call.
    // This makes "width + padding" mean the total box size (not content-box size),
    // so elements like `width:720pt; padding:0 48pt` stay within the 720pt body
    // instead of overflowing to 816pt.  Only affects elements that have an explicit
    // width/height set; elements sized by content or flex-grow are unaffected.
    document.querySelectorAll('*').forEach(el => { el.style.boxSizing = 'border-box'; });

    document.querySelectorAll('*').forEach((el) => {
      if (processed.has(el)) return;

      // Validate text elements don't have backgrounds, borders, or shadows
      if (textTags.includes(el.tagName)) {
        const computed = window.getComputedStyle(el);
        const hasBg = computed.backgroundColor && computed.backgroundColor !== 'rgba(0, 0, 0, 0)';
        const hasBorder = (computed.borderWidth && parseFloat(computed.borderWidth) > 0) ||
                          (computed.borderTopWidth && parseFloat(computed.borderTopWidth) > 0) ||
                          (computed.borderRightWidth && parseFloat(computed.borderRightWidth) > 0) ||
                          (computed.borderBottomWidth && parseFloat(computed.borderBottomWidth) > 0) ||
                          (computed.borderLeftWidth && parseFloat(computed.borderLeftWidth) > 0);
        const hasShadow = computed.boxShadow && computed.boxShadow !== 'none';

        if (hasBg || hasBorder || hasShadow) {
          errors.push(
            `Text element <${el.tagName.toLowerCase()}> has ${hasBg ? 'background' : hasBorder ? 'border' : 'shadow'}. ` +
            'Backgrounds, borders, and shadows are only supported on <div> elements, not text elements.'
          );
          return;
        }
      }

      // Extract placeholder elements (for charts, etc.)
      if (el.className && el.className.includes('placeholder')) {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) {
          errors.push(
            `Placeholder "${el.id || 'unnamed'}" has ${rect.width === 0 ? 'width: 0' : 'height: 0'}. Check the layout CSS.`
          );
        } else {
          placeholders.push({
            id: el.id || `placeholder-${placeholders.length}`,
            x: pxToInch(rect.left),
            y: pxToInch(rect.top),
            w: pxToInch(rect.width),
            h: pxToInch(rect.height)
          });
        }
        processed.add(el);
        el.querySelectorAll('*').forEach(child => processed.add(child));
        return;
      }

      // Extract images
      if (el.tagName === 'IMG') {
        const rect = el.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          elements.push({
            type: 'image',
            src: el.src,
            position: {
              x: pxToInch(rect.left),
              y: pxToInch(rect.top),
              w: pxToInch(rect.width),
              h: pxToInch(rect.height)
            }
          });
          processed.add(el);
          return;
        }
      }

      // Extract block container elements (DIV, SECTION, HEADER, FOOTER, etc.) with backgrounds/borders as shapes
      const isContainer = blockContainerTags.has(el.tagName);
      if (isContainer) {
        const computed = window.getComputedStyle(el);
        const hasBg = computed.backgroundColor && computed.backgroundColor !== 'rgba(0, 0, 0, 0)';

        // Validate: Check for unwrapped text content in block container
        for (const node of el.childNodes) {
          if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent.trim();
            if (text) {
              errors.push(
                `<${el.tagName.toLowerCase()}> contains unwrapped text "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}". ` +
                'All text must be wrapped in <p>, <h1>-<h6>, <ul>, or <ol> tags to appear in PowerPoint.'
              );
            }
          }
        }

        // Check for background images on shapes
        const bgImage = computed.backgroundImage;
        if (bgImage && bgImage !== 'none') {
          errors.push(
            'Background images on DIV elements are not supported. ' +
            'Use solid colors or borders for shapes, or use slide.addImage() in PptxGenJS to layer images.'
          );
          return;
        }

        // Check for borders - both uniform and partial
        const borderTop = computed.borderTopWidth;
        const borderRight = computed.borderRightWidth;
        const borderBottom = computed.borderBottomWidth;
        const borderLeft = computed.borderLeftWidth;
        const borders = [borderTop, borderRight, borderBottom, borderLeft].map(b => parseFloat(b) || 0);
        const hasBorder = borders.some(b => b > 0);
        const hasUniformBorder = hasBorder && borders.every(b => b === borders[0]);
        const borderLines = [];

        if (hasBorder && !hasUniformBorder) {
          const rect = el.getBoundingClientRect();
          const x = pxToInch(rect.left);
          const y = pxToInch(rect.top);
          const w = pxToInch(rect.width);
          const h = pxToInch(rect.height);

          // Collect lines to add after shape (inset by half the line width to center on edge)
          if (parseFloat(borderTop) > 0) {
            const widthPt = pxToPoints(borderTop);
            const inset = (widthPt / 72) / 2; // Convert points to inches, then half
            borderLines.push({
              type: 'line',
              x1: x, y1: y + inset, x2: x + w, y2: y + inset,
              width: widthPt,
              color: rgbToHex(computed.borderTopColor)
            });
          }
          if (parseFloat(borderRight) > 0) {
            const widthPt = pxToPoints(borderRight);
            const inset = (widthPt / 72) / 2;
            borderLines.push({
              type: 'line',
              x1: x + w - inset, y1: y, x2: x + w - inset, y2: y + h,
              width: widthPt,
              color: rgbToHex(computed.borderRightColor)
            });
          }
          if (parseFloat(borderBottom) > 0) {
            const widthPt = pxToPoints(borderBottom);
            const inset = (widthPt / 72) / 2;
            borderLines.push({
              type: 'line',
              x1: x, y1: y + h - inset, x2: x + w, y2: y + h - inset,
              width: widthPt,
              color: rgbToHex(computed.borderBottomColor)
            });
          }
          if (parseFloat(borderLeft) > 0) {
            const widthPt = pxToPoints(borderLeft);
            const inset = (widthPt / 72) / 2;
            borderLines.push({
              type: 'line',
              x1: x + inset, y1: y, x2: x + inset, y2: y + h,
              width: widthPt,
              color: rgbToHex(computed.borderLeftColor)
            });
          }
        }

        if (hasBg || hasBorder) {
          const rect = el.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) {
            const shadow = parseBoxShadow(computed.boxShadow);

            // Only add shape if there's background or uniform border
            if (hasBg || hasUniformBorder) {
              elements.push({
                type: 'shape',
                text: '',  // Shape only - child text elements render on top
                position: {
                  x: pxToInch(rect.left),
                  y: pxToInch(rect.top),
                  w: pxToInch(rect.width),
                  h: pxToInch(rect.height)
                },
                shape: {
                  fill: hasBg ? rgbToHex(computed.backgroundColor) : null,
                  transparency: hasBg ? extractAlpha(computed.backgroundColor) : null,
                  line: hasUniformBorder ? {
                    color: rgbToHex(computed.borderColor),
                    width: pxToPoints(computed.borderWidth)
                  } : null,
                  // Convert border-radius to rectRadius (in inches)
                  // % values: 50%+ = circle (1), <50% = percentage of min dimension
                  // pt values: divide by 72 (72pt = 1 inch)
                  // px values: divide by 96 (96px = 1 inch)
                  rectRadius: (() => {
                    const radius = computed.borderRadius;
                    const radiusValue = parseFloat(radius);
                    if (radiusValue === 0) return 0;

                    if (radius.includes('%')) {
                      if (radiusValue >= 50) return 1;
                      // Calculate percentage of smaller dimension
                      const minDim = Math.min(rect.width, rect.height);
                      return (radiusValue / 100) * pxToInch(minDim);
                    }

                    if (radius.includes('pt')) return radiusValue / 72;
                    return radiusValue / PX_PER_IN;
                  })(),
                  shadow: shadow
                }
              });
            }

            // Add partial border lines
            elements.push(...borderLines);

            processed.add(el);
            return;
          }
        }
      }

      // Extract bullet lists as single text block
      if (el.tagName === 'UL' || el.tagName === 'OL') {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const liElements = Array.from(el.querySelectorAll('li'));
        const items = [];
        const ulComputed = window.getComputedStyle(el);
        const ulPaddingLeftPt = pxToPoints(ulComputed.paddingLeft);

        // Split: margin-left for bullet position, indent for text position
        // margin-left + indent = ul padding-left
        const marginLeft = ulPaddingLeftPt * 0.5;
        const textIndent = ulPaddingLeftPt * 0.5;

        liElements.forEach((li, idx) => {
          const isLast = idx === liElements.length - 1;
          const runs = parseInlineFormatting(li, { breakLine: false });
          // Clean manual bullets from first run
          if (runs.length > 0) {
            runs[0].text = runs[0].text.replace(/^[•\-\*▪▸]\s*/, '');
            runs[0].options.bullet = { indent: textIndent };
          }
          // Set breakLine on last run
          if (runs.length > 0 && !isLast) {
            runs[runs.length - 1].options.breakLine = true;
          }
          items.push(...runs);
        });

        const computed = window.getComputedStyle(liElements[0] || el);

        elements.push({
          type: 'list',
          items: items,
          position: {
            x: pxToInch(rect.left),
            y: pxToInch(rect.top),
            w: pxToInch(rect.width),
            h: pxToInch(rect.height)
          },
          style: {
            fontSize: pxToPoints(computed.fontSize),
            fontFace: mapFontFace(computed.fontFamily, el.textContent),
            color: rgbToHex(computed.color),
            transparency: extractAlpha(computed.color),
            align: computed.textAlign === 'start' ? 'left' : computed.textAlign === 'end' ? 'right' : computed.textAlign,
            lineSpacing: computed.lineHeight && computed.lineHeight !== 'normal' ? pxToPoints(computed.lineHeight) : null,
            charSpacing: computed.letterSpacing && computed.letterSpacing !== 'normal' ? pxToPoints(computed.letterSpacing) : undefined,
            paraSpaceBefore: 0,
            paraSpaceAfter: pxToPoints(computed.marginBottom),
            // PptxGenJS margin array is [left, right, bottom, top]
            margin: [marginLeft, 0, 0, 0]
          }
        });

        liElements.forEach(li => processed.add(li));
        processed.add(el);
        return;
      }

      // Elements used as block-level text containers outside of any text tag:
      // 1. Inline elements (span/strong/b/em/i/a/label/code/…) misused as block containers
      // 2. Leaf-block elements (figcaption/dt/dd/td/th/blockquote/…) that directly hold text
      //    — only when they contain no block-text descendants to avoid duplicate extraction
      if (inlineAsBlockTags.has(el.tagName) && !el.closest('p,h1,h2,h3,h4,h5,h6,li')) {
        el._treatAsP = true;
      } else if (leafBlockTags.has(el.tagName) && !el.closest('p,h1,h2,h3,h4,h5,h6,li')) {
        if (!el.querySelector('p,h1,h2,h3,h4,h5,h6,ul,ol')) {
          el._treatAsP = true;
        }
      }

      // Extract text elements (P, H1, H2, etc.)
      if (!textTags.includes(el.tagName) && !el._treatAsP) return;

      const rect = el.getBoundingClientRect();
      const text = el.textContent.trim();
      if (rect.width === 0 || rect.height === 0 || !text) return;

      // Validate: Check for manual bullet symbols in text elements (not in lists)
      if (el.tagName !== 'LI' && /^[•\-\*▪▸○●◆◇■□]\s/.test(text.trimStart())) {
        errors.push(
          `Text element <${el.tagName.toLowerCase()}> starts with bullet symbol "${text.substring(0, 20)}...". ` +
          'Use <ul> or <ol> lists instead of manual bullet symbols.'
        );
        return;
      }

      const computed = window.getComputedStyle(el);
      const rotation = getRotation(computed.transform, computed.writingMode);
      const { x, y, w, h } = getPositionAndSize(el, rect, rotation);

      // v3: Detect white-space: nowrap and z-index
      const noWrap = computed.whiteSpace === 'nowrap' || computed.whiteSpace === 'pre';
      const cssZIndex = parseInt(computed.zIndex) || 0;

      const baseStyle = {
        fontSize: pxToPoints(computed.fontSize),
        fontFace: mapFontFace(computed.fontFamily, text),
        color: rgbToHex(computed.color),
        align: computed.textAlign === 'start' ? 'left' : computed.textAlign === 'end' ? 'right' : computed.textAlign,
        charSpacing: computed.letterSpacing && computed.letterSpacing !== 'normal' ? pxToPoints(computed.letterSpacing) : undefined,
        lineSpacing: computed.lineHeight && computed.lineHeight !== 'normal' ? pxToPoints(computed.lineHeight) : null,
        paraSpaceBefore: pxToPoints(computed.marginTop),
        paraSpaceAfter: pxToPoints(computed.marginBottom),
        // PptxGenJS margin array is [left, right, bottom, top] (not [top, right, bottom, left] as documented)
        margin: [
          pxToPoints(computed.paddingLeft),
          pxToPoints(computed.paddingRight),
          pxToPoints(computed.paddingBottom),
          pxToPoints(computed.paddingTop)
        ]
      };

      const transparency = extractAlpha(computed.color);
      if (transparency !== null) baseStyle.transparency = transparency;

      if (rotation !== null) baseStyle.rotate = rotation;

      const hasFormatting = el.querySelector('b, i, u, strong, em, span, br');

      if (hasFormatting) {
        // Text with inline formatting
        const transformStr = computed.textTransform;
        const runs = parseInlineFormatting(el, {}, [], (str) => applyTextTransform(str, transformStr));

        // Adjust lineSpacing based on largest fontSize in runs
        const adjustedStyle = { ...baseStyle };
        if (adjustedStyle.lineSpacing) {
          const maxFontSize = Math.max(
            adjustedStyle.fontSize,
            ...runs.map(r => r.options?.fontSize || 0)
          );
          if (maxFontSize > adjustedStyle.fontSize) {
            const lineHeightMultiplier = adjustedStyle.lineSpacing / adjustedStyle.fontSize;
            adjustedStyle.lineSpacing = maxFontSize * lineHeightMultiplier;
          }
        }

        elements.push({
          type: el._treatAsP ? 'p' : el.tagName.toLowerCase(),
          text: runs,
          noWrap,
          zIndex: cssZIndex,
          position: { x: pxToInch(x), y: pxToInch(y), w: pxToInch(w), h: pxToInch(h) },
          style: adjustedStyle
        });
      } else {
        // Plain text - inherit CSS formatting
        const textTransform = computed.textTransform;
        const transformedText = applyTextTransform(text, textTransform);

        const isBold = computed.fontWeight === 'bold' || parseInt(computed.fontWeight) >= 600;

        elements.push({
          type: el._treatAsP ? 'p' : el.tagName.toLowerCase(),
          text: transformedText,
          noWrap,
          zIndex: cssZIndex,
          position: { x: pxToInch(x), y: pxToInch(y), w: pxToInch(w), h: pxToInch(h) },
          style: {
            ...baseStyle,
            bold: isBold && !shouldSkipBold(computed.fontFamily),
            italic: computed.fontStyle === 'italic',
            underline: computed.textDecoration.includes('underline')
          }
        });
      }

      processed.add(el);
    });

    return { background, elements, placeholders, errors };
  });
}

async function html2pptx(htmlFile, pres, options = {}) {
  const {
    tmpDir = process.env.TMPDIR || '/tmp',
    slide = null,
    fontConfig = null  // { cjk: 'SimHei', latin: 'Century Gothic', emphasis: 'Franklin Gothic Medium' }
  } = options;

  try {
    // Use Chrome on macOS, default Chromium on Unix
    const launchOptions = { env: { TMPDIR: tmpDir } };
    if (process.platform === 'darwin') {
      launchOptions.channel = 'chrome';
    }

    const browser = await chromium.launch(launchOptions);

    let bodyDimensions;
    let slideData;

    const filePath = path.isAbsolute(htmlFile) ? htmlFile : path.join(process.cwd(), htmlFile);
    const validationErrors = [];

    try {
      const page = await browser.newPage();
      page.on('console', (msg) => {
        // Log the message text to your test runner's console
        console.log(`Browser console: ${msg.text()}`);
      });

      await page.goto(`file://${filePath}`);

      // Inject font config into the page for extractSlideData to use
      if (fontConfig) {
        await page.evaluate((fc) => { window.__FONT_CONFIG__ = fc; }, fontConfig);
      }

      bodyDimensions = await getBodyDimensions(page);

      await page.setViewportSize({
        width: Math.round(bodyDimensions.width),
        height: Math.round(bodyDimensions.height)
      });

      // Force a layout reflow after viewport resize so flex centering takes effect
      await page.evaluate(() => void document.body.offsetHeight);
      await page.waitForTimeout(100);

      // Re-read body dimensions after reflow to capture correct layout
      bodyDimensions = await getBodyDimensions(page);

      slideData = await extractSlideData(page);
    } finally {
      await browser.close();
    }

    // Apply emphasis font to bold numeric text (e.g. KPI values, percentages)
    if (fontConfig?.emphasis) applyEmphasisFont(slideData, fontConfig.emphasis);

    // Collect all validation errors
    const overflowWarnings = [];
    if (bodyDimensions.errors && bodyDimensions.errors.length > 0) {
      overflowWarnings.push(...bodyDimensions.errors);
    }

    // const dimensionErrors = validateDimensions(bodyDimensions, pres);
    // if (dimensionErrors.length > 0) {
    //   validationErrors.push(...dimensionErrors);
    // }

    // const textBoxPositionErrors = validateTextBoxPosition(slideData, bodyDimensions);
    // if (textBoxPositionErrors.length > 0) {
    //   validationErrors.push(...textBoxPositionErrors);
    // }

    if (slideData.errors && slideData.errors.length > 0) {
      validationErrors.push(...slideData.errors);
    }

    // v3: Min font size check (blocking)
    const fontErrors = checkMinFontSize(slideData);
    if (fontErrors.length > 0) validationErrors.push(...fontErrors);

    // Throw blocking errors (structural issues that corrupt the output)
    if (validationErrors.length > 0) {
      const errorMessage = validationErrors.length === 1
        ? validationErrors[0]
        : `Multiple validation errors found:\n${validationErrors.map((e, i) => `  ${i + 1}. ${e}`).join('\n')}`;
      throw new Error(errorMessage);
    }

    // v3: Collect all non-blocking warnings
    const slideWidthIn = pres.presLayout ? pres.presLayout.width / EMU_PER_IN : 10;
    const slideHeightIn = pres.presLayout ? pres.presLayout.height / EMU_PER_IN : 5.625;
    const allWarnings = [...overflowWarnings];
    allWarnings.push(...checkElementBounds(slideData, slideWidthIn, slideHeightIn));
    allWarnings.push(...checkVerticalBalance(slideData, slideHeightIn));
    allWarnings.push(...checkTextOverlaps(slideData));
    allWarnings.push(...checkCharCount(slideData));

    const targetSlide = slide || pres.addSlide();

    await addBackground(slideData, targetSlide, pres, tmpDir);
    addElements(slideData, targetSlide, pres, tmpDir);

    // Print warnings after successful conversion (non-blocking)
    if (allWarnings.length > 0) {
      const suggestions = allWarnings.map((w, i) => `  ${i + 1}. ${w}`).join('\n');
      console.warn(`[html2pptx] ${htmlFile}: ${allWarnings.length} warning(s):\n${suggestions}`);
    }

    return { slide: targetSlide, placeholders: slideData.placeholders, warnings: allWarnings };
  } catch (error) {
    if (!error.message.startsWith(htmlFile)) {
      throw new Error(`${htmlFile}: ${error.message}`);
    }
    throw error;
  }
}

module.exports = html2pptx;