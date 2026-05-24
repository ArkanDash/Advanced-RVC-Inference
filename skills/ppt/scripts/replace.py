#!/usr/bin/env python3
"""Apply text replacements to PowerPoint presentation.

Usage:
    python replace.py <input.pptx> <replacements.json> <output.pptx>

The replacements JSON should have the structure output by inventory.py.
ALL text shapes identified by inventory.py will have their text cleared
unless "paragraphs" is specified in the replacements for that shape.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from inventory import InventoryData, extract_text_inventory
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

_ALIGN_MAP = {
    "LEFT": PP_ALIGN.LEFT,
    "CENTER": PP_ALIGN.CENTER,
    "RIGHT": PP_ALIGN.RIGHT,
    "JUSTIFY": PP_ALIGN.JUSTIFY,
}

# Bullet indentation constants
# marL = font_size × (1 + level) × 1.6 pts, converted to EMUs (1 pt = 12700 EMU)
_INDENT_FACTOR = 1.6
_EMU_PER_PT = 12700


def _clear_paragraph_bullets(paragraph):
    """Remove all bullet XML elements from a paragraph's pPr."""
    pPr = paragraph._element.get_or_add_pPr()
    for child in list(pPr):
        if any(child.tag.endswith(t) for t in ("buChar", "buNone", "buAutoNum", "buFont")):
            pPr.remove(child)
    return pPr


def _apply_paragraph_properties(paragraph, para_data: Dict[str, Any]):
    text = para_data.get("text", "")
    pPr = _clear_paragraph_bullets(paragraph)

    if para_data.get("bullet", False):
        level = para_data.get("level", 0)
        paragraph.level = level
        font_size = para_data.get("font_size", 18.0)
        pPr.attrib["marL"] = str(int(font_size * _INDENT_FACTOR * (1 + level) * _EMU_PER_PT))
        pPr.attrib["indent"] = str(int(-font_size * 0.8 * _EMU_PER_PT))
        buChar = OxmlElement("a:buChar")
        buChar.set("char", "•")
        pPr.append(buChar)
        if "alignment" not in para_data:
            paragraph.alignment = PP_ALIGN.LEFT
    else:
        pPr.attrib["marL"] = "0"
        pPr.attrib["indent"] = "0"
        pPr.insert(0, OxmlElement("a:buNone"))

    if para_data.get("alignment") in _ALIGN_MAP:
        paragraph.alignment = _ALIGN_MAP[para_data["alignment"]]
    if "space_before" in para_data:
        paragraph.space_before = Pt(para_data["space_before"])
    if "space_after" in para_data:
        paragraph.space_after = Pt(para_data["space_after"])
    if "line_spacing" in para_data:
        paragraph.line_spacing = Pt(para_data["line_spacing"])

    run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
    run.text = text
    _apply_font_properties(run, para_data)


def _apply_font_properties(run, para_data: Dict[str, Any]):
    for attr in ("bold", "italic", "underline"):
        if attr in para_data:
            setattr(run.font, attr, para_data[attr])
    if "font_size" in para_data:
        run.font.size = Pt(para_data["font_size"])
    if "font_name" in para_data:
        run.font.name = para_data["font_name"]
    if "color" in para_data:
        h = para_data["color"].lstrip("#")
        if len(h) == 6:
            run.font.color.rgb = RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    elif "theme_color" in para_data:
        try:
            run.font.color.theme_color = getattr(MSO_THEME_COLOR, para_data["theme_color"])
        except AttributeError:
            print(f"  WARNING: Unknown theme color '{para_data['theme_color']}'")


def _check_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in JSON: '{key}'")
        result[key] = value
    return result


def _validate_replacements(inventory: InventoryData, replacements: Dict) -> List[str]:
    errors = []
    for slide_key, shapes_data in replacements.items():
        if not slide_key.startswith("slide-"):
            continue
        if slide_key not in inventory:
            errors.append(f"Slide '{slide_key}' not found in inventory")
            continue
        for shape_key in shapes_data:
            if shape_key not in inventory[slide_key]:
                available = sorted(inventory[slide_key].keys())
                errors.append(
                    f"Shape '{shape_key}' not found on '{slide_key}'. "
                    f"Available: {', '.join(available)}"
                )
    return errors


def apply_replacements(pptx_file: str, json_file: str, output_file: str):
    prs = Presentation(pptx_file)
    inventory = extract_text_inventory(Path(pptx_file), prs)

    # Snapshot original overflow so we can detect if replacements make it worse
    original_overflow: Dict[str, Dict[str, float]] = {
        slide_key: {
            shape_key: sd.frame_overflow_bottom
            for shape_key, sd in shapes.items()
            if sd.frame_overflow_bottom is not None
        }
        for slide_key, shapes in inventory.items()
    }

    with open(json_file) as f:
        replacements = json.load(f, object_pairs_hook=_check_duplicate_keys)

    errors = _validate_replacements(inventory, replacements)
    if errors:
        print("ERROR: Invalid shapes in replacement JSON:")
        for e in errors:
            print(f"  - {e}")
        raise ValueError(f"Found {len(errors)} validation error(s)")

    shapes_cleared = shapes_replaced = 0

    for slide_key, shapes_dict in inventory.items():
        if not slide_key.startswith("slide-"):
            continue
        for shape_key, shape_data in shapes_dict.items():
            if not shape_data.shape:
                continue
            tf = shape_data.shape.text_frame  # type: ignore
            tf.clear()
            shapes_cleared += 1

            para_list = replacements.get(slide_key, {}).get(shape_key, {}).get("paragraphs")
            if not para_list:
                continue
            shapes_replaced += 1
            # Inherit original font_size if not specified in replacement
            orig_paras = shape_data.paragraphs or []
            orig_font_size = orig_paras[0].get("font_size") if orig_paras else None
            for i, para_data in enumerate(para_list):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                if orig_font_size is not None and "font_size" not in para_data:
                    para_data = {**para_data, "font_size": orig_font_size}
                _apply_paragraph_properties(p, para_data)

    # Re-check overflow on the updated in-memory presentation.
    # Note: extract_text_inventory may add benign empty <a:solidFill/> elements
    # while reading font colors — these are harmless and ignored by PowerPoint.
    updated_inventory = extract_text_inventory(Path(pptx_file), prs)

    overflow_errors: List[str] = []
    warnings: List[str] = []
    for slide_key, shapes_dict in updated_inventory.items():
        for shape_key, sd in shapes_dict.items():
            for w in sd.warnings:
                warnings.append(f"{slide_key}/{shape_key}: {w}")
            new_ov = sd.frame_overflow_bottom
            if new_ov is not None:
                old_ov = original_overflow.get(slide_key, {}).get(shape_key, 0.0)
                if new_ov > old_ov + 0.01:
                    overflow_errors.append(
                        f'{slide_key}/{shape_key}: overflow increased by {new_ov - old_ov:.2f}" '
                        f'(was {old_ov:.2f}", now {new_ov:.2f}")'
                    )

    if overflow_errors or warnings:
        print("\nWARNING: Issues in replacement output:")
        for e in overflow_errors:
            print(f"  overflow  - {e}")
        for w in warnings:
            print(f"  warning   - {w}")

    prs.save(output_file)
    print(f"Saved: {output_file}")
    print(f"  Shapes cleared: {shapes_cleared}, replaced: {shapes_replaced}")


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    input_pptx, replacements_json, output_pptx = (
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    )
    for p in (input_pptx, replacements_json):
        if not p.exists():
            print(f"Error: File not found: {p}")
            sys.exit(1)

    try:
        apply_replacements(str(input_pptx), str(replacements_json), str(output_pptx))
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
