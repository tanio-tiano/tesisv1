"""Conversor Markdown -> PDF (reportlab, sin dependencias externas).

Pensado para documentos tecnicos controlados (planes/reportes): soporta titulos
(#, ##, ###), tablas con pipes, bloques de codigo ```...```, listas (con/sin
numero), citas (>), y formato inline **negrita** / `codigo` / *italica*.

Registra fuentes TrueType de Windows (Arial/Consolas) para que los acentos del
espanol y simbolos comunes se rendericen correctamente en cualquier visor.

Uso:
    python -m analysis.md_to_pdf entrada.md salida.pdf "Subtitulo opcional"
"""

from __future__ import annotations

import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _register_fonts():
    """Registra Arial/Consolas (TTF Windows). Si falla, usa las base de reportlab."""
    win = r"C:\Windows\Fonts"
    try:
        pdfmetrics.registerFont(TTFont("Body", os.path.join(win, "arial.ttf")))
        pdfmetrics.registerFont(TTFont("Body-Bold", os.path.join(win, "arialbd.ttf")))
        pdfmetrics.registerFont(TTFont("Body-Italic", os.path.join(win, "ariali.ttf")))
        pdfmetrics.registerFont(TTFont("Body-BoldItalic", os.path.join(win, "arialbi.ttf")))
        registerFontFamily("Body", normal="Body", bold="Body-Bold",
                           italic="Body-Italic", boldItalic="Body-BoldItalic")
        mono_src = "consola.ttf" if os.path.exists(os.path.join(win, "consola.ttf")) else "cour.ttf"
        pdfmetrics.registerFont(TTFont("Mono", os.path.join(win, mono_src)))
        return {"body": "Body", "bold": "Body-Bold", "italic": "Body-Italic", "mono": "Mono"}
    except Exception:
        return {"body": "Helvetica", "bold": "Helvetica-Bold",
                "italic": "Helvetica-Oblique", "mono": "Courier"}


FONTS = _register_fonts()

# Con TTF Arial casi todo Unicode renderiza; solo mapeamos emojis/simbolos raros.
_UNICODE = {"✅": "[OK]", "❌": "[X]", "⏳": "", "🤖": "", "💡": "", "📄": "", "🔧": ""}


def _norm(text):
    for key, value in _UNICODE.items():
        text = text.replace(key, value)
    return text


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(text):
    text = _esc(_norm(text))
    text = re.sub(r"`([^`]+)`", rf"<font face='{FONTS['mono']}'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [t](u) -> t
    return text


def _styles():
    f = FONTS
    navy = colors.HexColor("#14213d")
    s = {}
    s["h1"] = ParagraphStyle("h1", fontName=f["bold"], fontSize=18, leading=22,
                             spaceAfter=8, textColor=navy)
    s["h2"] = ParagraphStyle("h2", fontName=f["bold"], fontSize=13.5, leading=17,
                             spaceBefore=14, spaceAfter=5, textColor=navy)
    s["h3"] = ParagraphStyle("h3", fontName=f["bold"], fontSize=11.5, leading=14,
                             spaceBefore=9, spaceAfter=3, textColor=colors.HexColor("#333333"))
    s["body"] = ParagraphStyle("body", fontName=f["body"], fontSize=10, leading=14, spaceAfter=4)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=14, bulletIndent=3,
                                 bulletFontName=f["body"])
    s["bullet2"] = ParagraphStyle("bullet2", parent=s["body"], leftIndent=28, bulletIndent=17,
                                  bulletFontName=f["body"])
    s["quote"] = ParagraphStyle("quote", parent=s["body"], leftIndent=12, fontName=f["italic"],
                                textColor=colors.HexColor("#555555"))
    s["code"] = ParagraphStyle("code", fontName=f["mono"], fontSize=7.6, leading=9.6,
                               backColor=colors.HexColor("#f4f6f8"), borderPadding=5,
                               textColor=colors.HexColor("#21303b"))
    s["cell"] = ParagraphStyle("cell", parent=s["body"], fontSize=8.5, leading=11, spaceAfter=0)
    s["cellh"] = ParagraphStyle("cellh", parent=s["cell"], fontName=f["bold"])
    s["meta"] = ParagraphStyle("meta", parent=s["body"], fontSize=9.5,
                               textColor=colors.HexColor("#666666"), spaceAfter=10)
    return s


def _table(rows, styles, avail_width):
    cells = []
    for r, row in enumerate(rows):
        sty = styles["cellh"] if r == 0 else styles["cell"]
        cells.append([Paragraph(_inline(c), sty) for c in row])
    ncols = max(len(r) for r in cells)
    for r in cells:
        r += [Paragraph("", styles["cell"])] * (ncols - len(r))
    tbl = Table(cells, colWidths=[avail_width / ncols] * ncols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8bdc4")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def _split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md_path, pdf_path, subtitle=None):
    lines = open(md_path, encoding="utf-8").read().splitlines()
    s = _styles()
    avail = A4[0] - 30 * mm
    flow = []
    if subtitle:
        flow.append(Paragraph(_inline(subtitle), s["meta"]))

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flow.append(Spacer(1, 4)); i += 1; continue

        if stripped.startswith("```"):
            i += 1; buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(_norm(lines[i])); i += 1
            i += 1
            flow.append(Preformatted(_esc("\n".join(buf)) or " ", s["code"])); continue

        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            rows = [_split_row(line)]; i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i])); i += 1
            flow.append(_table(rows, s, avail)); flow.append(Spacer(1, 4)); continue

        if stripped.startswith("### "):
            flow.append(Paragraph(_inline(stripped[4:]), s["h3"])); i += 1; continue
        if stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[3:]), s["h2"])); i += 1; continue
        if stripped.startswith("# "):
            flow.append(Paragraph(_inline(stripped[2:]), s["h1"])); i += 1; continue
        if stripped.startswith("> "):
            flow.append(Paragraph(_inline(stripped[2:]), s["quote"])); i += 1; continue

        m = re.match(r"^(\s*)([-*])\s+(.*)$", line)
        if m:
            sty = s["bullet2"] if len(m.group(1)) >= 2 else s["bullet"]
            flow.append(Paragraph(_inline(m.group(3)), sty, bulletText="•")); i += 1; continue

        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            flow.append(Paragraph(_inline(m.group(3)), s["bullet"], bulletText=f"{m.group(2)}.")); i += 1
            continue

        flow.append(Paragraph(_inline(stripped), s["body"])); i += 1

    SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title="Plan WO + controlador SHAP",
    ).build(flow)
    return pdf_path


def main():
    md, pdf = sys.argv[1], sys.argv[2]
    subtitle = sys.argv[3] if len(sys.argv) > 3 else None
    print("Fuentes:", FONTS["body"])
    print("PDF generado:", convert(md, pdf, subtitle))


if __name__ == "__main__":
    main()
