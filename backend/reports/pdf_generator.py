#!/usr/bin/env python3
# tools/pdf_generator.py
"""
PDF Generator
-------------
Convert Markdown or HTML into a styled PDF.

Primary engine:  WeasyPrint (HTML/CSS to PDF, full-featured)
Fallback:        ReportLab (plain text, minimal styling) if WeasyPrint unavailable

Features (WeasyPrint path)
  • Markdown -> HTML (Python-Markdown) or pass raw HTML
  • Optional cover page (title, subtitle, date, logo)
  • Automatic Table of Contents (h1–h3)
  • Headers/Footers w/ left/center/right slots (page numbers via CSS counters)
  • Page size: A4/Letter/Legal; margins
  • Light or Dark theme; custom CSS override
  • Local images/fonts supported (use --base-dir for relative paths)

CLI examples
  python tools/pdf_generator.py \
    --input reports/daily_2025-09-03.md \
    --output reports/daily_2025-09-03.pdf \
    --title "Morning Tape" \
    --subtitle "Markets, News & Weather" \
    --logo assets/logo.png \
    --theme light --page-size A4 --toc

  python tools/pdf_generator.py \
    --input docs/overview.html \
    --output out/overview.pdf \
    --header-center "Confidential" --footer-right "Page [page]/[pages]"

Dependencies (recommended)
  pip install weasyprint markdown jinja2 pygments

Fallback-only (no HTML/CSS rendering):
  pip install reportlab
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import textwrap
from typing import Optional, Tuple, List

# Optional imports
WEASYPRINT_OK = False
MARKDOWN_OK = False
JINJA2_OK = False

try:
    from weasyprint import HTML, CSS, default_url_fetcher  # type: ignore
    WEASYPRINT_OK = True
except Exception:
    WEASYPRINT_OK = False

try:
    import markdown  # type: ignore
    MARKDOWN_OK = True
except Exception:
    MARKDOWN_OK = False

try:
    from jinja2 import Template  # type: ignore
    JINJA2_OK = True
except Exception:
    JINJA2_OK = False

# Fallback text-only PDF
REPORTLAB_OK = False
try:
    from reportlab.lib.pagesizes import A4, LETTER, legal # type: ignore
    from reportlab.lib.units import inch # type: ignore
    from reportlab.pdfgen import canvas # type: ignore
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ------------------------- Utils -------------------------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def today_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")

def page_size_tuple(name: str) -> Tuple[float, float]:
    n = name.lower()
    if not REPORTLAB_OK:
        # Not used in WeasyPrint (handled via CSS); return LETTER-ish default
        return (612, 792)
    if n == "a4":
        return A4
    if n == "legal":
        return legal
    return LETTER  # default

def guess_is_html(text: str) -> bool:
    t = text.lstrip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html") or "<html" in t[:200]

def md_to_html(src: str) -> str:
    if not MARKDOWN_OK:
        # Minimal fallback: wrap pre
        esc = (src
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
        return f"<pre>{esc}</pre>"
    exts = ["extra", "toc", "sane_lists", "admonition", "codehilite"]
    return markdown.markdown(src, extensions=exts, output_format="html5") # type: ignore

def build_cover_html(title: str, subtitle: Optional[str], logo_path: Optional[str]) -> str:
    logo_html = ""
    if logo_path:
        logo_html = f'<img class="logo" src="{logo_path}" alt="logo" />'
    sub_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    date_html = f'<div class="date">{today_str()}</div>'
    return f"""
<section class="cover-page">
  {logo_html}
  <h1 class="title">{title}</h1>
  {sub_html}
  {date_html}
</section>
"""

def extract_headings_for_toc(html: str) -> List[Tuple[int, str]]:
    """Naive heading scrape (h1-h3) without external deps."""
    import re
    headings = []
    for level in [1, 2, 3]:
        for m in re.finditer(rf"<h{level}[^>]*>(.*?)</h{level}>", html, re.IGNORECASE | re.DOTALL):
            text = re.sub("<[^<]+?>", "", m.group(1)).strip()
            if text:
                headings.append((level, text))
    return headings

def build_toc_html(headings: List[Tuple[int, str]], toc_title: str = "Table of Contents") -> str:
    if not headings:
        return ""
    items = []
    for lvl, txt in headings:
        cls = {1: "toc-h1", 2: "toc-h2", 3: "toc-h3"}.get(lvl, "toc-h2")
        items.append(f'<div class="{cls}">{txt}</div>')
    return f"""
<section class="toc-page">
  <h2>{toc_title}</h2>
  {''.join(items)}
</section>
"""

# ------------------------- CSS Themes -------------------------

BASE_CSS = """
@page {
  size: {{PAGE_SIZE}};
  margin: {{MARGIN}};
  @top-left { content: "{{HEADER_LEFT}}"; font-size: 9pt; color: {{MUTED}}; }
  @top-center { content: "{{HEADER_CENTER}}"; font-size: 9pt; color: {{MUTED}}; }
  @top-right { content: "{{HEADER_RIGHT}}"; font-size: 9pt; color: {{MUTED}}; }
  @bottom-left { content: "{{FOOTER_LEFT}}"; font-size: 9pt; color: {{MUTED}}; }
  @bottom-center { content: "{{FOOTER_CENTER}}"; font-size: 9pt; color: {{MUTED}}; }
  @bottom-right { content: "{{FOOTER_RIGHT}}"; font-size: 9pt; color: {{MUTED}}; }
}

:root {
  --bg: {{BG}};
  --fg: {{FG}};
  --muted: {{MUTED}};
  --accent: {{ACCENT}};
  --rule: {{RULE}};
  --mono: "SFMono-Regular", ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
}

html, body {
  background: var(--bg);
  color: var(--fg);
  font-family: var(--sans);
  font-size: 11pt;
  line-height: 1.45;
}

code, pre, kbd, samp { font-family: var(--mono); }

img { max-width: 100%; }

hr { border: 0; border-top: 1px solid var(--rule); margin: 1.2rem 0; }

h1, h2, h3, h4 {
  color: var(--fg);
  margin: 0.8rem 0 0.4rem;
  line-height: 1.2;
}

h1 { font-size: 24pt; border-bottom: 2px solid var(--rule); padding-bottom: 0.3rem; }
h2 { font-size: 18pt; border-bottom: 1px solid var(--rule); padding-bottom: 0.2rem; }
h3 { font-size: 14pt; }

table { border-collapse: collapse; width: 100%; margin: 0.6rem 0; }
th, td { padding: 6px 8px; border: 1px solid var(--rule); }
th { background: {{TABLE_TH_BG}}; }

blockquote {
  margin: 0.6rem 0; padding: 0.6rem 0.8rem;
  border-left: 3px solid var(--accent); background: {{BLOCK_BG}};
}

.cover-page {
  page: cover;
  page-break-after: always;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; min-height: 80vh;
}
.cover-page .logo { max-height: 120px; margin-bottom: 24px; }
.cover-page .title { font-size: 36pt; margin: 0; }
.cover-page .subtitle { margin-top: 8px; font-size: 14pt; color: var(--muted); }
.cover-page .date { margin-top: 16px; color: var(--muted); }

.toc-page { page-break-after: always; }
.toc-page h2 { margin-top: 0; }
.toc-h1 { font-weight: 700; margin: 6px 0; }
.toc-h2 { margin-left: 12px; margin: 4px 0; }
.toc-h3 { margin-left: 24px; margin: 2px 0; }

/* Page number helper for footers (WeasyPrint expands counters) */
@page {
  @bottom-right { content: "{{FOOTER_RIGHT}}"; }
}
"""

THEME_LIGHT = {
    "BG": "#ffffff",
    "FG": "#111213",
    "MUTED": "#6b7280",
    "ACCENT": "#2563eb",
    "RULE": "#e5e7eb",
    "TABLE_TH_BG": "#f3f4f6",
    "BLOCK_BG": "#f9fafb",
}

THEME_DARK = {
    "BG": "#0b0e14",
    "FG": "#e6e6e6",
    "MUTED": "#9aa5b1",
    "ACCENT": "#4f83ff",
    "RULE": "#202938",
    "TABLE_TH_BG": "#111826",
    "BLOCK_BG": "#0f141c",
}

# ------------------------- Renderers -------------------------

def render_weasy(
    *,
    html_body: str,
    output_pdf: str,
    title: Optional[str],
    subtitle: Optional[str],
    logo: Optional[str],
    page_size: str,
    margin: str,
    header_left: str,
    header_center: str,
    header_right: str,
    footer_left: str,
    footer_center: str,
    footer_right: str,
    theme: str,
    base_dir: Optional[str],
    include_toc: bool,
    extra_css_path: Optional[str],
) -> None:
    # Build document HTML (cover + toc + body)
    cover = build_cover_html(title or "", subtitle, logo) if title else ""
    toc_html = build_toc_html(extract_headings_for_toc(html_body)) if include_toc else ""
    doc_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{(title or 'Document')}</title>
  <style>
    /* Force page numbers if placeholder tokens used */
  </style>
</head>
<body>
  {cover}
  {toc_html}
  <section class="content">
    {html_body}
  </section>
</body>
</html>"""

    # Prepare CSS
    theme_vars = THEME_DARK if theme.lower() == "dark" else THEME_LIGHT
    css_template = BASE_CSS
    if JINJA2_OK:
        css_template = Template(BASE_CSS).render(
            PAGE_SIZE=page_size,
            MARGIN=margin,
            HEADER_LEFT=header_left,
            HEADER_CENTER=header_center,
            HEADER_RIGHT=header_right,
            FOOTER_LEFT=footer_left,
            FOOTER_CENTER=footer_center,
            # If user provides "[page]/[pages]" tokens, WeasyPrint will expand them automatically with counters
            FOOTER_RIGHT=footer_right.replace("[page]", "counter(page)").replace("[pages]", "counter(pages)"),
            **theme_vars
        )
    else:
        css_template = BASE_CSS.replace("{{PAGE_SIZE}}", page_size)\
            .replace("{{MARGIN}}", margin)\
            .replace("{{HEADER_LEFT}}", header_left)\
            .replace("{{HEADER_CENTER}}", header_center)\
            .replace("{{HEADER_RIGHT}}", header_right)\
            .replace("{{FOOTER_LEFT}}", footer_left)\
            .replace("{{FOOTER_CENTER}}", footer_center)\
            .replace("{{FOOTER_RIGHT}}", footer_right)\
            .replace("{{BG}}", theme_vars["BG"])\
            .replace("{{FG}}", theme_vars["FG"])\
            .replace("{{MUTED}}", theme_vars["MUTED"])\
            .replace("{{ACCENT}}", theme_vars["ACCENT"])\
            .replace("{{RULE}}", theme_vars["RULE"])\
            .replace("{{TABLE_TH_BG}}", theme_vars["TABLE_TH_BG"])\
            .replace("{{BLOCK_BG}}", theme_vars["BLOCK_BG"])

    css_list = [CSS(string=css_template)]
    if extra_css_path and os.path.exists(extra_css_path):
        css_list.append(CSS(filename=extra_css_path))

    # Base URL so relative assets (images, CSS) resolve
    base_url = base_dir if base_dir else os.getcwd()

    HTML(string=doc_html, base_url=base_url, url_fetcher=default_url_fetcher).write_pdf(
        output_pdf,
        stylesheets=css_list,
        presentational_hints=True,
        zoom=1.0,
    )


def render_reportlab(
    *,
    text: str,
    output_pdf: str,
    page_size: str,
    header_center: str,
    footer_right: str,
) -> None:
    """Very simple text-only fallback (no CSS)."""
    ensure_dir(output_pdf)
    w, h = page_size_tuple(page_size)
    c = canvas.Canvas(output_pdf, pagesize=(w, h))
    margin = 0.75 * inch if REPORTLAB_OK else 54
    width = w - 2 * margin
    y = h - margin

    def draw_header_footer():
        c.setFont("Helvetica", 9)
        c.setFillGray(0.4)
        c.drawString(margin, h - margin + 10, header_center[:80])
        # Simple page number token expansion
        pg_text = footer_right.replace("[page]", str(c.getPageNumber()))
        c.drawRightString(w - margin, margin - 20, pg_text)
        c.setFillGray(0.0)

    # Title (first line)
    lines = text.splitlines()
    c.setFont("Helvetica-Bold", 14)
    if lines:
        c.drawString(margin, y, lines[0][:100])
        y -= 24

    c.setFont("Helvetica", 10)
    for line in lines[1:]:
        wrapped = textwrap.wrap(line, width=int(width / 6)) or [""]
        for piece in wrapped:
            if y < margin + 40:
                draw_header_footer()
                c.showPage()
                y = h - margin
                c.setFont("Helvetica", 10)
            c.drawString(margin, y, piece)
            y -= 14

    draw_header_footer()
    c.showPage()
    c.save()

# ------------------------- CLI -------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Markdown/HTML to PDF generator with cover/TOC and themed styling.")
    p.add_argument("--input", "-i", required=True, help="Input file (.md or .html).")
    p.add_argument("--output", "-o", required=True, help="Output PDF path.")
    p.add_argument("--title", help="Cover title. If omitted, no cover page.")
    p.add_argument("--subtitle", help="Cover subtitle.")
    p.add_argument("--logo", help="Path to logo image for cover.")
    p.add_argument("--theme", choices=["light", "dark"], default="light")
    p.add_argument("--page-size", choices=["A4", "Letter", "Legal"], default="Letter")
    p.add_argument("--margin", default="20mm", help="Page margin (e.g., 20mm or 0.7in). WeasyPrint only.")
    p.add_argument("--header-left", default="", help="Header left text.")
    p.add_argument("--header-center", default="", help="Header center text.")
    p.add_argument("--header-right", default="", help="Header right text.")
    p.add_argument("--footer-left", default="", help="Footer left text.")
    p.add_argument("--footer-center", default="", help="Footer center text.")
    p.add_argument("--footer-right", default="[page]/[pages]", help="Footer right text. Tokens: [page], [pages].")
    p.add_argument("--toc", action="store_true", help="Include Table of Contents (h1–h3).")
    p.add_argument("--base-dir", default=None, help="Base directory to resolve relative assets (images/CSS).")
    p.add_argument("--extra-css", default=None, help="Path to additional CSS file to include (WeasyPrint only).")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)

    src = read_text(args.input)
    is_html = guess_is_html(src) or args.input.lower().endswith(".html")
    ensure_dir(args.output)

    if WEASYPRINT_OK:
        # Build HTML body
        html_body = src if is_html else md_to_html(src)
        try:
            render_weasy(
                html_body=html_body,
                output_pdf=args.output,
                title=args.title,
                subtitle=args.subtitle,
                logo=args.logo,
                page_size=args.page_size,
                margin=args.margin,
                header_left=args.header_left or "",
                header_center=args.header_center or "",
                header_right=args.header_right or "",
                footer_left=args.footer_left or "",
                footer_center=args.footer_center or "",
                footer_right=args.footer_right or "",
                theme=args.theme,
                base_dir=args.base_dir or (os.path.dirname(os.path.abspath(args.input)) if os.path.isfile(args.input) else os.getcwd()),
                include_toc=bool(args.toc),
                extra_css_path=args.extra_css,
            )
            print(f"[pdf_generator] wrote {args.output} via WeasyPrint")
            return 0
        except Exception as e:
            print(f"[pdf_generator] WeasyPrint failed: {e} — falling back to ReportLab (text-only)")

    if REPORTLAB_OK:
        # Plain text fallback (strip HTML tags crudely if input was HTML)
        plain = src
        if is_html:
            import re
            plain = re.sub(r"<[^>]+>", "", src)
        try:
            render_reportlab(
                text=plain,
                output_pdf=args.output,
                page_size=args.page_size,
                header_center=args.header_center or (args.title or ""),
                footer_right=args.footer_right.replace("[pages]", ""),  # ReportLab doesn't know total pages here
            )
            print(f"[pdf_generator] wrote {args.output} via ReportLab (fallback)")
            return 0
        except Exception as e:
            print(f"[pdf_generator] ReportLab fallback failed: {e}")
            return 2

    print("[pdf_generator] No PDF engine available. Install 'weasyprint' or 'reportlab'.")
    return 3


if __name__ == "__main__":
    sys.exit(main())