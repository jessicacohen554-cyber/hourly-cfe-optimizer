#!/usr/bin/env python3
"""Convert ref-vre-investment-thesis.html to PDF.

Strategy: Parse the HTML, strip Chart.js canvases and external JS/CSS deps,
inline all styles, and render with WeasyPrint. Tables contain the same data
as charts, so no information is lost.

Usage:
    cd data-center-cfe/
    pip install weasyprint
    python convert_to_pdf.py
"""
import re
from pathlib import Path
from weasyprint import HTML, CSS

SRC = Path(__file__).parent / "research" / "ref-vre-investment-thesis.html"
OUT = Path(__file__).parent / "output" / "ref-vre-investment-thesis.pdf"

html = SRC.read_text(encoding="utf-8")

# Remove all <script> tags (Chart.js, nav.js, etc.)
html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)

# Remove <link rel="stylesheet" ...> and <link rel="preconnect" ...>
html = re.sub(r'<link[^>]*rel=["\'](?:stylesheet|preconnect)["\'][^>]*/?>', '', html)

# Remove canvas elements (charts) — keep surrounding chart-panel divs for context
html = re.sub(r'<canvas[^>]*></canvas>', '<div style="text-align:center;color:#94a3b8;font-size:0.8rem;padding:16px 0;">[Chart — see data table below]</div>', html)

# Inject self-contained PDF styles
pdf_css = """
@page {
    size: A4;
    margin: 0.7in 0.6in;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 8pt;
        color: #94a3b8;
    }
}

* { box-sizing: border-box; }

body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #1e293b;
    background: white;
    margin: 0;
    padding: 0;
}

.header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    color: white;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 24px;
    border-radius: 8px;
    page-break-inside: avoid;
}
.header h1 { font-size: 22pt; font-weight: 800; margin: 0 0 6px; letter-spacing: -0.5px; }
.header .subtitle { font-size: 10pt; opacity: 0.85; font-weight: 400; }
.header-accent { display: none; }

.article-container { max-width: 100%; margin: 0; padding: 0; }

h2 { font-size: 14pt; font-weight: 700; color: #0f172a; margin: 28px 0 10px; padding-bottom: 4px; border-bottom: 2px solid #0EA5E9; page-break-after: avoid; }
h3 { font-size: 11pt; font-weight: 600; color: #0f172a; margin: 20px 0 8px; page-break-after: avoid; }
h4 { font-size: 10pt; font-weight: 600; color: #334155; margin: 12px 0 6px; page-break-after: avoid; }
p, li { font-size: 9.5pt; line-height: 1.65; color: #334155; }
ul { padding-left: 18px; margin: 8px 0; }
li { margin-bottom: 4px; }
a { color: #0EA5E9; text-decoration: none; }
strong { color: #0f172a; }

.key-finding, .opportunity-box, .warning-box { padding: 14px 18px; margin: 16px 0; border-radius: 0 6px 6px 0; font-size: 9pt; page-break-inside: avoid; }
.key-finding { background: #f0f9ff; border-left: 3px solid #0EA5E9; }
.opportunity-box { background: #f0fdf4; border-left: 3px solid #22C55E; }
.warning-box { background: #fef2f2; border-left: 3px solid #EF4444; }

.data-table-wrap { margin: 14px 0; border: 1px solid #e2e8f0; border-radius: 4px; overflow: visible; page-break-inside: avoid; }
.data-table-wrap table { width: 100%; border-collapse: collapse; font-size: 8pt; }
.data-table-wrap th { background: #0f172a; color: white; padding: 7px 10px; text-align: left; font-weight: 600; font-size: 7.5pt; text-transform: uppercase; letter-spacing: 0.04em; }
.data-table-wrap td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; font-size: 8pt; }
.data-table-wrap tr:nth-child(even) { background: #f8fafc; }

.chart-panel-article { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; margin: 14px 0; page-break-inside: avoid; }
.chart-panel-article h4 { margin-top: 0; }
.chart-container-article { display: none; }
.citation { font-size: 7.5pt; color: #94a3b8; font-style: italic; margin-top: 4px; }

.rating-strong { color: #16a34a; font-weight: 700; }
.rating-buy { color: #22C55E; font-weight: 600; }
.rating-hold { color: #F59E0B; font-weight: 600; }
.rating-avoid { color: #EF4444; font-weight: 600; }

.sources-section { margin-top: 28px; padding-top: 14px; border-top: 2px solid #e2e8f0; }
.sources-section ul { font-size: 7.5pt; }
.sources-section li { margin-bottom: 2px; line-height: 1.4; }
.sources-section code { background: #f1f5f9; padding: 1px 4px; border-radius: 2px; font-size: 7pt; }

nav, .page-footer, .bottom-banner, footer { display: none !important; }
h2 { page-break-before: auto; }
"""

# Remove existing <style> blocks (we'll replace with PDF-optimized CSS)
html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)

# Write PDF
OUT.parent.mkdir(parents=True, exist_ok=True)
print("Rendering PDF...")
doc = HTML(string=html, base_url=str(SRC.parent))
doc.write_pdf(str(OUT), stylesheets=[CSS(string=pdf_css)])

size_kb = OUT.stat().st_size / 1024
print(f"PDF saved to: {OUT}")
print(f"Size: {size_kb:.0f} KB ({size_kb/1024:.1f} MB)")
