#!/usr/bin/env python3
"""Build a self-contained standalone HTML from the dashboard.

Reads dashboard.html, downloads Chart.js + annotation plugin (or extracts
from existing standalone), embeds overprocure_results.json, and removes
Google Fonts CDN links so the result works fully offline.
"""
import json, re, os, urllib.request

DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Read source files ----
with open(os.path.join(DIR, 'dashboard.html'), 'r', encoding='utf-8') as f:
    html = f.read()

with open(os.path.join(DIR, 'overprocure_results.json'), 'r', encoding='utf-8') as f:
    json_data = f.read().strip()

# ---- Get Chart.js + annotation plugin ----
# Try to extract from existing standalone first; fall back to CDN download
existing_standalone = os.path.join(DIR, 'dashboard_standalone.html')
chartjs_script = None
annotation_script = None

if os.path.exists(existing_standalone):
    with open(existing_standalone, 'r', encoding='utf-8') as f:
        orig = f.read()
    m = re.search(r'<script>/\*\*\s*\n\s*\*\s*Skipped minification.*?</script>', orig, re.DOTALL)
    if not m:
        m = re.search(r'<script>.*?Chart\.js v4.*?</script>', orig, re.DOTALL)
    if m:
        chartjs_script = m.group(0)
    m = re.search(r'<script>/\*!\s*\n\s*\*\s*chartjs-plugin-annotation.*?</script>', orig, re.DOTALL)
    if not m:
        m = re.search(r'<script>.*?chartjs-plugin-annotation.*?</script>', orig, re.DOTALL)
    if m:
        annotation_script = m.group(0)

if not chartjs_script:
    print("Downloading Chart.js from CDN...")
    data = urllib.request.urlopen('https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js').read().decode()
    chartjs_script = f'<script>{data}</script>'

if not annotation_script:
    print("Downloading chartjs-plugin-annotation from CDN...")
    data = urllib.request.urlopen('https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js').read().decode()
    annotation_script = f'<script>{data}</script>'

print(f"Chart.js script: {len(chartjs_script):,} chars")
print(f"Annotation plugin: {len(annotation_script):,} chars")

# ---- 1. Remove Google Fonts CDN links ----
html = html.replace('<link rel="preconnect" href="https://fonts.googleapis.com">\n', '')
html = html.replace('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n', '')
html = re.sub(r'<link href="https://fonts\.googleapis\.com/css2\?family=Lora[^"]*" rel="stylesheet">\n',
              '<!-- Fonts: Falls back to system fonts offline -->\n', html)

# ---- 2. Replace CDN script tags with inlined versions ----
html = html.replace(
    '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>',
    chartjs_script
)
html = html.replace(
    '<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>',
    annotation_script
)

# ---- 3. Embed JSON data inline ----
html = html.replace(
    "const resp = await fetch('overprocure_results.json');\n        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);\n        DATA = await resp.json();",
    f"DATA = {json_data};"
)

# ---- 4. Remove methodology link (no external page in standalone) ----
html = html.replace(
    "window.open('optimizer_methodology.html','_blank')",
    "alert('Methodology documentation is available in the full project distribution.')"
)

# ---- Save ----
out_path = os.path.join(DIR, 'dashboard_standalone.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nStandalone built: {len(html):,} bytes")
print(f"Saved to: {out_path}")
