#!/usr/bin/env python3
"""Build the single self-contained docs/index.html from the template + kernels.json.

GitHub Pages serves a lone index.html fine; this inlines window.KERNEL_DATA so the
page has zero external JS/JSON dependencies (one file, one request).
"""
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parent
data = json.loads((DOCS / "data" / "kernels.json").read_text())
template = (DOCS / "_dashboard_template.html").read_text()

# compact JSON (no spaces) keeps the inlined payload small
payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
html = template.replace("__KERNEL_DATA__", payload)

out = DOCS / "index.html"
out.write_text(html, encoding="utf-8")
kb = len(html.encode("utf-8")) / 1024
print(f"wrote {out} ({kb:.0f} KB, {len(data['kernels'])} kernels, "
      f"{data['multishape_summary']['total_shapes']} shapes)")
