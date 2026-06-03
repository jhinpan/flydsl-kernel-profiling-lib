#!/usr/bin/env python3
"""Build the single self-contained docs/index.html from the template + kernels.json.

GitHub Pages serves a lone index.html fine; this inlines window.KERNEL_DATA so the
page has zero external JS/JSON dependencies (one file, one request).
"""
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parent
# explicit utf-8: the template + data carry non-ASCII (Chinese i18n, ·, —), which
# would break read_text() on a non-utf-8 default locale (e.g. Windows cp1252).
data = json.loads((DOCS / "data" / "kernels.json").read_text(encoding="utf-8"))
template = (DOCS / "_dashboard_template.html").read_text(encoding="utf-8")

# compact JSON (no spaces) keeps the inlined payload small
payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
# Neutralize sequences that would prematurely break out of the inline <script>
# block if they ever appear in a data string (e.g. a kernel name with "</script>").
# The backslash escapes survive JS string parsing, so the data is byte-identical
# after the browser evaluates window.KERNEL_DATA.
payload = (payload.replace("</", "<\\/").replace("<!--", "<\\!--").replace("<script", "<\\script"))
html = template.replace("__KERNEL_DATA__", payload)

out = DOCS / "index.html"
out.write_text(html, encoding="utf-8")
kb = len(html.encode("utf-8")) / 1024
print(f"wrote {out} ({kb:.0f} KB, {len(data['kernels'])} kernels, "
      f"{data['multishape_summary']['total_shapes']} shapes)")
