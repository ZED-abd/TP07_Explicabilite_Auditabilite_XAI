"""Convertir le rapport Markdown en HTML puis en PDF (Playwright)."""

from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "RAPPORT_TP07.md"
HTML = ROOT / "RAPPORT_TP07.html"
PDF = ROOT / "RAPPORT_TP07.pdf"
CHROME = r"C:\Users\zakar\AppData\Local\ms-playwright\chromium-1243\chrome-win64\chrome.exe"

texte = MD.read_text(encoding="utf-8")
corps = markdown.markdown(
    texte,
    extensions=["tables", "fenced_code", "toc", "nl2br", "sane_lists"],
)

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport TP07 — Explicabilité et auditabilité</title>
<style>
  @page {{ size: A4; margin: 20mm; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #222; }}
  h1 {{ font-size: 17pt; margin-top: 0; border-bottom: 2px solid #2c5aa0; padding-bottom: 4px; }}
  h2 {{ font-size: 13pt; color: #2c5aa0; margin-top: 16px; }}
  h3 {{ font-size: 11.5pt; color: #444; margin-top: 12px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #bbb; padding: 4px 6px; }}
  th {{ background: #eef2f8; }}
  img {{ max-width: 100%; height: auto; margin: 8px 0; border: 1px solid #ddd; }}
  pre, code {{ font-family: "Consolas", monospace; font-size: 9pt; }}
  pre {{ background: #f6f6f6; border: 1px solid #ddd; padding: 8px; overflow-x: auto; white-space: pre-wrap; }}
  blockquote {{ border-left: 3px solid #2c5aa0; margin: 8px 0; padding: 4px 10px; background: #f0f4fa; }}
  ul, ol {{ margin: 6px 0 6px 20px; }}
  hr {{ border: none; border-top: 1px solid #ccc; }}
</style>
</head>
<body>
{corps}
</body>
</html>
"""

HTML.write_text(html, encoding="utf-8")
print("HTML ecrit :", HTML)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=CHROME)
    page = browser.new_page()
    page.goto(HTML.as_uri(), wait_until="load")
    page.wait_for_timeout(1500)
    page.pdf(
        path=str(PDF),
        format="A4",
        print_background=True,
        margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"},
    )
    browser.close()

print("PDF ecrit :", PDF, "—", PDF.stat().st_size, "octets")
