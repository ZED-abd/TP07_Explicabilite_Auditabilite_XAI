"""Captures reelles de l'interface MLflow et de la page LIME (outil de rendu)."""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "captures"
CAPTURES.mkdir(exist_ok=True)
CHROME = r"C:\Users\zakar\AppData\Local\ms-playwright\chromium-1243\chrome-win64\chrome.exe"
BASE = "http://127.0.0.1:5000"
RUN_ID = (ROOT / "reports" / "run_xai.txt").read_text(encoding="utf-8").strip()


def attendre(page, texte, timeout=25000):
    page.get_by_text(texte, exact=False).first.wait_for(timeout=timeout)
    page.wait_for_timeout(1200)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=CHROME)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})

    # Capture 1 : version du modele, alias production et debut de la Model Card.
    page.goto(f"{BASE}/#/models/fraude/versions/1", wait_until="domcontentloaded")
    attendre(page, "production")
    page.screenshot(path=CAPTURES / "capture1_registre_modele.png", full_page=False)

    # Capture 5a : vue d'ensemble du run XAI (metriques, parametres, tags).
    page.goto(
        f"{BASE}/#/experiments/2/runs/{RUN_ID}", wait_until="domcontentloaded"
    )
    attendre(page, "xai-fraude-v1")
    page.screenshot(path=CAPTURES / "capture5_run_overview.png", full_page=False)

    # Capture 5b : artefacts du run, dossier explications ouvert.
    page.goto(
        f"{BASE}/#/experiments/2/runs/{RUN_ID}/artifacts",
        wait_until="domcontentloaded",
    )
    attendre(page, "explications")
    page.get_by_text("explications", exact=True).first.click()
    attendre(page, "shap_benali.png")
    page.screenshot(path=CAPTURES / "capture5_run_artifacts.png", full_page=False)

    # Capture 6 : tags de la version du modele (Model Card, run d'explications).
    page.goto(f"{BASE}/#/models/fraude/versions/1", wait_until="domcontentloaded")
    attendre(page, "production")
    page.get_by_text("Tags", exact=True).last.scroll_into_view_if_needed()
    page.wait_for_timeout(800)
    page.screenshot(path=CAPTURES / "capture6_tags_modele.png", full_page=False)

    # Capture 4 : page HTML interactive LIME (viewport serre + zoom).
    lime_page = browser.new_page(
        viewport={"width": 1200, "height": 430}, device_scale_factor=2
    )
    lime_page.goto(
        (ROOT / "reports" / "lime_benali.html").as_uri(), wait_until="load"
    )
    lime_page.evaluate("document.body.style.zoom = '1.35'")
    lime_page.wait_for_timeout(1200)
    lime_page.screenshot(
        path=CAPTURES / "capture4_lime_html.png", full_page=False
    )

    browser.close()

print("captures ecrites dans", CAPTURES)
