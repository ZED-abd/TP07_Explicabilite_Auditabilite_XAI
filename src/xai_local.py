"""Explications locales SHAP (activite 3)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from xai_commun import CAS, REPORTS, Xte, m

ex = shap.TreeExplainer(m)
proba = m.predict_proba(Xte)[:, 1]
idx = Xte.index[proba.argmax()]  # transaction de test la plus suspecte

resultats = {}
for nom, x in [("benali", CAS), (f"test_{idx}", Xte.loc[[idx]])]:
    e = ex(x)[0, :, 1]  # 1 transaction, classe fraude
    base = float(np.asarray(e.base_values).ravel()[0])
    contributions = {
        col: round(float(c), 4) for col, c in zip(x.columns, e.values)
    }
    prediction = float(m.predict_proba(x)[0, 1])
    somme = float(base + e.values.sum())

    print(f"--- {nom} :", x.iloc[0].to_dict())
    print("valeur de base :", round(base, 4))
    for col, c in sorted(
        zip(x.columns, e.values), key=lambda t: -abs(t[1])
    ):
        print(f"  {col:11s} {c:+.4f}")
    print(
        "base + somme   :",
        round(somme, 4),
        "| prediction :",
        round(prediction, 4),
        "| ecart :",
        f"{abs(somme - prediction):.2e}",
    )

    shap.plots.waterfall(e, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / f"shap_{nom}.png", dpi=120)
    plt.close("all")

    resultats[nom] = {
        "transaction": x.iloc[0].to_dict(),
        "valeur_de_base": round(base, 4),
        "contributions": contributions,
        "somme_contributions": round(float(e.values.sum()), 4),
        "base_plus_somme": round(somme, 4),
        "prediction": round(prediction, 4),
        "ecart_absolu": abs(somme - prediction),
        "index_test": int(idx) if nom.startswith("test_") else None,
    }

(REPORTS / "shap_local.json").write_text(
    json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("resultats ecrits dans reports/shap_local.json")
