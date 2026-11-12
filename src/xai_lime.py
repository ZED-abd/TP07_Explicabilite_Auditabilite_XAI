"""Explication locale LIME et etude de stabilite (activite 4)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from xai_commun import CAS, REPORTS, X, Xtr, m

COLONNES = list(X.columns)


def expliquer(graine, n):
    ex = LimeTabularExplainer(
        Xtr.values,
        feature_names=COLONNES,
        class_names=["normal", "fraude"],
        mode="classification",
        random_state=graine,
    )
    return ex.explain_instance(
        CAS.values[0], m.predict_proba, num_features=4, num_samples=n
    )


def poids(explanation):
    """Associe chaque variable a son poids LIME (regle contenant son nom)."""
    regles = dict(explanation.as_list())
    return {
        col: float(next((v for k, v in regles.items() if col in k), 0.0))
        for col in COLONNES
    }


# Explication de reference : 2 000 voisins, graine 0.
e = expliquer(0, 2000)
print("regles :", [(r, round(w, 3)) for r, w in e.as_list()])
e.save_to_file(str(REPORTS / "lime_benali.html"))
fig = e.as_pyplot_figure()
fig.tight_layout()
fig.savefig(REPORTS / "lime_benali.png", dpi=120)
plt.close(fig)

# Etude de stabilite : deux tailles de voisinage, cinq graines.
lignes = []
for n in [500, 2000]:
    print(f"--- {n} voisins     montant    nuit  anciennete  freq_24h")
    for graine in range(5):
        w = poids(expliquer(graine, n))
        lignes.append({"n_voisins": n, "graine": graine, **w})
        print(
            f"graine {graine}       {w['montant']:+.3f}  {w['nuit']:+.3f}"
            f"     {w['anciennete']:+.3f}    {w['freq_24h']:+.3f}"
        )

# Statistiques par taille de voisinage.
stats = {}
for n in [500, 2000]:
    stats[str(n)] = {}
    for col in COLONNES:
        vals = np.array(
            [r[col] for r in lignes if r["n_voisins"] == n], dtype=float
        )
        signes = np.sign(vals)
        stats[str(n)][col] = {
            "moyenne": round(float(vals.mean()), 4),
            "ecart_type": round(float(vals.std(ddof=0)), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "changements_de_signe": int(np.sum(signes[1:] != signes[:-1])),
            "signe_stable": bool(np.all(signes == signes[0])),
        }

(REPORTS / "lime_stabilite.json").write_text(
    json.dumps({"lignes": lignes, "statistiques": stats}, indent=2),
    encoding="utf-8",
)

# Comparaison avec SHAP (activite 3, graine 0, 2 000 voisins).
shap_local = json.loads((REPORTS / "shap_local.json").read_text(encoding="utf-8"))
shap_benali = shap_local["benali"]["contributions"]
lime_benali = {
    col: round(float(v), 4)
    for col, v in poids(e).items()
}
comparaison = {
    "shap": shap_benali,
    "lime_2000_graine0": lime_benali,
    "meme_signe": {
        col: bool(np.sign(shap_benali[col]) == np.sign(lime_benali[col]))
        for col in COLONNES
    },
    "rang_shap": sorted(COLONNES, key=lambda c: -abs(shap_benali[c])),
    "rang_lime": sorted(COLONNES, key=lambda c: -abs(lime_benali[c])),
}
(REPORTS / "comparaison_shap_lime.json").write_text(
    json.dumps(comparaison, indent=2), encoding="utf-8"
)

print("\nComparaison SHAP / LIME (Mme Benali) :")
print(f"{'variable':12s} {'SHAP':>9s} {'LIME':>9s} {'meme signe':>11s}")
for col in COLONNES:
    print(
        f"{col:12s} {shap_benali[col]:+9.3f} {lime_benali[col]:+9.3f}"
        f" {str(comparaison['meme_signe'][col]):>11s}"
    )
print("resultats ecrits dans reports/lime_stabilite.json et comparaison_shap_lime.json")
