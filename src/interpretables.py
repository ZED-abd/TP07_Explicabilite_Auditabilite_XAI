"""Modeles interpretables et compromis performance / lisibilite (activite 5)."""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from xai_commun import CAS, REPORTS, X, Xte, Xtr, m, yte, ytr

res = {}


def note(nom, modele, Xt):
    pred = modele.predict(Xt)
    res[nom] = {
        "f1": round(float(f1_score(yte, pred)), 4),
        "fraudes_detectees": int(((pred == 1) & (yte == 1)).sum()),
    }
    print(
        f'{nom:28s} F1 = {res[nom]["f1"]:.4f} | fraudes detectees : '
        f'{res[nom]["fraudes_detectees"]} / {int(yte.sum())}'
    )


def logm(d):
    """Le montant agit par ordre de grandeur."""
    d = d.copy()
    d["log_montant"] = np.log10(d.pop("montant"))
    return d


note("foret fraude@production", m, Xte)

lr_brut = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
    Xtr, ytr
)
note("logistique, montant brut", lr_brut, Xte)

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
    logm(Xtr), ytr
)
note("logistique, log10(montant)", lr, logm(Xte))

coef = {
    c: round(float(v), 2)
    for c, v in zip(logm(Xtr).columns, lr[-1].coef_[0])
}
print("coefficients standardises :", coef)

cote = np.exp(lr[-1].coef_[0] / lr[0].scale_)
print(
    "cote x",
    round(cote[0], 3),
    "par mois d'anciennete ; x",
    round(cote[1], 2),
    "par transaction en 24 h",
)

arbres = {}
for p in [2, 3, 6, None]:
    t = DecisionTreeClassifier(max_depth=p, random_state=0).fit(Xtr, ytr)
    note(f"arbre profondeur {p} ({t.get_n_leaves()} feuilles)", t, Xte)
    arbres[str(p)] = {
        "profondeur": p,
        "feuilles": int(t.get_n_leaves()),
        "f1": res[f"arbre profondeur {p} ({t.get_n_leaves()} feuilles)"]["f1"],
        "fraudes_detectees": res[
            f"arbre profondeur {p} ({t.get_n_leaves()} feuilles)"
        ]["fraudes_detectees"],
    }

t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
regles = export_text(t3, feature_names=list(X.columns), decimals=1)
print(regles)
(REPORTS / "arbre_profondeur3.txt").write_text(regles, encoding="utf-8")

feuille_benali = int(t3.apply(CAS)[0])
proba_benali = float(t3.predict_proba(CAS)[0, 1])
print(
    "Mme Benali : feuille/noeud",
    feuille_benali,
    "| P =",
    round(proba_benali, 3),
)

resultat = {
    "modeles": res,
    "coefficients_standardises": coef,
    "cotes": {
        "anciennete_par_mois": round(float(cote[0]), 3),
        "freq_24h_par_transaction": round(float(cote[1]), 2),
    },
    "arbres": arbres,
    "benali_arbre_profondeur3": {
        "feuille": feuille_benali,
        "probabilite": round(proba_benali, 3),
    },
}
(REPORTS / "interpretables.json").write_text(
    json.dumps(resultat, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("resultats ecrits dans reports/interpretables.json")
