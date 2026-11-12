"""Verification du modele fraude@production (activite 1)."""

import json

import mlflow
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xai_commun import ALIAS, CAS, NOM, REPORTS, VERSION, m, Xte, yte

run = mlflow.get_run(VERSION.run_id)
pred = m.predict(Xte)
proba = m.predict_proba(Xte)[:, 1]

# Le run parent du TP2 ne contient pas les hyperparametres : ils sont dans
# le run enfant "entrainer". On lit donc aussi les attributs du modele.
n_estimators = run.data.params.get("n_estimators", getattr(m, "n_estimators", None))
max_depth = run.data.params.get("max_depth", getattr(m, "max_depth", None))

f1 = f1_score(yte, pred)
precision = precision_score(yte, pred, zero_division=0)
recall = recall_score(yte, pred, zero_division=0)
auc = roc_auc_score(yte, proba)
matrice = confusion_matrix(yte, pred).tolist()
proba_benali = float(m.predict_proba(CAS)[0, 1])

print("modele     :", NOM, "version", VERSION.version, "| run", VERSION.run_id[:8])
print(
    "classe     :",
    type(m).__name__,
    "|",
    {"n_estimators": str(n_estimators), "max_depth": str(max_depth)},
)
print(
    "F1         :",
    round(f1, 4),
    "| F1 enregistre dans le run :",
    round(run.data.metrics.get("f1", float("nan")), 4),
)
print(
    "precision  :",
    round(precision, 4),
    "| rappel :",
    round(recall, 4),
    "| AUC :",
    round(auc, 4),
)
print("matrice    :", matrice)
print("Mme Benali : P(fraude) =", round(proba_benali, 4))

verification = {
    "modele": NOM,
    "alias": ALIAS,
    "version": VERSION.version,
    "run_id": VERSION.run_id,
    "classe": type(m).__name__,
    "n_estimators": int(n_estimators) if n_estimators is not None else None,
    "max_depth": int(max_depth) if max_depth is not None else None,
    "f1": round(float(f1), 4),
    "f1_run": round(float(run.data.metrics.get("f1", float("nan"))), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "auc": round(float(auc), 4),
    "matrice_confusion": matrice,
    "benali_proba": round(proba_benali, 4),
    "n_test": int(len(yte)),
    "n_fraudes_test": int(yte.sum()),
}
(REPORTS / "verification.json").write_text(
    json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("verification ecrite dans reports/verification.json")
