"""Journalisation des explications dans MLflow (activite 6)."""

import hashlib
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score

from xai_commun import (
    ALIAS,
    DATA,
    NOM,
    REPORTS,
    VERSION,
    m,
    Xte,
    y,
    yte,
)


def empreinte(chemin):
    """Empreinte MD5, meme principe que DVC."""
    return hashlib.md5(Path(chemin).read_bytes()).hexdigest()


pred = m.predict(Xte)
md5 = empreinte(DATA)

# Manifeste des fichiers d'explication (nom, taille, sha256) pour l'audit.
manifeste = []
for f in sorted(REPORTS.glob("*.*")):
    if f.suffix not in (".png", ".html", ".json"):
        continue
    manifeste.append(
        {
            "fichier": f.name,
            "octets": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
    )
(REPORTS / "manifeste.json").write_text(
    json.dumps(manifeste, indent=2), encoding="utf-8"
)

mlflow.set_experiment("tp7-explicabilite")
with mlflow.start_run(run_name=f"xai-{NOM}-v{VERSION.version}") as run:
    mlflow.set_tags(
        {
            "modele": NOM,
            "alias": ALIAS,
            "modele_version": VERSION.version,
            "modele_run_id": VERSION.run_id,
            "donnees": "data/transactions.csv",
            "donnees_md5": md5,
            "donnees_lignes": str(len(y)),
            "donnees_fraudes": str(int(y.sum())),
            "methodes": "permutation ; SHAP TreeExplainer ; LIME",
            "environnement": "python 3.13.3 ; mlflow 3.17.0 ; scikit-learn 1.9.1",
        }
    )
    mlflow.log_params(
        {
            "shap_echantillon": 400,
            "lime_num_samples": 2000,
            "lime_random_state": 0,
            "permutation_n_repeats": 10,
        }
    )
    mlflow.log_metrics(
        {
            "f1": float(f1_score(yte, pred)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
        }
    )
    for f in sorted(REPORTS.glob("*.*")):
        if f.suffix not in (".png", ".html", ".json"):
            continue
        mlflow.log_artifact(str(f), artifact_path="explications")
    mlflow.log_artifact("requirements_modele.txt")
    (REPORTS / "run_xai.txt").write_text(run.info.run_id, encoding="utf-8")

client = MlflowClient()
client.set_model_version_tag(
    NOM, VERSION.version, "explications_run", run.info.run_id
)
client.set_model_version_tag(NOM, VERSION.version, "donnees_md5", md5)

print(
    "run",
    run.info.run_id[:8],
    ": explications du modele",
    NOM,
    "v" + VERSION.version,
    "archivees",
)
print("md5 donnees :", md5)
print("artefacts :", len(list(client.list_artifacts(run.info.run_id, "explications"))))
print("tag explications_run :", client.get_model_version(NOM, VERSION.version).tags.get("explications_run"))
