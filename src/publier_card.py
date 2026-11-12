"""Publier la Model Card sur le run XAI et la version du modele (activite 7)."""

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from xai_commun import NOM, REPORTS

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
version = client.get_model_version_by_alias(NOM, "production")
run_id = (REPORTS / "run_xai.txt").read_text(encoding="utf-8").strip()

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact("MODEL_CARD.md")

card = Path("MODEL_CARD.md").read_text(encoding="utf-8")
client.update_model_version(NOM, version.version, description=card)
client.set_model_version_tag(
    NOM, version.version, "model_card", "MODEL_CARD.md, run " + run_id[:8]
)
client.set_model_version_tag(NOM, version.version, "model_card_run", run_id)

# Verification reelle dans le registre et dans le run.
relu = client.get_model_version(NOM, version.version)
artefacts = [a.path for a in client.list_artifacts(run_id)]
print("Model Card attachee a fraude v" + version.version, "et au run", run_id[:8])
print("description commence par :", relu.description.splitlines()[0])
print("tag model_card :", relu.tags.get("model_card"))
print("MODEL_CARD.md dans le run :", "MODEL_CARD.md" in artefacts)
