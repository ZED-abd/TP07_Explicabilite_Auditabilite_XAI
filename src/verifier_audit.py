"""Verification finale de la chaine modele -> donnees -> explications -> Model Card."""

import hashlib
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from xai_commun import DATA, NOM, REPORTS, ROOT, VERSION

client = MlflowClient()
resultats = {}


def verifier(nom, condition, detail=""):
    resultats[nom] = {"ok": bool(condition), "detail": detail}
    print(("PASS" if condition else "FAIL"), "-", nom, ":", detail)


# 1. Modele et alias.
mv = client.get_model_version_by_alias(NOM, "production")
verifier("alias_production", mv.version == VERSION.version, f"{NOM}@production = v{mv.version}")
verifier(
    "run_origine",
    mv.run_id == "8d553d60d4ad4f16b6d425afd1f8f593",
    mv.run_id,
)

# 2. Run d'explications.
run_id = (REPORTS / "run_xai.txt").read_text(encoding="utf-8").strip()
run = client.get_run(run_id)
verifier("run_xai_finished", run.info.status == "FINISHED", run.info.status)
verifier("tag_modele_version", run.data.tags.get("modele_version") == VERSION.version, run.data.tags.get("modele_version"))
verifier("tag_modele_run_id", run.data.tags.get("modele_run_id") == mv.run_id, run.data.tags.get("modele_run_id"))
verifier("tag_methodes", "SHAP" in run.data.tags.get("methodes", ""), run.data.tags.get("methodes", ""))
verifier("param_shap", run.data.params.get("shap_echantillon") == "400", run.data.params.get("shap_echantillon"))
verifier("param_lime", run.data.params.get("lime_num_samples") == "2000", run.data.params.get("lime_num_samples"))
verifier("metrique_f1", abs(run.data.metrics.get("f1", 0) - 0.8636363636363636) < 1e-9, str(run.data.metrics.get("f1")))

# 3. Empreinte des donnees.
md5 = hashlib.md5(DATA.read_bytes()).hexdigest()
verifier("md5_donnees", md5 == run.data.tags.get("donnees_md5"), md5)
verifier("md5_modele", md5 == mv.tags.get("donnees_md5"), mv.tags.get("donnees_md5"))

# 4. Artefacts.
artefacts = [
    Path(a.path).name for a in client.list_artifacts(run_id, "explications")
]
attendus = {
    "shap_global.png",
    "shap_benali.png",
    "shap_test_3194.png",
    "lime_benali.html",
    "lime_benali.png",
    "importance.json",
    "shap_importance.json",
    "shap_local.json",
    "lime_stabilite.json",
    "comparaison_shap_lime.json",
    "interpretables.json",
    "verification.json",
    "manifeste.json",
}
verifier("artefacts_explications", attendus.issubset(set(artefacts)), f"{len(artefacts)} fichiers")
verifier("model_card_dans_run", "MODEL_CARD.md" in [a.path for a in client.list_artifacts(run_id)], "MODEL_CARD.md")

# 5. Tag et description de la Model Card.
verifier("tag_model_card", "MODEL_CARD.md" in mv.tags.get("model_card", ""), mv.tags.get("model_card", ""))
verifier("description_model_card", mv.description.startswith("# Model Card"), mv.description.splitlines()[0])
verifier("tag_explications_run", mv.tags.get("explications_run") == run_id, mv.tags.get("explications_run"))

# 6. Efficacite SHAP : base + somme des contributions = prediction.
shap_local = json.loads((REPORTS / "shap_local.json").read_text(encoding="utf-8"))
for nom, bloc in shap_local.items():
    ecart = abs(bloc["base_plus_somme"] - bloc["prediction"])
    verifier(f"efficacite_shap_{nom}", ecart < 1e-9, f"ecart = {ecart:.2e}")

# 7. Fichiers et tables.
for fichier in [
    "verification.json",
    "importance.json",
    "shap_importance.json",
    "shap_local.json",
    "lime_stabilite.json",
    "comparaison_shap_lime.json",
    "interpretables.json",
    "shap_global.png",
    "shap_benali.png",
    "lime_benali.html",
    "lime_benali.png",
]:
    p = REPORTS / fichier
    verifier(f"fichier_{fichier}", p.exists() and p.stat().st_size > 50, f"{p.stat().st_size if p.exists() else 0} octets")

lime = json.loads((REPORTS / "lime_stabilite.json").read_text(encoding="utf-8"))
verifier("lime_10_explications", len(lime["lignes"]) == 10, f"{len(lime['lignes'])} lignes")

card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
for rubrique in [
    "## Modèle",
    "## Objectif et usage prévu",
    "## Usages exclus",
    "## Données",
    "## Performances",
    "## Méthodes d'explicabilité",
    "## Comparaison avec un modèle interprétable",
    "## Limites et risques connus",
    "## Recommandations d'utilisation et de surveillance",
    "## Traçabilité",
]:
    verifier(f"model_card_{rubrique[3:].strip()}", rubrique in card, "presente")

(REPORTS / "audit_mlflow.json").write_text(
    json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8"
)
echecs = [k for k, v in resultats.items() if not v["ok"]]
print("\nBILAN :", len(resultats) - len(echecs), "verifications OK,", len(echecs), "echec(s)")
if echecs:
    print("Echecs :", echecs)
