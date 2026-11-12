"""Rendre le modele du TP2 disponible sous fraude@production.

Le registre TP2 local contient le modele sous le nom ``fraude-tp2`` et ne
possede pas d'alias. Ce script ne reentraine rien : il enregistre le meme
modele (meme source, meme run) sous le nom attendu par le sujet, puis pose
l'alias ``production``. Il est idempotent.
"""

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

SOURCE_NOM = "fraude-tp2"
NOM = "fraude"
ALIAS = "production"

# Derniere version du modele TP2 (la version 2 dans le registre local).
versions = client.search_model_versions(f"name='{SOURCE_NOM}'")
if not versions:
    raise SystemExit(f"Modele source {SOURCE_NOM} introuvable dans le registre.")
source = sorted(versions, key=lambda v: int(v.version))[-1]
print(f"Source : {SOURCE_NOM} v{source.version} | run {source.run_id}")

try:
    cible = client.get_registered_model(NOM)
    print(f"Le modele {NOM} existe deja.")
except Exception:
    cible = None

if cible is None:
    mv = mlflow.register_model(f"models:/{SOURCE_NOM}/{source.version}", NOM)
    print(f"Modele {NOM} v{mv.version} cree depuis {SOURCE_NOM} v{source.version}.")
else:
    try:
        mv = client.get_model_version_by_alias(NOM, ALIAS)
        print(f"Alias {NOM}@{ALIAS} deja present (v{mv.version}).")
    except Exception:
        latest = sorted(
            client.search_model_versions(f"name='{NOM}'"),
            key=lambda v: int(v.version),
        )[-1]
        mv = latest
        print(f"Modele {NOM} existant, version retenue : v{mv.version}.")

client.set_registered_model_alias(NOM, ALIAS, mv.version)
print(f"Alias {NOM}@{ALIAS} -> version {mv.version} | run {mv.run_id}")
