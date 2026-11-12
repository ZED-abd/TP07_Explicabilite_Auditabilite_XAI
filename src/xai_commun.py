"""Module commun du TP07.

Il charge le modele servi par l'alias ``production`` du registre MLflow,
reconstruit exactement le decoupage du TP2 et expose la transaction de
Mme Benali.

Adaptation documentee : le registre TP2 local contient le modele sous le nom
``fraude-tp2`` (300 arbres, profondeur 12). Le script
``src/preparer_modele_fraude.py`` a enregistre ce meme modele sous le nom
``fraude`` et lui a pose l'alias ``production``, sans reentrainement.
"""

import os
import sys
from pathlib import Path

# MLflow 3 affiche un emoji en fin de run ; la console Windows cp1252 ne peut
# pas l'encoder. On force UTF-8 pour stdout/stderr.
for _flux in (sys.stdout, sys.stderr):
    try:
        _flux.reconfigure(encoding="utf-8")
    except Exception:
        pass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

NOM, ALIAS = "fraude", "production"
VERSION = MlflowClient().get_model_version_by_alias(NOM, ALIAS)
m = mlflow.sklearn.load_model(f"models:/{NOM}@{ALIAS}")

df = pd.read_csv(DATA)
X, y = df.drop(columns="fraude"), df["fraude"]
FEATURES = list(X.columns)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Transaction de Mme Benali : 1 800 MAD, 30 mois d'anciennete,
# 4 transactions sur 24 h, paiement de nuit.
CAS = pd.DataFrame(
    [{"montant": 1800.0, "anciennete": 30, "freq_24h": 4, "nuit": 1}]
)
