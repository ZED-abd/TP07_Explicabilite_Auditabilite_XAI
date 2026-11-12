# TP07 — Explicabilité et auditabilité du modèle de fraude

Projet `tp7-xai` réalisé dans le cadre du cours **Déploiement d'IA (2026-2027)** — EMSI, 5IASD.

## Objectif

Charger le modèle de détection de fraude depuis le registre MLflow, l'expliquer globalement
(permutation, SHAP) et localement (SHAP, LIME), comparer avec des modèles interprétables,
archiver les explications dans MLflow et publier une Model Card attachée à la version du modèle.

## Prérequis

- Python 3.13.3
- Le registre MLflow du TP2 : `mlflow.db` et `mlartifacts/` (serveur sur `http://127.0.0.1:5000`)
- Le fichier `data/transactions.csv` du TP2 (5 000 transactions, 568 fraudes)

## Adaptation du registre TP2

Le registre TP2 local contient le modèle sous le nom **`fraude-tp2`**, version 2
(`RandomForestClassifier`, 300 arbres, profondeur 12, run `8d553d60d4ad4f16b6d425afd1f8f593`,
F1 = 0,8636), sans alias. Le sujet attend `fraude@production`.

Le script `src/preparer_modele_fraude.py` enregistre **le même modèle** (même source, même run,
sans réentraînement) sous le nom `fraude`, version 1, puis pose l'alias `production`.
Les valeurs de référence diffèrent donc légèrement de celles du PDF (forêt 500/16, F1 0,8669) :
c'est autorisé par le sujet, qui demande de relever les chiffres de la version réellement servie.

## Installation

```powershell
cd tp7-xai
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_modele.txt
pip install shap lime matplotlib
```

## Exécution

Terminal 1 (serveur MLflow du TP2) :

```powershell
cd C:\Users\zakar\Downloads\mlops-workspace\tp2-mlflow
.venv\Scripts\Activate.ps1
mlflow server --host 127.0.0.1 --port 5000 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlartifacts
```

Terminal 2 (projet TP07) :

```powershell
cd tp7-xai
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"   # evite l'erreur cp1252 sur l'emoji de fin de run MLflow

python src/preparer_modele_fraude.py
python src/verifier_modele.py
python src/xai_importance.py
python src/xai_global.py
python src/xai_local.py
python src/xai_lime.py
python src/interpretables.py
python src/journaliser.py
python src/publier_card.py
python src/verifier_audit.py
```

## Résultats principaux

- Modèle : `fraude` v1, alias `production`, forêt 300 arbres / profondeur 12.
- Test : F1 = 0,8636 ; précision = 0,8769 ; rappel = 0,8507 ; AUC = 0,9884.
- Mme Benali : P(fraude) = 0,8899 ; SHAP : nuit +0,3677, montant +0,3555.
- SHAP global : montant 0,1155 > anciennete 0,0578 > nuit 0,0572 > freq_24h 0,0555.
- LIME (2 000 voisins, graine 0) : montant +0,4834, nuit +0,1821.
- Run d'explications : `84c264e9b80f4930804598a458e91a6d` (expérience `tp7-explicabilite`).
- Audit final : 40 vérifications réussies, 0 échec.

## Structure

```text
tp7-xai/
├── .gitignore
├── requirements_modele.txt
├── MODEL_CARD.md
├── README.md
├── RAPPORT_TP07.md
├── src/
│   ├── xai_commun.py
│   ├── preparer_modele_fraude.py
│   ├── verifier_modele.py
│   ├── xai_importance.py
│   ├── xai_global.py
│   ├── xai_local.py
│   ├── xai_lime.py
│   ├── interpretables.py
│   ├── journaliser.py
│   ├── publier_card.py
│   └── verifier_audit.py
├── data/transactions.csv
├── reports/ (JSON, PNG, HTML, run_xai.txt)
├── captures/ (captures 1 a 6)
└── scripts/prendre_captures.py
```
