# Model Card — fraude (version servie par l'alias `production`)

## Modèle

- **Nom et version dans le registre MLflow** : `fraude`, version **1**, alias `production`.
- **Famille et réglages** : `RandomForestClassifier` (scikit-learn 1.9.1), **300 arbres**, profondeur maximale **12**, `random_state=42`.
- **Run d'origine (MLflow)** : `8d553d60d4ad4f16b6d425afd1f8f593` (expérience `tp2-pipeline`). Le modèle TP2 était enregistré sous `fraude-tp2` v2 ; il a été réenregistré sous `fraude` v1 sans réentraînement, puis l'alias `production` lui a été posé.

## Objectif et usage prévu

Détecter automatiquement les transactions à risque de fraude pour aider un conseiller à prioriser une revue manuelle. Le modèle produit une probabilité de fraude ; il ne décide pas seul d'un blocage définitif.

## Usages exclus

- Bloquer définitivement un compte ou un paiement sans revue humaine.
- Utiliser le score comme preuve de fraude ou comme décision juridique.
- Appliquer le modèle à des populations, produits ou zones non représentés dans les données d'entraînement.
- Utiliser les explications SHAP/LIME comme preuve de causalité.

## Données

- **Fichier** : `data/transactions.csv`, **5 000 transactions**, **568 fraudes (11,36 %)**, 4 variables : `montant`, `anciennete`, `freq_24h`, `nuit`.
- **Empreinte MD5** (tag `donnees_md5`) : `cf9f6f86c141e7df32c61715c493483f`.
- **Origine** : données **synthétiques** générées par `donnees.py` du TP2 (graine 7), avec un découpage train/test `test_size=0.25`, `random_state=42` (1 250 transactions de test).

## Performances (jeu de test, 1 250 transactions)

- **F1** : 0,8636 — **Précision** : 0,8769 — **Rappel** : 0,8507 — **AUC** : 0,9884.
- **Matrice de confusion** : `[[1100, 16], [20, 114]]` (1 100 vrais négatifs, 16 faux positifs, 20 faux négatifs, 114 vrais positifs).
- **Mme Benali** : P(fraude) = **0,8899** (transaction 1 800 MAD, 30 mois, 4 transactions/24 h, nuit).

## Méthodes d'explicabilité

- **Globale** : importance MDI, importance par permutation (10 répétitions, perte de F1) et SHAP `TreeExplainer` sur 400 transactions de test.
- **Locale** : SHAP (waterfall) pour Mme Benali et pour la transaction de test la plus suspecte ; LIME avec 500 et 2 000 voisins, 5 graines.
- **Variables les plus influentes** : `montant` (permutation 0,591 ; SHAP moyen |valeur| 0,1155), puis `freq_24h` (0,265 ; 0,0555), `anciennete` (0,263 ; 0,0578) et `nuit` (0,193 ; 0,0572). Pour Mme Benali, SHAP place `nuit` (+0,3677) puis `montant` (+0,3555) en tête ; LIME place `montant` (+0,4834) puis `nuit` (+0,1821).
- **Limites des méthodes** : LIME varie selon la graine et le voisinage ; SHAP explique le modèle, pas la causalité.

## Comparaison avec un modèle interprétable

| Modèle | F1 | Fraudes détectées |
|---|---:|---:|
| Forêt `fraude@production` | 0,8636 | 114 / 134 |
| Régression logistique, montant brut | 0,7407 | 90 / 134 |
| Régression logistique, `log10(montant)` | 0,8593 | 113 / 134 |
| Arbre profondeur 2 (4 feuilles) | 0,2293 | 18 / 134 |
| Arbre profondeur 3 (8 feuilles) | 0,5737 | 72 / 134 |
| Arbre profondeur 6 (50 feuilles) | 0,7510 | 95 / 134 |
| Arbre sans limite (206 feuilles) | 0,7556 | 102 / 134 |

La régression logistique avec `log10(montant)` retrouve presque la forêt (0,8593 contre 0,8636, soit 1 fraude de moins) tout en restant lisible. L'arbre de profondeur 3 est très lisible mais perd 42 fraudes.

## Limites et risques connus

- **Données synthétiques** : les résultats ne valent pas validation sur des transactions réelles ; les fraudes rares et les nouveaux schémas ne sont pas représentés.
- **Peu de variables** : le modèle ignore l'historique client, le commerçant, le pays, le moyen de paiement et la temporalité fine ; une variable omise peut créer un biais.
- **Déséquilibre** : 568 fraudes sur 5 000 ; le test ne contient que 134 fraudes, donc les écarts de F1 de l'ordre de 0,005 à 0,01 sont dans le bruit.
- **Seuil par défaut 0,5** : il privilégie la précision ; un seuil plus bas augmenterait le rappel au prix des faux positifs.
- **Explications non causales** : SHAP et LIME décrivent le comportement du modèle, pas la cause réelle de la fraude ; LIME peut changer de signe.
- **Version locale** : le modèle servi est la forêt TP2 300/12 (F1 0,8636), et non la forêt 500/16 de référence (F1 0,8669) ; les identifiants de run diffèrent.

## Recommandations d'utilisation et de surveillance

- **Revue humaine** obligatoire avant tout blocage définitif ; le score sert à prioriser, pas à condamner.
- **Surveiller** chaque mois : F1, précision, rappel, matrice de confusion, taux de faux positifs et faux négatifs, et dérive des distributions de `montant`, `anciennete`, `freq_24h`, `nuit`.
- **Recalculer** l'importance SHAP globale après chaque réentraînement ou changement d'alias, et vérifier que les variables importantes restent autorisées et stables.
- **Alerter** si le F1 chute sous le seuil de validation du TP2 (0,82) ou si le rappel baisse de plus de 5 points.
- **Documenter** chaque version : run d'entraînement, empreinte des données, run d'explications, Model Card et décision de mise en production.

## Traçabilité

- **Run des explications** (expérience `tp7-explicabilite`) : `84c264e9b80f4930804598a458e91a6d`, nom `xai-fraude-v1` ; tag `explications_run` posé sur `fraude` v1.
- **Run d'origine du modèle** : `8d553d60d4ad4f16b6d425afd1f8f593`.
- **Données** : `data/transactions.csv`, MD5 `cf9f6f86c141e7df32c61715c493483f`.
- **Auteurs** : Zakaria Abdelbaki — 5IASD — groupe AI_G6.
- **Date** : 12 novembre 2026 — **Prochaine revue** : 12 février 2027.
