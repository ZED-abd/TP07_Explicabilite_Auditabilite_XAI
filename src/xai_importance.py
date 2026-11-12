"""Importance globale : MDI et permutation (activite 2)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from xai_commun import REPORTS, X, Xte, m, yte

r = permutation_importance(
    m, Xte, yte, scoring="f1", n_repeats=10, random_state=0, n_jobs=1
)

t = pd.DataFrame(
    {
        "MDI (foret)": m.feature_importances_,
        "permutation": r.importances_mean,
        "ecart-type": r.importances_std,
    },
    index=X.columns,
).round(3)
t = t.sort_values("permutation", ascending=False)
print(t.to_string())
t.to_json(REPORTS / "importance.json")

ax = t["permutation"].plot.barh(xerr=t["ecart-type"], color="#4592EF")
ax.set_xlabel("perte de F1 quand la colonne est melangee")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(REPORTS / "permutation.png", dpi=120)
plt.close("all")
print("importance ecrite dans reports/importance.json et reports/permutation.png")
