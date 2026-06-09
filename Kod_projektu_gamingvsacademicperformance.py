import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── 1. WCZYTANIE I PREPROCESSING ────────────────────────────────────────────

df = pd.read_csv("Gaming_Academic_Performance.csv")
print(df.head())
print(df.info())

# Binaryzacja zmiennej docelowej (GPA / academic score)
# Zakładamy kolumnę 'GPA' lub 'academic_performance' – dostosuj nazwę!
TARGET_COL = "grades"  # ← zmień na właściwą nazwę kolumny

median_gpa = df[TARGET_COL].median()
df["target"] = (df[TARGET_COL] >= median_gpa).astype(int)  # 1 = powyżej mediany

# Enkodowanie zmiennych kategorycznych
cat_cols = df.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Podział na cechy i target
FEATURE_COLS = [c for c in df.columns if c not in [TARGET_COL, "target"]]
X = df[FEATURE_COLS].values
y = df["target"].values

# ─── 2. EKSPERYMENT 1 – PORÓWNANIE KLASYFIKATORÓW ────────────────────────────

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True))
    ]),
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score),
    "roc_auc": "roc_auc",
}

results_exp1 = {}
for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    results_exp1[name] = {
        "accuracy": cv_results["test_accuracy"],
        "f1":       cv_results["test_f1"],
        "roc_auc":  cv_results["test_roc_auc"],
    }
    print(f"{name}: ACC={cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")

# Analiza statystyczna – test Wilcoxona między najlepszym a resztą
best = max(results_exp1, key=lambda n: results_exp1[n]["f1"].mean())
print(f"\nNajlepszy model: {best}")
for name, res in results_exp1.items():
    if name != best:
        stat, p = stats.wilcoxon(results_exp1[best]["f1"], res["f1"])
        print(f"  Wilcoxon {best} vs {name}: p={p:.4f} {'*' if p < 0.05 else ''}")

# Wykres wyników
fig, ax = plt.subplots(figsize=(10, 5))
f1_data = [results_exp1[n]["f1"] for n in models]
ax.boxplot(f1_data, labels=models.keys())
ax.set_title("Eksperyment 1: F1-score (10-fold CV)")
ax.set_ylabel("F1-score")
plt.tight_layout()
plt.savefig("results/figures/exp1_classifiers.png", dpi=150)
plt.show()

# ─── 3. EKSPERYMENT 2 – WPŁYW SELEKCJI CECH ─────────────────────────────────

# Zakładamy, że znamy nazwy kolumn – dostosuj do datasetu!
GAMING_COLS   = [c for c in FEATURE_COLS if "game" in c.lower() or "gaming" in c.lower() or "hour" in c.lower()]
DEMOG_COLS    = [c for c in FEATURE_COLS if c not in GAMING_COLS]

feature_sets = {
    "Tylko demograficzne": DEMOG_COLS if DEMOG_COLS else FEATURE_COLS,
    "Tylko gamingowe":     GAMING_COLS if GAMING_COLS else FEATURE_COLS,
    "Wszystkie cechy":     FEATURE_COLS,
}

# Dodaj wariant z selekcją RFE
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_base, n_features_to_select=5)
rfe.fit(X, y)
rfe_cols = [FEATURE_COLS[i] for i, s in enumerate(rfe.support_) if s]
feature_sets["RFE (top 5)"] = rfe_cols
print(f"\nRFE wybrane cechy: {rfe_cols}")

results_exp2 = {}
clf = RandomForestClassifier(n_estimators=100, random_state=42)
for set_name, cols in feature_sets.items():
    idx = [FEATURE_COLS.index(c) for c in cols if c in FEATURE_COLS]
    X_sub = X[:, idx]
    cv_res = cross_validate(clf, X_sub, y, cv=cv, scoring=scoring)
    results_exp2[set_name] = cv_res["test_f1"]
    print(f"{set_name}: F1={cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}")

# Analiza statystyczna
print("\nTesty statystyczne (Wilcoxon) vs 'Wszystkie cechy':")
for name, scores in results_exp2.items():
    if name != "Wszystkie cechy":
        stat, p = stats.wilcoxon(results_exp2["Wszystkie cechy"], scores)
        print(f"  Wszystkie vs {name}: p={p:.4f} {'*' if p < 0.05 else ''}")

# Wykres
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(list(results_exp2.values()), labels=results_exp2.keys())
ax.set_title("Eksperyment 2: Wpływ zestawu cech (RF, 10-fold CV)")
ax.set_ylabel("F1-score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/figures/exp2_features.png", dpi=150)
plt.show()

# ─── 4. FEATURE IMPORTANCE ───────────────────────────────────────────────────
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X, y)
importances = pd.Series(rf_final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
importances.plot(kind="bar", ax=ax)
ax.set_title("Ważność cech (Random Forest)")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("results/figures/feature_importance.png", dpi=150)
plt.show()

print("\nGotowe!")