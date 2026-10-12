import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.preprocessing import StandardScaler

# Import the ORIGINAL project's data loader and settings
import run_experiments_cpu as exp


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "fig4a_shap_xgboost.png"
)

IMPORTANCE_FILE = os.path.join(
    RESULTS_DIR,
    "shap_xgboost_feature_importance.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA USING THE ORIGINAL PROJECT FUNCTION
# ============================================================

print("=" * 70)
print("XGBOOST SHAP ONLY")
print("=" * 70)

print("\nLoading data using original load_real()...")

X, y, step, src, dst = exp.load_real(DATA_DIR)

print("Original X shape:", X.shape)

if X.shape[1] != 165:
    raise ValueError(
        f"Expected 165 features, found {X.shape[1]}"
    )


# ============================================================
# ORIGINAL TRAIN / TEST SPLIT
# ============================================================

labelled = np.where(y >= 0)[0]

tr = labelled[
    (step[labelled] >= exp.TRAIN_STEPS[0])
    &
    (step[labelled] <= exp.TRAIN_STEPS[1])
]

te = labelled[
    (step[labelled] >= exp.TEST_STEPS[0])
    &
    (step[labelled] <= exp.TEST_STEPS[1])
]

ytr = y[tr]
yte = y[te]

print("\nTRAIN / TEST")
print("Training:", len(tr))
print("Testing :", len(te))

print("Training illicit:", int(ytr.sum()))
print("Training licit  :", int((ytr == 0).sum()))

print("Testing illicit :", int(yte.sum()))
print("Testing licit   :", int((yte == 0).sum()))


# ============================================================
# EXACT ORIGINAL SCALING
# ============================================================

print("\nFitting original StandardScaler...")

scaler = StandardScaler().fit(X[tr])

Xs = scaler.transform(X).astype(np.float32)


# ============================================================
# XGBOOST SETTINGS FROM ORIGINAL PROJECT
# ============================================================

spw = (
    (ytr == 0).sum()
    /
    max((ytr == 1).sum(), 1)
)

print("\nscale_pos_weight:", spw)

print("\nTraining ONLY XGBoost...")
print("RF / LSTM / AE / GNN will NOT be trained.")


model = xgb.XGBClassifier(
    random_state=exp.SEED,
    scale_pos_weight=spw,
    **exp.HP["xgb"]
)

model.fit(
    Xs[tr],
    ytr
)

print("XGBoost training complete.")


# ============================================================
# SHAP SAMPLE
# ============================================================

rng = np.random.default_rng(exp.SEED)

sub = rng.choice(
    len(te),
    min(2000, len(te)),
    replace=False
)

X_shap = Xs[te][sub]


# ============================================================
# FEATURE NAMES
# ============================================================

names = exp.feat_names()

print("\nNumber of feature names:", len(names))

if len(names) != 165:
    raise ValueError(
        f"Expected 165 feature names, found {len(names)}"
    )


# ============================================================
# TREE SHAP
# ============================================================

print("\nCalculating TreeSHAP...")

explainer = shap.TreeExplainer(model)

sv = explainer.shap_values(X_shap)

if isinstance(sv, list):
    sv = sv[-1]


# ============================================================
# FIGURE 4A
# ============================================================

print("\nGenerating Figure 4A...")

plt.figure(
    figsize=(3.5, 3.2),
    dpi=300
)

shap.summary_plot(
    sv,
    X_shap,
    feature_names=names,
    max_display=12,
    show=False,
    plot_size=None
)

plt.gcf().axes[0].tick_params(
    labelsize=6
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE CSV
# ============================================================

mean_abs_shap = np.mean(
    np.abs(sv),
    axis=0
)

importance = pd.DataFrame({
    "feature": names,
    "mean_absolute_shap": mean_abs_shap
})

importance = importance.sort_values(
    "mean_absolute_shap",
    ascending=False
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "\nCreated:"
)

print(
    os.path.abspath(OUTPUT_FILE)
)

print(
    os.path.abspath(IMPORTANCE_FILE)
)

print("\nTop 12 features:")
print(
    importance.head(12).to_string(index=False)
)