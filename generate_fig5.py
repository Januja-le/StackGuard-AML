import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score

# ============================================================
# FIGURE 5 — PER-STEP RECALL AND F1
# ============================================================

BASE_DIR = r"C:\Users\Januj\Downloads\today"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_FILE = os.path.join(RESULTS_DIR, "test_predictions.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "fig5_per_step.png")

# ------------------------------------------------------------
# Load existing predictions
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Loaded:", INPUT_FILE)
print("Rows:", len(df))

# Actual labels
y_true = df["label"].astype(int)

# Stacked ensemble scores
y_score = df["Stacked_ensemble"].astype(float)

# Original ensemble threshold from results.json
THRESHOLD = 0.9469534176588061

# Convert scores to predictions
y_pred = (y_score >= THRESHOLD).astype(int)

# ------------------------------------------------------------
# Calculate Recall and F1 for each time step
# ------------------------------------------------------------

steps = sorted(df["time_step"].unique())

results = []

for step in steps:

    mask = df["time_step"] == step

    true_step = y_true[mask]
    pred_step = y_pred[mask]

    recall = recall_score(
        true_step,
        pred_step,
        zero_division=0
    )

    f1 = f1_score(
        true_step,
        pred_step,
        zero_division=0
    )

    results.append({
        "time_step": step,
        "recall": recall,
        "f1": f1
    })

metrics = pd.DataFrame(results)

# ------------------------------------------------------------
# Print values
# ------------------------------------------------------------

print("\nPer-step results:")
print(metrics.to_string(index=False))

# ------------------------------------------------------------
# Generate academic figure
# ------------------------------------------------------------

plt.figure(figsize=(7, 4.5), dpi=300)

plt.plot(
    metrics["time_step"],
    metrics["recall"],
    marker="o",
    linewidth=1.8,
    markersize=4,
    label="Recall"
)

plt.plot(
    metrics["time_step"],
    metrics["f1"],
    marker="s",
    linewidth=1.8,
    markersize=4,
    label="F1-score"
)

# Mark the point where the drop begins
plt.axvline(
    x=43,
    linestyle="--",
    linewidth=1.0
)

plt.text(
    43.2,
    0.82,
    "Drop begins",
    fontsize=8,
    verticalalignment="center"
)

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

plt.xlabel("Time Step", fontsize=10)
plt.ylabel("Score", fontsize=10)

plt.xticks(
    range(35, 50),
    fontsize=8
)

plt.yticks(
    np.arange(0, 1.01, 0.1),
    fontsize=8
)

plt.ylim(0, 1.05)

plt.grid(
    True,
    linestyle=":",
    linewidth=0.6,
    alpha=0.6
)

plt.legend(
    fontsize=8,
    frameon=True
)

plt.tight_layout()

# ------------------------------------------------------------
# Save PNG
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

# Also save a JPG if needed
plt.savefig(
    os.path.join(RESULTS_DIR, "fig5_per_step.jpg"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n" + "=" * 65)
print("SUCCESS")
print("=" * 65)

print("Created:")
print(OUTPUT_FILE)
print(os.path.join(RESULTS_DIR, "fig5_per_step.jpg"))