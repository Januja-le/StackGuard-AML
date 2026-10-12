import matplotlib.pyplot as plt
import numpy as np

# Actual confusion matrix
cm = np.array([
    [15562, 25],   # Actual Licit
    [300, 783]     # Actual Illicit
])

fig, ax = plt.subplots(figsize=(5, 4))

# Display matrix
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

# Axis labels
ax.set(
    xticks=np.arange(2),
    yticks=np.arange(2),
    xticklabels=["Licit", "Illicit"],
    yticklabels=["Licit", "Illicit"],
    xlabel="Predicted Label",
    ylabel="Actual Label",
    title="Confusion Matrix"
)

# Write values inside cells
for i in range(2):
    for j in range(2):
        ax.text(
            j, i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=13
        )

# Clean layout
plt.tight_layout()

# Save as high-resolution JPEG
plt.savefig(
    "confusion_matrix.jpg",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    format="jpg"
)

plt.show()