# ============================================================
#  Iris Flower Classifier
#  Author : [Your Name]
#  Dataset : UCI Iris Dataset (built into scikit-learn)
#  Goal    : Predict the species of an iris flower based on
#             its measurements using a Machine Learning model.
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── 2. LOAD DATA ─────────────────────────────────────────────
iris = load_iris()

# Convert to a DataFrame so it's easier to work with
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target                          # 0, 1, or 2
df["species_name"] = df["species"].map(             # human-readable
    {0: "setosa", 1: "versicolor", 2: "virginica"}
)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(df.head(10))
print(f"\nShape: {df.shape}  →  {df.shape[0]} rows, {df.shape[1]} columns")
print("\nClass distribution:")
print(df["species_name"].value_counts())
print("\nBasic statistics:")
print(df.describe().round(2))


# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Iris Dataset – Feature Distributions by Species", fontsize=15)

features = iris.feature_names
colors   = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}

for ax, feature in zip(axes.flatten(), features):
    for species, color in colors.items():
        subset = df[df["species_name"] == species]
        ax.hist(subset[feature], alpha=0.6, label=species, color=color, bins=15)
    ax.set_title(feature)
    ax.set_xlabel("cm")
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
plt.close()
print("\n✅  Saved: eda_distributions.png")

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.close()
print("✅  Saved: eda_correlation.png")


# ── 4. PREPARE DATA ──────────────────────────────────────────
X = df[iris.feature_names]   # features (inputs)
y = df["species"]            # labels  (output)

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features so all values are on a similar range
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")


# ── 5. TRAIN MODEL ───────────────────────────────────────────
# K-Nearest Neighbors: finds the k closest training examples
# and takes a majority vote. Simple but powerful for small datasets.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("\n✅  Model trained!")


# ── 6. EVALUATE MODEL ────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy : {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(
    classification_report(
        y_test, y_pred, target_names=iris.target_names
    )
)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("✅  Saved: confusion_matrix.png")


# ── 7. MAKE A PREDICTION ─────────────────────────────────────
print("\n" + "=" * 50)
print("LIVE PREDICTION EXAMPLE")
print("=" * 50)

# [sepal length, sepal width, petal length, petal width]  (in cm)
sample = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled    = scaler.transform(sample)
prediction       = model.predict(sample_scaled)[0]
predicted_species = iris.target_names[prediction]

print(f"Input  : sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"Predicted species → {predicted_species.upper()}")
print("\nDone! 🎉")
