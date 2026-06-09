"""
Task 1: Iris Flower Classification - CodeAlpha Data Science Internship
Author: CodeAlpha Intern
Description: ML model to classify Iris flower species using Scikit-learn
Dataset: Iris.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 1. LOAD DATASET FROM CSV
# ─────────────────────────────────────────────────────────
df = pd.read_csv('Iris.csv')
df = df.drop('Id', axis=1)
df.columns = ['sepal length (cm)', 'sepal width (cm)',
              'petal length (cm)', 'petal width (cm)', 'species']
df['species'] = df['species'].str.replace('Iris-', '')

print("=" * 60)
print("       IRIS FLOWER CLASSIFICATION — CodeAlpha Task 1")
print("=" * 60)
print(f"\nDataset Shape : {df.shape}")
print(f"Features      : {list(df.columns[:-1])}")
print(f"Classes       : {list(df['species'].unique())}")
print(f"\nClass Distribution:\n{df['species'].value_counts().to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nStatistical Summary:\n{df.describe().round(2).to_string()}")
print(f"\nMissing Values:\n{df.isnull().sum().to_string()}")

# ─────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Dataset — Exploratory Data Analysis", fontsize=16, fontweight='bold')

palette = {"setosa": "#4CAF50", "versicolor": "#2196F3", "virginica": "#F44336"}

ax = axes[0, 0]
for sp, color in palette.items():
    subset = df[df['species'] == sp]
    ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
               label=sp, color=color, alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
ax.set_xlabel("Sepal Length (cm)", fontsize=11)
ax.set_ylabel("Sepal Width (cm)", fontsize=11)
ax.set_title("Sepal: Length vs Width", fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for sp, color in palette.items():
    subset = df[df['species'] == sp]
    ax.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
               label=sp, color=color, alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)", fontsize=11)
ax.set_title("Petal: Length vs Width", fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
species_order = ['setosa', 'versicolor', 'virginica']
colors_list = [palette[s] for s in species_order]
bp = ax.boxplot(
    [df[df['species'] == s]['petal length (cm)'].values for s in species_order],
    labels=species_order, patch_artist=True
)
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Petal Length Distribution by Species", fontsize=13, fontweight='bold')
ax.set_ylabel("Petal Length (cm)", fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
corr = df.drop('species', axis=1).corr()
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)
labels = ['Sepal\nLen', 'Sepal\nWid', 'Petal\nLen', 'Petal\nWid']
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=9)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                fontsize=10, color='black' if abs(corr.iloc[i, j]) < 0.7 else 'white')

plt.tight_layout()
plt.savefig("1_eda_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[✔] EDA plot saved.")

# ─────────────────────────────────────────────────────────
# 3. PREPROCESSING & TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────
X = df.drop('species', axis=1).values
le = LabelEncoder()
y = le.fit_transform(df['species'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────
# 4. TRAIN MULTIPLE MODELS & COMPARE
# ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression":    LogisticRegression(max_iter=200),
    "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":          DecisionTreeClassifier(random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', C=1.0, random_state=42),
}

results = {}
print("\n" + "=" * 60)
print("           MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<28} {'Test Acc':>9}  {'CV Mean':>8}  {'CV Std':>7}")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
    results[name] = {
        "model": model, "accuracy": acc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "y_pred": y_pred
    }
    print(f"{name:<28} {acc:>9.4f}  {cv_scores.mean():>8.4f}  {cv_scores.std():>7.4f}")

# ─────────────────────────────────────────────────────────
# 5. ACCURACY COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names    = list(results.keys())
accs     = [results[n]["accuracy"] for n in names]
cv_means = [results[n]["cv_mean"]  for n in names]
cv_stds  = [results[n]["cv_std"]   for n in names]

x = np.arange(len(names))
w = 0.35
bars1 = ax.bar(x - w/2, accs,     w, label='Test Accuracy',    color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + w/2, cv_means, w, label='CV Mean (5-fold)', color='#4CAF50', alpha=0.85,
               yerr=cv_stds, capsize=4, error_kw=dict(ecolor='black', lw=1.5))

for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
ax.set_ylim(0.8, 1.05)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Model Comparison — Test Accuracy vs Cross-Validation", fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("2_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[✔] Model comparison chart saved.")

# ─────────────────────────────────────────────────────────
# 6. BEST MODEL — DETAILED EVALUATION
# ─────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["accuracy"])
best      = results[best_name]
y_pred    = best["y_pred"]
class_names = le.classes_

print(f"\n{'=' * 60}")
print(f"  BEST MODEL: {best_name}  (Accuracy = {best['accuracy']:.4f})")
print(f"{'=' * 60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Best Model: {best_name}", fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix", fontsize=13, fontweight='bold')

per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1].bar(class_names, per_class_acc,
            color=[palette[n] for n in class_names], alpha=0.85)
axes[1].set_title("Per-Class Accuracy", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Accuracy", fontsize=11)
axes[1].set_ylim(0, 1.1)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(per_class_acc):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("3_best_model_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("[✔] Best model evaluation chart saved.")

# ─────────────────────────────────────────────────────────
# 7. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("         SAMPLE PREDICTIONS (first 10 test samples)")
print("=" * 60)
print(f"{'#':<4} {'Actual':<14} {'Predicted':<14} {'Match'}")
print("-" * 44)
for i in range(10):
    actual    = class_names[y_test[i]]
    predicted = class_names[y_pred[i]]
    match     = "✔" if actual == predicted else "✘"
    print(f"{i+1:<4} {actual:<14} {predicted:<14} {match}")

print(f"\n✅ Final Test Accuracy : {best['accuracy'] * 100:.2f}%")
print(f"✅ Cross-Val Accuracy  : {best['cv_mean'] * 100:.2f}% ± {best['cv_std'] * 100:.2f}%")
print("\n[✔] All 3 chart images saved in the same folder!")
print("[✔] Task 1 Complete — CodeAlpha Data Science Internship")
