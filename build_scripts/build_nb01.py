from pathlib import Path
"""Build 01_decision_trees.ipynb - Q1 to Q4"""
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()
cells = []

# ── REVISION / ONE-SHOT CELL ─────────────────────────────────────────────────
cells.append(new_code_cell("""\
# ============================================================
# REVISION / ONE-SHOT  -  Decision Trees  (Q1-Q4)
# ============================================================

# --- Standard imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# --- Load CSV dataset ---
df = pd.read_csv("datasets/diabetes.csv")
X = df.drop("Outcome", axis=1)   # features
y = df["Outcome"]                 # target

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fit Decision Tree ---
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Plot tree ---
plt.figure(figsize=(12, 5))
plot_tree(model, feature_names=list(X.columns), class_names=["0","1"], filled=True)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

# ============================================================
# VIVA NOTES:
# max_depth      - limits tree levels; prevents overfitting (Q1 uses depth=3)
# max_leaf_nodes - caps leaf count; another way to prune (Q3)
# min_samples_leaf - min samples each leaf must have; smooths tree (Q3)
# criterion      - "gini" (default) or "entropy" for info gain split
# compare models - fit both, call accuracy_score, print side by side (Q2, Q4)
# read the plot  - each box shows: feature split, gini, samples, class dist
# ============================================================
"""))

# ── Q1 ───────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
### Q1
Prepare decision tree classifier in python using sklearn library for data set diabetes.csv - (pregnant, glucose, bp, skin, insulin, bmi, pedigree, age, label). Use all features except label as independent variable. Use complete dataset for training and limit the depth of tree up to 3 levels and plot using tree.plot_tree() method.
"""))

cells.append(new_code_cell("""\
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset - columns map: label -> Outcome
df = pd.read_csv("datasets/diabetes.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
"""))

cells.append(new_code_cell("""\
# Use all features except Outcome (the label) as X
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train on complete dataset (no split, as question says)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)
print("Training accuracy (full data):", model.score(X, y))
"""))

cells.append(new_code_cell("""\
# Plot the tree (depth=3 keeps it readable)
plt.figure(figsize=(20, 8))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=["No Diabetes", "Diabetes"],
    filled=True,         # color nodes by majority class
    rounded=True
)
plt.title("Decision Tree - Diabetes Dataset (max_depth=3)")
plt.tight_layout()
plt.show()
"""))

# ── Q2 ───────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
### Q2
Implement a Decision Tree Classifier and compare its performance with a SVM Classifier on the iris dataset. Display the accuracy of both models.
"""))

cells.append(new_code_cell("""\
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris from sklearn
iris = load_iris()
X, y = iris.data, iris.target

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

cells.append(new_code_cell("""\
# Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

# SVM model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))

print(f"Decision Tree Accuracy : {dt_acc:.4f}")
print(f"SVM Accuracy           : {svm_acc:.4f}")
"""))

cells.append(new_code_cell("""\
# Bar chart comparison
models = ["Decision Tree", "SVM"]
accs   = [dt_acc, svm_acc]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, accs, color=["steelblue", "tomato"], edgecolor="black")
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{acc:.4f}", ha="center", va="bottom", fontsize=11)
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Decision Tree vs SVM - Iris Dataset")
plt.tight_layout()
plt.show()
"""))

# ── Q3 ───────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
### Q3
Build decision tree classifier for iris data set. One with maximum leaf nodes up to 8 and another one with minimum sample per leaf as 5. Compare accuracy of both models.
"""))

cells.append(new_code_cell("""\
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

cells.append(new_code_cell("""\
# Model 1: max_leaf_nodes=8 (caps number of leaves)
model_leaf = DecisionTreeClassifier(max_leaf_nodes=8, random_state=42)
model_leaf.fit(X_train, y_train)
acc_leaf = accuracy_score(y_test, model_leaf.predict(X_test))

# Model 2: min_samples_leaf=5 (each leaf must have >= 5 samples)
model_min = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
model_min.fit(X_train, y_train)
acc_min = accuracy_score(y_test, model_min.predict(X_test))

print(f"max_leaf_nodes=8  Accuracy : {acc_leaf:.4f}")
print(f"min_samples_leaf=5 Accuracy: {acc_min:.4f}")
"""))

cells.append(new_code_cell("""\
# Compare both trees side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

plot_tree(model_leaf, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, ax=axes[0])
axes[0].set_title(f"max_leaf_nodes=8  |  Acc={acc_leaf:.4f}")

plot_tree(model_min, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, ax=axes[1])
axes[1].set_title(f"min_samples_leaf=5  |  Acc={acc_min:.4f}")

plt.tight_layout()
plt.show()
"""))

# ── Q4 ───────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
### Q4
Implement a Decision Tree Classifier and compare its performance with a Logistic Classifier on the diabetes dataset. Develop Python code to display the accuracy of both models.

*Note: "Logistic Classifier" means LogisticRegression in sklearn.*
"""))

cells.append(new_code_cell("""\
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  # "Logistic Classifier" = LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("datasets/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

cells.append(new_code_cell("""\
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

# Logistic Regression (max_iter increased to ensure convergence)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

print(f"Decision Tree Accuracy   : {dt_acc:.4f}")
print(f"Logistic Reg. Accuracy   : {lr_acc:.4f}")
"""))

cells.append(new_code_cell("""\
# Bar chart
models = ["Decision Tree", "Logistic Regression"]
accs   = [dt_acc, lr_acc]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, accs, color=["steelblue", "mediumseagreen"], edgecolor="black")
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{acc:.4f}", ha="center", va="bottom", fontsize=11)
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Decision Tree vs Logistic Regression - Diabetes Dataset")
plt.tight_layout()
plt.show()
"""))

# ── Assemble and save ─────────────────────────────────────────────────────────
nb.cells = cells
ROOT_DIR = Path(__file__).resolve().parent.parent
out_path = str(ROOT_DIR / "01_decision_trees.ipynb")
with open(out_path, "w") as f:
    nbformat.write(nb, f)
print("Saved:", out_path)
