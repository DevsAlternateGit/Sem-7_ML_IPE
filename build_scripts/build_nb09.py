from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── REVISION CELL ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import matplotlib; matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- Quick revision snippet ---
iris = load_iris()
X_rev = iris.data; y_rev = iris.target
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_rev, y_rev)   # supervised - needs y

plt.figure(figsize=(5, 3))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_rev, cmap='viridis', s=20)
plt.title("LDA revision - Iris (2D)")
plt.tight_layout(); plt.show()

# Viva notes:
# - LDA maximizes between-class variance / within-class variance ratio
# - Max components = min(n_classes - 1, n_features)
# - LDA is SUPERVISED - requires class labels during fit (unlike PCA)
# - Good for classification preprocessing; separates classes better than PCA
# - Assumes Gaussian class distributions with equal covariance matrices
"""))

# ── Q22 ───────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### Q22
Use load_wine from sklearn.datasets to load the wine dataset. Initialize the LDA object with n_components=2 to reduce the dataset to 2 dimensions. Use matplotlib to plot the LDA-reduced dataset in 2D, with each class represented by a different color.
"""))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
class_names = wine.target_names
print("Dataset shape:", X.shape)
print("Classes:", class_names)
# Wine has 3 classes, 13 features - max LDA components = min(3-1, 13) = 2
"""))

cells.append(nbf.v4.new_code_cell("""\
# Apply LDA - supervised (needs y)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
print("Shape after LDA:", X_lda.shape)
"""))

cells.append(nbf.v4.new_code_cell("""\
# 2D scatter plot with 3 class colors
colors = ['red', 'green', 'blue']
plt.figure(figsize=(8, 6))

for i, (cls, col) in enumerate(zip(class_names, colors)):
    mask = y == i
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1],
                color=col, label=f"Class {i}: {cls}", s=60, alpha=0.85)

plt.title("LDA - Wine Dataset (2D projection)")
plt.xlabel("LD1"); plt.ylabel("LD2")
plt.legend()
plt.tight_layout()
plt.show()
"""))

# ── Q23 ───────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### Q23
Use load_breast_cancer from sklearn.datasets to load the cancer dataset. Initialize the LDA object with one component to reduce the dataset to 1D. Use matplotlib to plot the LDA-reduced dataset.
"""))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load Breast Cancer dataset
bc = load_breast_cancer()
X = bc.data
y = bc.target
class_names = bc.target_names
print("Dataset shape:", X.shape)
print("Classes:", class_names)
# Breast cancer has 2 classes, 30 features
# max LDA components = min(2-1, 30) = 1 - so only 1 component is possible
"""))

cells.append(nbf.v4.new_code_cell("""\
# Apply LDA with n_components=1
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)
print("Shape after LDA:", X_lda.shape)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Plot 1D projection as overlapping histograms for each class
colors = ['royalblue', 'tomato']
plt.figure(figsize=(9, 5))

for i, (cls, col) in enumerate(zip(class_names, colors)):
    mask = y == i
    plt.hist(X_lda[mask, 0], bins=30, color=col,
             alpha=0.55, label=cls, edgecolor='black', linewidth=0.4)

plt.title("LDA - Breast Cancer Dataset (1D projection)")
plt.xlabel("LD1")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
"""))

# ── Q24 ───────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### Q24
Use load_digits() from sklearn.datasets to load the handwritten numbers dataset. Initialize the LDA object with two components to reduce the dataset. Use matplotlib to plot the LDA-reduced dataset.
"""))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load Digits dataset
digits = load_digits()
X = digits.data
y = digits.target
print("Dataset shape:", X.shape)
print("Number of classes:", len(set(y)))
# Digits has 10 classes, 64 features
# max LDA components = min(10-1, 64) = 9, so n_components=2 is valid
"""))

cells.append(nbf.v4.new_code_cell("""\
# Apply LDA with n_components=2
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
print("Shape after LDA:", X_lda.shape)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Scatter plot with 10 colors (one per digit)
plt.figure(figsize=(10, 7))
cmap = plt.cm.tab10
colors = [cmap(i / 10) for i in range(10)]

for digit in range(10):
    mask = y == digit
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1],
                color=colors[digit], label=str(digit), s=20, alpha=0.75)

plt.title("LDA - Digits Dataset (2D projection)")
plt.xlabel("LD1"); plt.ylabel("LD2")
plt.legend(title="Digit", loc='upper right', ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
"""))

# ── Q25 ───────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### Q25
Use load_iris from sklearn.datasets to load the Iris dataset. Initialize the LDA object with n_components=2 to reduce the dataset to 2 dimensions. Use matplotlib to plot the LDA-reduced dataset in 2D.
"""))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
print("Dataset shape:", X.shape)
print("Classes:", class_names)
# Iris has 3 classes, 4 features - max LDA components = min(3-1, 4) = 2
"""))

cells.append(nbf.v4.new_code_cell("""\
# Apply LDA with n_components=2
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
print("Shape after LDA:", X_lda.shape)
"""))

cells.append(nbf.v4.new_code_cell("""\
# 2D scatter plot with species colors
colors = ['red', 'green', 'blue']
plt.figure(figsize=(8, 6))

for i, (cls, col) in enumerate(zip(class_names, colors)):
    mask = y == i
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1],
                color=col, label=cls.capitalize(), s=60, alpha=0.85)

plt.title("LDA - Iris Dataset (2D projection)")
plt.xlabel("LD1"); plt.ylabel("LD2")
plt.legend(title="Species")
plt.tight_layout()
plt.show()
"""))

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.8.0"}}

ROOT_DIR = Path(__file__).resolve().parent.parent
path = str(ROOT_DIR / "09_lda.ipynb")
with open(path, "w") as f:
    nbf.write(nb, f)
print("Written:", path)
