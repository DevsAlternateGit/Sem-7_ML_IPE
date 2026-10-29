from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── REVISION CELL ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import matplotlib; matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Quick revision snippet ---
np.random.seed(42)
X_rev = np.random.rand(50, 2) * 100

inertias = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_rev)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 3))
plt.plot(range(1, 8), inertias, 'bo-')
plt.title("Elbow Curve (revision)")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.tight_layout(); plt.show()

# Viva notes:
# - Inertia = within-cluster sum of squared errors (WCSS); lower is better
# - Elbow = optimal k where adding more k gives diminishing returns in inertia
# - Silhouette score measures cluster cohesion and separation (-1 to +1, higher=better)
# - KMeans assumes spherical clusters; sensitive to initialisation and outliers
# - n_init=10 runs KMeans 10 times and picks the best inertia result
"""))

# ── Q12 ───────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### Q12
Write a Python script that implements the Elbow Method to determine the optimal number of clusters for a generated dataset. (Generate dataset of your choice with 14 points)
"""))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
"""))

cells.append(nbf.v4.new_code_cell("""\
# Generate exactly 14 random 2D points
np.random.seed(42)
X = np.random.rand(14, 2) * 100
print("Dataset shape:", X.shape)
print("Points:\\n", X)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Compute inertia for k = 1 to 10
inertias = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

print("k  -> Inertia")
for k, inr in zip(k_values, inertias):
    print(f"k={k} -> {inr:.2f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, 'bo-', linewidth=2, markersize=8)
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.xticks(list(k_values))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
"""))

# ── Q13 ───────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### Q13
Use make_blobs to generate 300 samples. Analyse the data and apply KMeans clustering and generate the graph for better representation. Show what happens if you change n_clusters count from 2 to 7 (2, 3, 4, 5, 6, 7)?
"""))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
"""))

cells.append(nbf.v4.new_code_cell("""\
# Generate 300 samples with 4 natural centers
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
print("Dataset shape:", X.shape)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Plot raw data (no cluster labels yet)
plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], s=30, color='steelblue', alpha=0.7)
plt.title("Raw Data (make_blobs - 300 samples)")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Subplot grid 2x3 for k = 2, 3, 4, 5, 6, 7
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
k_values = [2, 3, 4, 5, 6, 7]

for ax, k in zip(axes.flatten(), k_values):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    # Scatter colored by cluster
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=25, alpha=0.7)
    # Mark centroids with X
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='X', s=150, zorder=5, label='Centroids')
    ax.set_title(f"k = {k}  (Inertia: {model.inertia_:.1f})")
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.legend(loc='upper right', fontsize=7)

plt.suptitle("KMeans Clustering: k = 2 to 7", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Conclusion:
# At k=4, clusters best match natural groupings (inertia drops sharply before this).
# Increasing k beyond optimal splits natural groups into arbitrary sub-clusters.
# Inertia always decreases with k, so use elbow/silhouette to pick optimal k.
"""))

# ── Q14 ───────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### Q14
Generate random dataset with 35 points. Apply KMeans clustering and generate the plot with each cluster points with different colours. What is the impact of varying cluster count? Justify your understanding with suitable graphs.
"""))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
"""))

cells.append(nbf.v4.new_code_cell("""\
# Generate exactly 35 random 2D points
np.random.seed(42)
X = np.random.rand(35, 2) * 100
print("Dataset shape:", X.shape)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Apply KMeans with k=3 (default first look)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)
centroids = model.cluster_centers_

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=80, alpha=0.85)
plt.scatter(centroids[:, 0], centroids[:, 1],
            c='black', marker='X', s=200, zorder=5, label='Centroids')
plt.title("KMeans Clustering - 35 Points (k=3)")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.show()
print(f"Inertia for k=3: {model.inertia_:.2f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Subplot grid for k = 2, 3, 4, 5 - impact of varying cluster count
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
k_values = [2, 3, 4, 5]

for ax, k in zip(axes.flatten(), k_values):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=70, alpha=0.8)
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='X', s=180, zorder=5, label='Centroids')
    ax.set_title(f"k = {k}  (Inertia: {model.inertia_:.1f})")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=8)

plt.suptitle("Impact of Varying Cluster Count (k=2 to 5)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Conclusion:
# At k=3, clusters best match natural groupings (inertia drops sharply before this).
# Increasing k beyond optimal splits natural groups into arbitrary sub-clusters.
# Inertia always decreases with k, so use elbow/silhouette to pick optimal k.
"""))

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.8.0"}}

ROOT_DIR = Path(__file__).resolve().parent.parent
path = str(ROOT_DIR / "06_kmeans_elbow.ipynb")
with open(path, "w") as f:
    nbf.write(nb, f)
print("Written:", path)
