# Helper function to generate 03_naive_bayes.ipynb
import nbformat as nbf
from pathlib import Path

def build_03(root_dir):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # REVISION
    cells.append(nbf.v4.new_code_cell(
"""import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Naive Bayes One-Shot Revision ---
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# VIVA NOTES:
# 1. GaussianNB assumes each feature follows a Gaussian (normal) distribution
# 2. P(x|y) modeled as Gaussian per feature - mean and variance estimated from training data
# 3. Uses Bayes theorem: P(y|x) = P(x|y)*P(y) / P(x) - selects class with max posterior
# 4. Very fast to train (just compute mean/variance per class per feature)
# 5. Good baseline for high-dimensional data; works well when independence assumption holds
# 6. MultinomialNB for count data; BernoulliNB for binary features"""
    ))
    
    # Q6 markdown
    cells.append(nbf.v4.new_markdown_cell(
"""### Q6
The breast cancer dataset is available in the sklearn.datasets module and can be loaded using load_breast_cancer(). Implement a Naive Bayes Classifier on the Breast Cancer Dataset using python's sklearn library. Assume that all features of the dataset are continuous variables and must be used for building the model. Perform the following tasks: load the data set, split data for training and testing, build a model and predict labels of test data."""
    ))
    
    # Q6 imports
    cells.append(nbf.v4.new_code_cell(
"""import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"""
    ))
    
    # Q6 load
    cells.append(nbf.v4.new_code_cell(
"""# Load breast cancer dataset (569 samples, 30 continuous features, binary labels)
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Dataset shape: {X.shape}")
print(f"Classes: {data.target_names}")
print(f"Class distribution: malignant={sum(y==0)}, benign={sum(y==1)}")"""
    ))
    
    # Q6 split and model
    cells.append(nbf.v4.new_code_cell(
"""# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GaussianNB - suited for continuous features (assumes Gaussian distribution)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")"""
    ))
    
    # Q6 evaluate
    cells.append(nbf.v4.new_code_cell(
"""# Print accuracy and detailed classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))"""
    ))
    
    # Q6 plot
    cells.append(nbf.v4.new_code_cell(
"""# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Gaussian Naive Bayes (Breast Cancer)")
plt.tight_layout()
plt.show()"""
    ))
    
    nb.cells = cells
    out = root_dir / "03_naive_bayes.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    print("03_naive_bayes.ipynb created")
