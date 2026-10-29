from build_03_helper import build_03
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
#!/usr/bin/env python3
"""Build all 4 AML practical notebooks."""
import nbformat as nbf

def make_code(src):
    return nbf.v4.new_code_cell(src)

def make_md(src):
    return nbf.v4.new_markdown_cell(src)

# ============================================================
# 02_cross_validation.ipynb
# ============================================================
def build_02():
    nb = nbf.v4.new_notebook()
    cells = []

    # REVISION
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"from sklearn.datasets import make_classification\n"
"from sklearn.model_selection import LeaveOneOut\n"
"from sklearn.tree import DecisionTreeClassifier\n"
"\n"
"# --- LOOCV One-Shot Revision ---\n"
"X, y = make_classification(n_samples=20, n_features=5, n_classes=2, random_state=42)\n"
"loo = LeaveOneOut()\n"
"clf = DecisionTreeClassifier(random_state=42)\n"
"accs = []\n"
"for i, (tr, te) in enumerate(loo.split(X)):\n"
"    clf.fit(X[tr], y[tr])\n"
"    accs.append(int(clf.predict(X[te])[0] == y[te][0]))\n"
"print(f'Mean LOOCV Accuracy: {np.mean(accs):.4f}')\n"
"\n"
"# VIVA NOTES:\n"
"# 1. LOOCV: n_folds = n_samples - every sample used once as test set\n"
"# 2. Most expensive CV - trains n separate models\n"
"# 3. Low bias (uses almost all data per fold), but high variance\n"
"# 4. Good for small datasets where data is precious\n"
"# 5. Mean accuracy = proportion of correct single-sample predictions\n"
"# 6. Preferred over k-fold when dataset is very small (n < 50)\n"
    ))

    # Q5 markdown
    cells.append(make_md(
"### Q5\n"
"Write a Python code to perform Leave-One-Out Cross-Validation (LOOCV) on a dataset "
"using a Decision Tree Classifier. The dataset consists of features and labels as specified below.\n\n"
"**Dataset Description:**\n"
"- The dataset consists of 100 samples with 5 features.\n"
"- The target variable is binary (0 or 1).\n\n"
"**Instructions:**\n"
"- Import the necessary libraries.\n"
"- Load or create a synthetic dataset with the specified characteristics.\n"
"- Implement Leave-One-Out Cross-Validation (LOOCV).\n"
"- Train a Decision Tree Classifier on each training set and evaluate it on the left-out sample.\n"
"- Print the accuracy for each iteration and the average accuracy across all iterations."
    ))

    # Q5 cell 1 - imports
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"from sklearn.datasets import make_classification\n"
"from sklearn.model_selection import LeaveOneOut\n"
"from sklearn.tree import DecisionTreeClassifier\n"
    ))

    # Q5 cell 2 - dataset
    cells.append(make_code(
"# Generate synthetic dataset: 100 samples, 5 features, binary target\n"
"X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)\n"
"print(f'Dataset shape: X={X.shape}, y={y.shape}')\n"
"print(f'Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}')\n"
    ))

    # Q5 cell 3 - LOOCV
    cells.append(make_code(
"# LOOCV - each sample left out once as the test set\n"
"loo = LeaveOneOut()\n"
"clf = DecisionTreeClassifier(random_state=42)\n"
"fold_accs = []\n"
"\n"
"for i, (train_idx, test_idx) in enumerate(loo.split(X)):\n"
"    X_train, X_test = X[train_idx], X[test_idx]\n"
"    y_train, y_test = y[train_idx], y[test_idx]\n"
"    clf.fit(X_train, y_train)\n"
"    y_pred = clf.predict(X_test)\n"
"    acc = int(y_pred[0] == y_test[0])  # 1=correct, 0=wrong\n"
"    fold_accs.append(acc)\n"
"    print(f'Fold {i+1:3d}: Predicted={y_pred[0]}, Actual={y_test[0]}, Accuracy={acc}')\n"
"\n"
"print(f'\\nAverage LOOCV Accuracy: {np.mean(fold_accs):.4f}')\n"
    ))

    # Q5 cell 4 - plot
    cells.append(make_code(
"# Bar plot of per-fold accuracy (100 bars)\n"
"plt.figure(figsize=(14, 4))\n"
"colors = ['green' if a else 'red' for a in fold_accs]\n"
"plt.bar(range(1, 101), fold_accs, color=colors, alpha=0.7)\n"
"plt.axhline(y=np.mean(fold_accs), color='blue', linestyle='--',\n"
"            label=f'Mean Acc = {np.mean(fold_accs):.4f}')\n"
"plt.xlabel('Fold Number')\n"
"plt.ylabel('Accuracy (1=Correct, 0=Wrong)')\n"
"plt.title('LOOCV Per-Fold Accuracy - Decision Tree Classifier')\n"
"plt.legend()\n"
"plt.tight_layout()\n"
"plt.show()\n"
    ))

    nb.cells = cells
    with open(ROOT_DIR / '02_cross_validation.ipynb', 'w') as f:
        nbf.write(nb, f)
    print('02_cross_validation.ipynb created')


# ============================================================
# 04_svm.ipynb
# ============================================================
def build_04():
    nb = nbf.v4.new_notebook()
    cells = []

    # REVISION
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"from sklearn.svm import SVC\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.datasets import load_breast_cancer\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.metrics import accuracy_score\n"
"\n"
"# --- SVM One-Shot Revision ---\n"
"data = load_breast_cancer()\n"
"X, y = data.data, data.target\n"
"sc = StandardScaler()\n"
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
"X_train = sc.fit_transform(X_train)\n"
"X_test = sc.transform(X_test)\n"
"for kernel in ['linear', 'rbf']:\n"
"    m = SVC(kernel=kernel, random_state=42)\n"
"    m.fit(X_train, y_train)\n"
"    print(f'{kernel}: {accuracy_score(y_test, m.predict(X_test)):.4f}')\n"
"\n"
"# VIVA NOTES:\n"
"# 1. Kernel trick maps data to high-dim space without explicit computation (RBF, poly, etc.)\n"
"# 2. C = regularization - high C means low bias, high variance (fits training more)\n"
"# 3. gamma (RBF) = inverse radius - high gamma means tight boundary (overfitting)\n"
"# 4. Always scale features before SVM (sensitive to feature magnitude)\n"
"# 5. SVMs maximize margin between classes using support vectors only\n"
"# 6. Multiclass SVM uses OvO (one-vs-one) by default in sklearn\n"
    ))

    # Q7 markdown
    cells.append(make_md(
"### Q7\n"
"Consider Social_Network_Ads.csv dataset - (UserID, Gender, Age, EstimatedSalary, Purchased). "
"Use Age and EstimatedSalary as input features and Purchased as target feature. "
"Split test data set 30% of complete dataset. Build two models of support vector classifier in python "
"using sklearn library, one for linear and another for RBF kernel with C and gamma parameters set. "
"Predict test labels and print test accuracy."
    ))

    # Q7 cell 1 - imports
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"from sklearn.svm import SVC\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.metrics import accuracy_score\n"
    ))

    # Q7 cell 2 - load data
    cells.append(make_code(
"# Load dataset - use Age and EstimatedSalary as features, Purchased as target\n"
"df = pd.read_csv('datasets/Social_Network_Ads.csv')\n"
"print(df.head())\n"
"print(f'Shape: {df.shape}')\n"
"X = df[['Age', 'EstimatedSalary']].values\n"
"y = df['Purchased'].values\n"
    ))

    # Q7 cell 3 - split and scale
    cells.append(make_code(
"# 70/30 split, then scale features (SVM is sensitive to scale)\n"
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n"
"sc = StandardScaler()\n"
"X_train = sc.fit_transform(X_train)\n"
"X_test = sc.transform(X_test)\n"
"print(f'Train: {X_train.shape}, Test: {X_test.shape}')\n"
    ))

    # Q7 cell 4 - train both SVMs
    cells.append(make_code(
"# Linear SVC\n"
"model_lin = SVC(kernel='linear', C=1, random_state=42)\n"
"model_lin.fit(X_train, y_train)\n"
"y_pred_lin = model_lin.predict(X_test)\n"
"print(f'Linear SVC Accuracy: {accuracy_score(y_test, y_pred_lin):.4f}')\n"
"\n"
"# RBF SVC\n"
"model_rbf = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)\n"
"model_rbf.fit(X_train, y_train)\n"
"y_pred_rbf = model_rbf.predict(X_test)\n"
"print(f'RBF SVC Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}')\n"
    ))

    # Q7 cell 5 - decision boundary plot
    cells.append(make_code(
"# Decision boundary plot for both kernels\n"
"def plot_boundary(ax, model, X_train, y_train, title):\n"
"    h = 0.02\n"
"    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1\n"
"    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1\n"
"    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),\n"
"                         np.arange(y_min, y_max, h))\n"
"    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n"
"    Z = Z.reshape(xx.shape)\n"
"    ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')\n"
"    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=30)\n"
"    ax.set_title(title)\n"
"    ax.set_xlabel('Age (scaled)')\n"
"    ax.set_ylabel('EstimatedSalary (scaled)')\n"
"\n"
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
"plot_boundary(axes[0], model_lin, X_train, y_train, 'SVC - Linear Kernel')\n"
"plot_boundary(axes[1], model_rbf, X_train, y_train, 'SVC - RBF Kernel (C=1, gamma=0.1)')\n"
"plt.tight_layout()\n"
"plt.show()\n"
    ))

    # Q8 markdown
    cells.append(make_md(
"### Q8\n"
"Develop a Support Vector Classifier to predict whether a tumor is malignant or benign based on "
"30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. "
"The dataset contains 569 samples with binary labels indicating tumor type (breast_cancer dataset of sklearn). "
"Load the dataset from sklearn. Split the dataset into training and testing sets. "
"Train the SVC model using the training data. Visualize the results."
    ))

    # Q8 cell 1 - imports
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from sklearn.datasets import load_breast_cancer\n"
"from sklearn.svm import SVC\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
"from sklearn.decomposition import PCA\n"
    ))

    # Q8 cell 2 - load
    cells.append(make_code(
"# Load breast cancer dataset (569 samples, 30 features, binary: malignant/benign)\n"
"data = load_breast_cancer()\n"
"X, y = data.data, data.target\n"
"print(f'Shape: {X.shape}, Classes: {data.target_names}')\n"
    ))

    # Q8 cell 3 - split, scale, train
    cells.append(make_code(
"# 80/20 split, scale features before SVM\n"
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
"sc = StandardScaler()\n"
"X_train = sc.fit_transform(X_train)\n"
"X_test = sc.transform(X_test)\n"
"\n"
"model = SVC(kernel='rbf', random_state=42)\n"
"model.fit(X_train, y_train)\n"
"y_pred = model.predict(X_test)\n"
"print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n"
"print(classification_report(y_test, y_pred, target_names=data.target_names))\n"
    ))

    # Q8 cell 4 - confusion matrix
    cells.append(make_code(
"# (a) Confusion matrix heatmap\n"
"cm = confusion_matrix(y_test, y_pred)\n"
"plt.figure(figsize=(6, 5))\n"
"sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
"            xticklabels=data.target_names, yticklabels=data.target_names)\n"
"plt.xlabel('Predicted')\n"
"plt.ylabel('True')\n"
"plt.title('Confusion Matrix - SVC RBF (Breast Cancer)')\n"
"plt.tight_layout()\n"
"plt.show()\n"
    ))

    # Q8 cell 5 - PCA scatter
    cells.append(make_code(
"# (b) 2D PCA projection - predicted vs true labels\n"
"pca = PCA(n_components=2, random_state=42)\n"
"X_test_2d = pca.fit_transform(X_test)\n"
"\n"
"fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
"for ax, labels, title in zip(axes, [y_test, y_pred], ['True Labels', 'Predicted Labels']):\n"
"    sc_plot = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=labels,\n"
"                         cmap='coolwarm', edgecolors='k', s=50, alpha=0.8)\n"
"    ax.set_title(f'PCA Projection - {title}')\n"
"    ax.set_xlabel('PC1')\n"
"    ax.set_ylabel('PC2')\n"
"    plt.colorbar(sc_plot, ax=ax, ticks=[0, 1])\n"
"plt.suptitle('SVC RBF - Breast Cancer (0=malignant, 1=benign)')\n"
"plt.tight_layout()\n"
"plt.show()\n"
    ))

    # Q9 markdown
    cells.append(make_md(
"### Q9\n"
"Implement an SVC model to classify iris flowers into three species "
"(Setosa, Versicolor, and Virginica) based on their sepal and petal dimensions. "
"The dataset contains 150 samples with four features: sepal length, sepal width, petal length, and petal width. "
"Load the dataset from sklearn. Split the dataset into training and testing sets. "
"Train the SVC model using the training data. Visualize the results."
    ))

    # Q9 cell 1 - imports
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from sklearn.datasets import load_iris\n"
"from sklearn.svm import SVC\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.metrics import accuracy_score, confusion_matrix\n"
    ))

    # Q9 cell 2 - load
    cells.append(make_code(
"# Load Iris dataset (150 samples, 4 features, 3 classes)\n"
"data = load_iris()\n"
"X, y = data.data, data.target\n"
"print(f'Shape: {X.shape}, Classes: {data.target_names}')\n"
    ))

    # Q9 cell 3 - split, scale, train
    cells.append(make_code(
"# 80/20 split, scale features\n"
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
"sc = StandardScaler()\n"
"X_train = sc.fit_transform(X_train)\n"
"X_test = sc.transform(X_test)\n"
"\n"
"model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)\n"
"model.fit(X_train, y_train)\n"
"y_pred = model.predict(X_test)\n"
"print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n"
    ))

    # Q9 cell 4 - confusion matrix
    cells.append(make_code(
"# (a) Confusion matrix heatmap\n"
"cm = confusion_matrix(y_test, y_pred)\n"
"plt.figure(figsize=(6, 5))\n"
"sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',\n"
"            xticklabels=data.target_names, yticklabels=data.target_names)\n"
"plt.xlabel('Predicted')\n"
"plt.ylabel('True')\n"
"plt.title('Confusion Matrix - SVC RBF (Iris)')\n"
"plt.tight_layout()\n"
"plt.show()\n"
    ))

    # Q9 cell 5 - decision boundary (first 2 features, unscaled)
    cells.append(make_code(
"# (b) Decision boundary using first 2 features (sepal length, sepal width) - unscaled range\n"
"X2 = data.data[:, :2]  # sepal length and sepal width only\n"
"y2 = data.target\n"
"\n"
"X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)\n"
"sc2 = StandardScaler()\n"
"X2_train_s = sc2.fit_transform(X2_train)\n"
"\n"
"m2 = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)\n"
"m2.fit(X2_train_s, y2_train)\n"
"\n"
"# Meshgrid over unscaled iris sepal range\n"
"x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5\n"
"y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5\n"
"xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),\n"
"                     np.arange(y_min, y_max, 0.02))\n"
"grid_scaled = sc2.transform(np.c_[xx.ravel(), yy.ravel()])\n"
"Z = m2.predict(grid_scaled).reshape(xx.shape)\n"
"\n"
"plt.figure(figsize=(8, 6))\n"
"plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlGn')\n"
"scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap='RdYlGn', edgecolors='k', s=40)\n"
"plt.xlabel('Sepal Length (cm)')\n"
"plt.ylabel('Sepal Width (cm)')\n"
"plt.title('SVC Decision Boundary - Iris (Sepal Features)')\n"
"plt.colorbar(scatter, ticks=[0, 1, 2])\n"
"plt.tight_layout()\n"
"plt.show()\n"
    ))

    nb.cells = cells
    with open(ROOT_DIR / '04_svm.ipynb', 'w') as f:
        nbf.write(nb, f)
    print('04_svm.ipynb created')


# ============================================================
# 05_regression.ipynb
# ============================================================
def build_05():
    nb = nbf.v4.new_notebook()
    cells = []

    # REVISION
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"from sklearn.linear_model import LogisticRegression, LinearRegression\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.metrics import accuracy_score, r2_score, mean_squared_error\n"
"\n"
"# --- Regression One-Shot Revision ---\n"
"from sklearn.datasets import load_iris\n"
"data = load_iris()\n"
"X, y = data.data[:, :2], (data.target > 0).astype(int)\n"
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
"lr = LogisticRegression(random_state=42)\n"
"lr.fit(X_train, y_train)\n"
"print(f'LogReg Accuracy: {accuracy_score(y_test, lr.predict(X_test)):.4f}')\n"
"\n"
"# VIVA NOTES:\n"
"# 1. Logistic Regression: uses sigmoid function - output is probability (0 to 1)\n"
"# 2. Decision boundary: P(y=1|x) >= 0.5 -> class 1\n"
"# 3. Linear Regression: minimizes MSE (Mean Squared Error) using OLS\n"
"# 4. R2 score: 1=perfect fit, 0=baseline mean predictor, negative=worse than mean\n"
"# 5. Residual plot: checks homoscedasticity (constant variance) - should be random scatter\n"
"# 6. Logistic uses log-loss, Linear uses MSE as cost functions\n"
    ))

    # Q10 markdown
    cells.append(make_md(
"### Q10\n"
"Imagine a telecommunications company that wants to predict whether a customer will churn "
"(leave the service) based on various features such as age, account length, and monthly charges. "
"The company has historical data on customers, including whether they churned or not.\n\n"
"Age: (in years), Account_Length: (in months), Monthly_Charges: (in dollars)\n"
"Churn: Target variable (1 if the customer churned, 0 otherwise)\n\n"
"Build a logistic regression model using python that can predict the probability of a customer "
"churning by splitting 80% of given data for training. Predict targets on trained model of test data. Print accuracy."
    ))

    # Q10 cell 1 - imports
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.metrics import accuracy_score\n"
    ))

    # Q10 cell 2 - create data
    cells.append(make_code(
"# Inline dataset as provided in the question\n"
"data = {\n"
"    'Age':             [25, 34, 45, 29, 50, 38, 42, 35, 48, 55],\n"
"    'Account_Length':  [12, 24, 36, 18, 48, 30, 42, 24, 36, 60],\n"
"    'Monthly_Charges': [70, 90, 80, 100, 60, 80, 90, 70, 80, 100],\n"
"    'Churn':           [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]\n"
"}\n"
"df = pd.DataFrame(data)\n"
"print(df)\n"
"\n"
"X = df[['Age', 'Account_Length', 'Monthly_Charges']]\n"
"y = df['Churn']\n"
    ))

    # Q10 cell 3 - train and predict
    cells.append(make_code(
"# 80/20 split - only 10 rows, so test set is tiny (2 samples), accuracy is coarse\n"
"# Only 10 rows: test set is tiny (2 samples), so accuracy is coarse\n"
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
"\n"
"model = LogisticRegression(max_iter=1000, random_state=42)\n"
"model.fit(X_train, y_train)\n"
"y_pred = model.predict(X_test)\n"
"\n"
"print(f'Test samples: {len(y_test)}')\n"
"print(f'Predicted: {list(y_pred)}')\n"
"print(f'Actual:    {list(y_test.values)}')\n"
"print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')\n"
    ))

    # Q11 markdown
    cells.append(make_md(
"### Q11\n"
"Consider Salary.csv with years_of_experience and salary. Write a python code for fitting best fit "
"simple linear regression with independent variable years_of_experience and dependent variable salary. "
"Use complete dataset to train model. Plot: fitted model along with trained data points, and residual plot."
    ))

    # Q11 cell 1 - imports
    cells.append(make_code(
"import matplotlib; matplotlib.use('Agg')\n"
"import pandas as pd\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"from sklearn.linear_model import LinearRegression\n"
"from sklearn.metrics import r2_score, mean_squared_error\n"
    ))

    # Q11 cell 2 - load data
    cells.append(make_code(
"# Load only relevant columns and drop missing values (12 NaN rows)\n"
"df = pd.read_csv('datasets/Salary.csv')[['Years of Experience', 'Salary']].dropna()\n"
"print(f'Rows after dropna: {len(df)}')\n"
"print(df.head())\n"
"\n"
"X = df[['Years of Experience']].values  # 2D for sklearn\n"
"y = df['Salary'].values\n"
    ))

    # Q11 cell 3 - fit model
    cells.append(make_code(
"# Use COMPLETE dataset for training (no split) as question states\n"
"model = LinearRegression()\n"
"model.fit(X, y)\n"
"y_pred = model.predict(X)\n"
"\n"
"print(f'Coefficient (slope): {model.coef_[0]:.4f}')\n"
"print(f'Intercept:           {model.intercept_:.4f}')\n"
"print(f'R2 Score:            {r2_score(y, y_pred):.4f}')\n"
"print(f'MSE:                 {mean_squared_error(y, y_pred):.4f}')\n"
    ))

    # Q11 cell 4 - plots
    cells.append(make_code(
"# Plot 1: Scatter of actual data + fitted regression line\n"
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
"\n"
"axes[0].scatter(X, y, color='steelblue', alpha=0.6, label='Actual Data')\n"
"axes[0].plot(X, y_pred, color='red', linewidth=2, label='Regression Line')\n"
"axes[0].set_xlabel('Years of Experience')\n"
"axes[0].set_ylabel('Salary')\n"
"axes[0].set_title('Simple Linear Regression - Salary vs Years of Experience')\n"
"axes[0].legend()\n"
"\n"
"# Plot 2: Residual plot - checks homoscedasticity (should be random scatter)\n"
"residuals = y - y_pred\n"
"axes[1].scatter(y_pred, residuals, color='darkorange', alpha=0.6)\n"
"axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5)\n"
"axes[1].set_xlabel('Predicted Salary')\n"
"axes[1].set_ylabel('Residuals (Actual - Predicted)')\n"
"axes[1].set_title('Residual Plot')\n"
"\n"
"plt.tight_layout()\n"
"plt.show()\n"
    ))

    nb.cells = cells
    with open(ROOT_DIR / '05_regression.ipynb', 'w') as f:
        nbf.write(nb, f)
    print('05_regression.ipynb created')


if __name__ == '__main__':
    build_02()
    build_03(ROOT_DIR)
    build_04()
    build_05()
    print('All notebooks created.')
