# ------------------------------------------------------------
# Book example: Personality and Birth Order
#interpretability example from Chapter 5 of "Interpretable Machine Learning with Python"

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model, tree, discriminant_analysis, ensemble, neural_network

# -----------------------------
# Load data from CSV (replace filename)
# -----------------------------
DATA_PATH = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/ML/Ml_surgical/birthorder.csv"   # <-- change to your CSV name

birthorder_df = pd.read_csv(DATA_PATH)

# Target column expected by the book
if "birthorder" not in birthorder_df.columns:
    raise ValueError("Column 'birthorder' not found. Check your CSV column names.")

# -----------------------------
# Split
# -----------------------------
rand = 9
y = birthorder_df["birthorder"]
X = birthorder_df.drop(["birthorder"], axis=1).copy()

# If your CSV has text columns, uncomment next line:
# X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rand)

# -----------------------------
# Models
# -----------------------------
class_models = {
    "decision_tree": tree.DecisionTreeClassifier(max_depth=6, random_state=rand, class_weight="balanced"),
    "gradient_boosting": ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=4, subsample=0.5, learning_rate=0.05),
    "random_forest": ensemble.RandomForestClassifier(max_depth=11, n_estimators=300, max_features="sqrt", random_state=rand),
    "logistic": linear_model.LogisticRegression(multi_class="ovr", solver="lbfgs", class_weight="balanced", max_iter=500),
    "lda": discriminant_analysis.LinearDiscriminantAnalysis(n_components=2),
    "mlp": make_pipeline(
        StandardScaler(),
        neural_network.MLPClassifier(hidden_layer_sizes=(11,), early_stopping=True, random_state=rand, validation_fraction=0.25, max_iter=500),
    ),
}

# -----------------------------
# Train + Evaluate
# -----------------------------
rows = []
for name, model in class_models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rows.append({
        "model": name,
        "Accuracy_train": metrics.accuracy_score(y_train, y_train_pred),
        "Accuracy_test": metrics.accuracy_score(y_test, y_test_pred),
        "Recall_train": metrics.recall_score(y_train, y_train_pred, average="weighted"),
        "Recall_test": metrics.recall_score(y_test, y_test_pred, average="weighted"),
        "Precision_train": metrics.precision_score(y_train, y_train_pred, average="weighted"),
        "Precision_test": metrics.precision_score(y_test, y_test_pred, average="weighted"),
        "F1_test": metrics.f1_score(y_test, y_test_pred, average="weighted"),
        "MCC_test": metrics.matthews_corrcoef(y_test, y_test_pred),
    })

metrics_df = pd.DataFrame(rows).sort_values("MCC_test", ascending=False)
print(metrics_df)
