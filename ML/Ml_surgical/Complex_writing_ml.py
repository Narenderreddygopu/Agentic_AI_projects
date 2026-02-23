# ------------------------------------------------------------
# Book example: Personality and Birth Order
# Interpretable ML example (Clean Professional Version)
# ------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model, tree, discriminant_analysis, ensemble, neural_network
from sklearn.multiclass import OneVsRestClassifier

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

DATA_PATH = r"/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/ML/Ml_surgical/birthorder.csv"

birthorder_df = pd.read_csv(DATA_PATH)

if "birthorder" not in birthorder_df.columns:
    raise ValueError("Column 'birthorder' not found in CSV.")

print("Dataset Shape:", birthorder_df.shape)
print("Class Distribution:\n", birthorder_df["birthorder"].value_counts())
print("-" * 50)

# ------------------------------------------------------------
# Train/Test Split (Stratified for better class balance)
# ------------------------------------------------------------

rand = 9

y = birthorder_df["birthorder"]
X = birthorder_df.drop(["birthorder"], axis=1).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=rand,
    stratify=y  # important improvement
)

# ------------------------------------------------------------
# Define Models
# ------------------------------------------------------------

class_models = {
    "decision_tree": tree.DecisionTreeClassifier(
        max_depth=6,
        random_state=rand,
        class_weight="balanced"
    ),

    "gradient_boosting": ensemble.GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        subsample=0.5,
        learning_rate=0.05
    ),

    "random_forest": ensemble.RandomForestClassifier(
        max_depth=11,
        n_estimators=300,
        max_features="sqrt",
        random_state=rand
    ),

    # Fixed Logistic Regression (no deprecation warning)
    "logistic": OneVsRestClassifier(
        linear_model.LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500
        )
    ),

    "lda": discriminant_analysis.LinearDiscriminantAnalysis(
        n_components=2
    ),

    "mlp": make_pipeline(
        StandardScaler(),
        neural_network.MLPClassifier(
            hidden_layer_sizes=(11,),
            early_stopping=True,
            random_state=rand,
            validation_fraction=0.25,
            max_iter=500
        ),
    ),
}

# ------------------------------------------------------------
# Train + Evaluate
# ------------------------------------------------------------

results = []

for name, model in class_models.items():

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results.append({
        "model": name,
        "Accuracy_train": metrics.accuracy_score(y_train, y_train_pred),
        "Accuracy_test": metrics.accuracy_score(y_test, y_test_pred),

        "Recall_train": metrics.recall_score(
            y_train, y_train_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall_test": metrics.recall_score(
            y_test, y_test_pred,
            average="weighted",
            zero_division=0
        ),

        "Precision_train": metrics.precision_score(
            y_train, y_train_pred,
            average="weighted",
            zero_division=0
        ),
        "Precision_test": metrics.precision_score(
            y_test, y_test_pred,
            average="weighted",
            zero_division=0
        ),

        "F1_test": metrics.f1_score(
            y_test, y_test_pred,
            average="weighted"
        ),

        "MCC_test": metrics.matthews_corrcoef(
            y_test, y_test_pred
        ),
    })

# ------------------------------------------------------------
# Results Table
# ------------------------------------------------------------

metrics_df = pd.DataFrame(results)
metrics_df = metrics_df.sort_values("MCC_test", ascending=False)

pd.set_option("display.precision", 4)
print("\nModel Performance (Sorted by MCC):\n")
print(metrics_df)
print("-" * 50)

# ------------------------------------------------------------
# Optional: Confusion Matrix for Best Model
# ------------------------------------------------------------

best_model_name = metrics_df.iloc[0]["model"]
best_model = class_models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print("Confusion Matrix:\n")
print(metrics.confusion_matrix(y_test, best_model.predict(X_test)))