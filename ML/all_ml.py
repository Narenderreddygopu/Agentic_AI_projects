# ================================================================
#    REAL VISUALIZATION FOR EACH ML MODEL (UPDATED + FIXED)
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB

# ==========================
# 1. LOAD USER DATASET
# ==========================
np.random.seed(42)

df = pd.DataFrame({
    "Duration_Min": np.random.randint(60, 180, 50),
    "Viewer_Count": np.concatenate([np.random.randint(60, 100, 30),
                                    np.random.randint(1, 60, 20)]),
    "Genre": np.random.choice(["Action", "Drama", "Comedy", "Horror"], 50),
})

df["IMDB_Rating"] = (
    5 + (df["Viewer_Count"] / 100) * 3
    + (df["Duration_Min"] > 120).astype(int) * 0.3
    + np.random.uniform(-0.5, 0.5, 50)
).round(1)

X = df[["Duration_Min", "Viewer_Count"]].values
y_reg = df["IMDB_Rating"].values
y_clf = (y_reg > np.median(y_reg)).astype(int)

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X_scaled, y_clf, test_size=0.2, random_state=42
)

# ================================================================
#   2. LINEAR REGRESSION — TRUE BEST-FIT LINE (FIXED)
# ================================================================
lin = LinearRegression().fit(X_scaled, y_reg)

x_vals = np.linspace(X_scaled[:,0].min(), X_scaled[:,0].max(), 300)
X_slice = np.column_stack([x_vals, np.full_like(x_vals, X_scaled[:,1].mean())])
y_pred_slice = lin.predict(X_slice)

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], y_reg, color='blue', alpha=0.6)
plt.plot(x_vals, y_pred_slice, color='red', linewidth=3)
plt.title("Linear Regression — Best Fit Line (2D Slice)")
plt.xlabel("Duration (scaled)")
plt.ylabel("IMDB Rating")
plt.grid(True)
plt.show()

# ================================================================
#   3. LOGISTIC REGRESSION — SIGMOID CURVE
# ================================================================
log = LogisticRegression().fit(X_scaled[:,0].reshape(-1,1), y_clf)

x_vals = np.linspace(X_scaled[:,0].min(), X_scaled[:,0].max(), 300)
logits = log.predict_proba(x_vals.reshape(-1,1))[:,1]

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], y_clf, color='black')
plt.plot(x_vals, logits, color='red', linewidth=3)
plt.title("Logistic Regression — Sigmoid Curve (P(High Rating))")
plt.xlabel("Duration (scaled)")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# ================================================================
#   4. KNN — NEAREST NEIGHBOUR GRAPH (ACTUAL CONNECTIONS)
# ================================================================
knn = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, y_clf)

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_clf, cmap='coolwarm', s=90)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
dist, idx = nbrs.kneighbors(X_scaled)

for i in range(len(X_scaled)):
    for j in idx[i]:
        plt.plot([X_scaled[i,0], X_scaled[j,0]],
                 [X_scaled[i,1], X_scaled[j,1]],
                 'gray', linewidth=0.5)

plt.title("KNN — Nearest Neighbour Connections")
plt.xlabel("Duration (scaled)")
plt.ylabel("Viewer Count (scaled)")
plt.grid(True)
plt.show()

# ================================================================
#   5. K-MEANS — CLUSTERS + CENTROIDS
# ================================================================
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans.labels_, cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            color='red', marker='X', s=300)
plt.title("K-Means — Clusters + Centroids")
plt.xlabel("Duration (scaled)")
plt.ylabel("Viewer Count (scaled)")
plt.grid(True)
plt.show()

# ================================================================
#   6. DECISION TREE — THE ACTUAL TREE 

# ================================================================
dt = DecisionTreeClassifier(max_depth=4).fit(X_scaled, y_clf)

plt.figure(figsize=(22,12))
plot_tree(dt, feature_names=["Duration", "ViewerCount"], filled=True)
plt.title("Decision Tree — Full Tree Diagram")
plt.show()

# ================================================================
#   7. RANDOM FOREST — MULTI-TREE SNAPSHOT
# ================================================================
rf = RandomForestClassifier(n_estimators=4, max_depth=4).fit(X_scaled, y_clf)

fig, axes = plt.subplots(2,2, figsize=(22,14))
axes = axes.ravel()

for i, estimator in enumerate(rf.estimators_):
    plot_tree(estimator, feature_names=["Duration", "ViewerCount"], filled=True, ax=axes[i])
    axes[i].set_title(f"Tree {i+1}")

plt.suptitle("Random Forest — 4 Tree Snapshot", fontsize=20)
plt.show()

# ================================================================
#   8. SVM — MARGINS + SUPPORT VECTORS + HYPERPLANE
# ================================================================
svm = SVC(kernel="linear").fit(X_scaled, y_clf)

w = svm.coef_[0]
b = svm.intercept_[0]

xx = np.linspace(X_scaled[:,0].min(), X_scaled[:,0].max(), 300)
yy = -(w[0]*xx + b)/w[1]

margin = 1/np.sqrt(np.sum(w**2))
yy_down = yy - margin*np.sqrt(1 + (w[0]/w[1])**2)
yy_up   = yy + margin*np.sqrt(1 + (w[0]/w[1])**2)

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_clf, cmap='coolwarm', s=100)
plt.scatter(svm.support_vectors_[:,0], svm.support_vectors_[:,1],
            s=200, facecolors='none', edgecolors='black')

plt.plot(xx, yy, 'k-', linewidth=3)
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.title("SVM — Hyperplane, Margins, Support Vectors")
plt.xlabel("Duration (scaled)")
plt.ylabel("Viewer Count (scaled)")
plt.grid(True)
plt.show()

# ================================================================
#   9. GRADIENT BOOSTING — RESIDUAL REDUCTION PER STAGE
# ================================================================
gbr = GradientBoostingRegressor(n_estimators=5).fit(X_scaled, y_reg)

plt.figure(figsize=(10,6))
for i, pred in enumerate(gbr.staged_predict(X_scaled)):
    residuals = y_reg - pred
    plt.plot(residuals, label=f"Stage {i+1}")

plt.title("Gradient Boosting — Residuals Over Boosting Stages")
plt.xlabel("Sample Index")
plt.ylabel("Residual")
plt.legend()
plt.grid(True)
plt.show()

# ================================================================
#   10. NAIVE BAYES — GAUSSIAN DISTRIBUTIONS (FIXED)
# ================================================================
gnb = GaussianNB().fit(X_scaled, y_clf)

plt.figure(figsize=(10,6))
x_axis = np.linspace(-3, 3, 300)

for cls in [0,1]:
    mean = gnb.theta_[cls, 0]
    var  = gnb.var_[cls, 0]     # FIXED HERE!
    pdf = 1/np.sqrt(2*np.pi*var) * np.exp(-(x_axis-mean)**2/(2*var))
    plt.plot(x_axis, pdf, label=f"Class {cls}")

plt.title("Naive Bayes — Gaussian for Duration (per Class)")
plt.xlabel("Duration (scaled)")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()

# ================================================================
# END — FULL UPDATED SCRIPT (ALL MODELS FIXED)
# ================================================================