import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1) SUPERVISED - REGRESSION: LINEAR REGRESSION (WITH GRAPH)
# ============================================================
X_reg = np.array([800, 1000, 1200, 1500, 1800, 2000, 2300, 2500, 2800, 3000]).reshape(-1, 1)
y_reg = np.array([200, 240, 280, 330, 370, 410, 460, 500, 560, 600])

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\n[Linear Regression]")
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Plot regression fit
plt.figure()
plt.scatter(X_reg, y_reg)
x_line = np.linspace(X_reg.min(), X_reg.max(), 200).reshape(-1, 1)
plt.plot(x_line, lr.predict(x_line))
plt.title("Linear Regression: House Price vs Sqft")
plt.xlabel("Sqft")
plt.ylabel("Price ($1000s)")
plt.show()


# ============================================================
# 2) SUPERVISED - CLASSIFICATION: LOGISTIC REGRESSION (WITH GRAPHS)
# ============================================================
# Toy feature: length_cm
X_clf = np.array([30, 40, 150, 160, 120, 15, 25, 20, 90, 10]).reshape(-1, 1)
y_clf = np.array(['y','y','y','y','n','n','y','y','n','n'])  # y = has legs, n = no legs

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

logr = LogisticRegression()
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n[Logistic Regression]")
print("Accuracy:", acc)

# Plot probability curve P(y='y')
plt.figure()
x_vals = np.linspace(X_clf.min(), X_clf.max(), 300).reshape(-1, 1)
proba = logr.predict_proba(x_vals)
# class order corresponds to logr.classes_
idx_y = list(logr.classes_).index('y')
plt.scatter(X_clf.flatten(), (y_clf == 'y').astype(int))
plt.plot(x_vals.flatten(), proba[:, idx_y])
plt.title("Logistic Regression: P(has legs='y') vs length")
plt.xlabel("length_cm")
plt.ylabel("Probability / Actual (0/1)")
plt.show()

# Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred, labels=['y','n'])
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["has legs (y)", "no legs (n)"])
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# ============================================================
# 3) UNSUPERVISED: K-MEANS CLUSTERING (WITH GRAPH)
# ============================================================
X_km = np.array([
    [25, 20], [27, 18], [24, 22],     # cluster A
    [60, 65], [62, 63], [58, 67],     # cluster B
    [90, 15], [88, 18], [92, 12]      # cluster C
])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_km)
centers = kmeans.cluster_centers_

print("\n[K-Means]")
print("Centers:\n", centers)

plt.figure()
plt.scatter(X_km[:, 0], X_km[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ============================================================
# 4) UNSUPERVISED: PCA DIMENSION REDUCTION (WITH GRAPH)
# ============================================================
X_pca = np.array([
    [1, 2, 1.1,  10, 11, 10.5],
    [2, 3, 2.0,  20, 19, 20.5],
    [3, 4, 3.2,  30, 29, 31.0],
    [4, 5, 3.9,  40, 41, 39.0],
    [5, 6, 5.1,  50, 49, 52.0],
])

X_scaled = StandardScaler().fit_transform(X_pca)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print("\n[PCA]")
print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title("PCA: Data Reduced to 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# ============================================================
# 5) ENSEMBLE (BAGGING): RANDOM FOREST (WITH CONFUSION MATRIX GRAPH)
# ============================================================
# Use same classification data but as 0/1 labels
y01 = (y_clf == 'y').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y01, test_size=0.3, random_state=42, stratify=y01
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("\n[Random Forest]")
print("Accuracy:", accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no legs (0)", "has legs (1)"])
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()


# ============================================================
# 6) ENSEMBLE (BOOSTING): GRADIENT BOOSTING (WITH CONFUSION MATRIX GRAPH)
# ============================================================
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
pred = gb.predict(X_test)

print("\n[Gradient Boosting]")
print("Accuracy:", accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no legs (0)", "has legs (1)"])
disp.plot()
plt.title("Gradient Boosting Confusion Matrix")
plt.show()


# ============================================================
# 7) ENSEMBLE (STACKING): STACKING CLASSIFIER (WITH CONFUSION MATRIX GRAPH)
# ============================================================
stack = StackingClassifier(
    estimators=[
        ("knn", KNeighborsClassifier(n_neighbors=3)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ],
    final_estimator=LogisticRegression(),
)

stack.fit(X_train, y_train)
pred = stack.predict(X_test)

print("\n[Stacking]")
print("Accuracy:", accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no legs (0)", "has legs (1)"])
disp.plot()
plt.title("Stacking Confusion Matrix")
plt.show()
