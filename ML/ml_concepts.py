"""
Machine Learning Algorithms - Complete Guide
From Linear Regression to Advanced Techniques
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix




# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("=" * 50)
logger.info("# Setup logging is done by (using logger)")
logger.info("=" * 50)


print("=" * 100)
print("MACHINE LEARNING ALGORITHMS - COMPREHENSIVE EXAMPLES")
print("=" * 100)

# ============================================================================
# 1. LINEAR REGRESSION (Review - You already know this!)
# ============================================================================
print("\n" + "=" * 80)
print("1. LINEAR REGRESSION - Predicting Continuous Values")
print("=" * 80)

# Data: House size (sq ft) vs Price ($1000s)
X_linear = np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]).reshape(-1, 1)
y_linear = np.array([150, 180, 220, 250, 280, 320, 350, 380, 410, 450])

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train_lr)
y_pred_lr = model_lr.predict(X_test_lr)
logger.info("Linear Regression Coefficients:")
print(f"Slope: {model_lr.coef_[0]:.4f} (price increase per sq ft)")
print(f"Intercept: {model_lr.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test_lr, y_pred_lr):.2f}")
print(f"\nExample: A 1500 sq ft house costs: ${model_lr.predict([[1500]])[0]:.2f}k")

# ============================================================================
# 2. POLYNOMIAL REGRESSION - Fitting Curved Patterns
# ============================================================================
print("\n" + "=" * 80)
print("2. POLYNOMIAL REGRESSION - When Linear Lines Don't Fit")
print("=" * 80)

# Data: Temperature vs Ice Cream Sales (curved relationship)
X_poly = np.array([10, 15, 20, 25, 30, 35, 40]).reshape(-1, 1)
y_poly = np.array([20, 35, 60, 95, 140, 195, 260])  # Sales increase faster at higher temps

# Transform to polynomial features (degree 2: X, X²)
poly_features = PolynomialFeatures(degree=2)
X_poly_transformed = poly_features.fit_transform(X_poly)

model_poly = LinearRegression()
model_poly.fit(X_poly_transformed, y_poly)
y_pred_poly = model_poly.predict(X_poly_transformed)

print(f"Original features: [temperature]")
print(f"Polynomial features: [1, temperature, temperature²]")
print(f"Mean Squared Error: {mean_squared_error(y_poly, y_pred_poly):.2f}")
print(f"\nAt 28°C, predicted sales: {model_poly.predict(poly_features.transform([[28]]))[0]:.0f} units")

# ============================================================================
# 3. MULTIPLE LINEAR REGRESSION - Multiple Input Features
# ============================================================================
print("\n" + "=" * 80)
print("3. MULTIPLE LINEAR REGRESSION - Using Multiple Features")
print("=" * 80)

# Data: House price based on size, bedrooms, and age
# Features: [Size (sq ft), Bedrooms, Age (years)]
X_multi = np.array([
    [1200, 3, 10],
    [1500, 3, 5],
    [1800, 4, 8],
    [2000, 4, 3],
    [2200, 5, 15],
    [1000, 2, 20],
    [1600, 3, 12],
    [2400, 5, 2],
    [1400, 3, 7],
    [1900, 4, 6]
])
y_multi = np.array([250, 300, 350, 400, 420, 180, 290, 480, 270, 380])

X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_mlr = LinearRegression()
model_mlr.fit(X_train_mlr, y_train_mlr)
y_pred_mlr = model_mlr.predict(X_test_mlr)

print(f"Coefficients:")
print(f"  Size: {model_mlr.coef_[0]:.4f} ($/sq ft)")
print(f"  Bedrooms: {model_mlr.coef_[1]:.2f} ($/bedroom)")
print(f"  Age: {model_mlr.coef_[2]:.2f} ($/year)")
print(f"Intercept: {model_mlr.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test_mlr, y_pred_mlr):.2f}")

# Prediction
new_house = [[1700, 3, 5]]  # 1700 sq ft, 3 bedrooms, 5 years old
print(f"\nPredicted price for 1700 sq ft, 3 bed, 5yr house: ${model_mlr.predict(new_house)[0]:.2f}k")

# ============================================================================
# 4. LOGISTIC REGRESSION - Classification (Binary)
# ============================================================================
print("\n" + "=" * 80)
print("4. LOGISTIC REGRESSION - Yes/No Classification")
print("=" * 80)

# Data: Student study hours vs Pass/Fail
# Features: [Study Hours, Previous Score]
X_log = np.array([
    [1, 45], [2, 50], [3, 55], [4, 60], [5, 65],
    [6, 70], [7, 75], [8, 80], [9, 85], [10, 90],
    [2, 40], [3, 48], [4, 52], [5, 58], [6, 62]
])
y_log = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])  # 0=Fail, 1=Pass

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.3, random_state=42)

model_log = LogisticRegression()
model_log.fit(X_train_log, y_train_log)
y_pred_log = model_log.predict(X_test_log)

print(f"Accuracy: {accuracy_score(y_test_log, y_pred_log) * 100:.1f}%")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test_log, y_pred_log))

# Probability prediction
new_student = [[5, 62]]
prob = model_log.predict_proba(new_student)[0]
print(f"\nStudent with 5 hours study, 62 previous score:")
print(f"  Probability of Failing: {prob[0]*100:.1f}%")
print(f"  Probability of Passing: {prob[1]*100:.1f}%")

# ============================================================================
# 5. DECISION TREE - Intuitive Rule-Based Learning
# ============================================================================
print("\n" + "=" * 80)
print("5. DECISION TREE - Rule-Based Decisions")
print("=" * 80)

# Data: Loan Approval based on Income and Credit Score
# Features: [Income (in $1000s), Credit Score]
X_tree = np.array([
    [30, 600], [40, 650], [50, 700], [60, 750], [70, 800],
    [35, 550], [45, 620], [55, 680], [65, 720], [75, 780],
    [32, 580], [48, 640], [58, 710], [68, 770], [80, 820]
])
y_tree = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1])  # 0=Rejected, 1=Approved

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.3, random_state=42)

model_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
model_tree.fit(X_train_tree, y_train_tree)
y_pred_tree = model_tree.predict(X_test_tree)

print(f"Accuracy: {accuracy_score(y_test_tree, y_pred_tree) * 100:.1f}%")
print(f"Feature Importances:")
print(f"  Income: {model_tree.feature_importances_[0]:.3f}")
print(f"  Credit Score: {model_tree.feature_importances_[1]:.3f}")

# Example prediction
applicant = [[52, 690]]
decision = "Approved" if model_tree.predict(applicant)[0] == 1 else "Rejected"
print(f"\nApplicant: $52k income, 690 credit score → {decision}")

# ============================================================================
# 6. RANDOM FOREST - Multiple Trees Voting Together
# ============================================================================
print("\n" + "=" * 80)
print("6. RANDOM FOREST - Wisdom of the Crowd")
print("=" * 80)

# Using same loan data as Decision Tree
model_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model_rf.fit(X_train_tree, y_train_tree)
y_pred_rf = model_rf.predict(X_test_tree)

print(f"Number of Trees: 100")
print(f"Accuracy: {accuracy_score(y_test_tree, y_pred_rf) * 100:.1f}%")
print(f"Feature Importances:")
print(f"  Income: {model_rf.feature_importances_[0]:.3f}")
print(f"  Credit Score: {model_rf.feature_importances_[1]:.3f}")

# Compare with Decision Tree
print(f"\nComparison:")
print(f"  Single Decision Tree: {accuracy_score(y_test_tree, y_pred_tree) * 100:.1f}%")
print(f"  Random Forest (100 trees): {accuracy_score(y_test_tree, y_pred_rf) * 100:.1f}%")

# ============================================================================
# 7. SUPPORT VECTOR MACHINE (SVM) - Finding Optimal Boundaries
# ============================================================================
print("\n" + "=" * 80)
print("7. SUPPORT VECTOR MACHINE - Maximum Margin Classification")
print("=" * 80)

# Data: Customer segments based on Age and Spending
X_svm = np.array([
    [25, 40], [30, 50], [35, 60], [40, 70], [45, 80],
    [50, 45], [55, 55], [60, 65], [65, 75], [70, 85],
    [28, 90], [33, 95], [38, 100], [43, 110], [48, 120]
])
y_svm = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # 0=Low, 1=Medium, 2=High spender

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.3, random_state=42)

model_svm = SVC(kernel='rbf', random_state=42)
model_svm.fit(X_train_svm, y_train_svm)
y_pred_svm = model_svm.predict(X_test_svm)

print(f"Kernel: RBF (Radial Basis Function)")
print(f"Accuracy: {accuracy_score(y_test_svm, y_pred_svm) * 100:.1f}%")

# Example prediction
customer = [[42, 85]]
segment = ["Low", "Medium", "High"][model_svm.predict(customer)[0]]
print(f"\nCustomer: 42 years old, $85k spending → {segment} Spender")

# ============================================================================
# 8. K-NEAREST NEIGHBORS (KNN) - "You Are Your Neighbors"
# ============================================================================
print("\n" + "=" * 80)
print("8. K-NEAREST NEIGHBORS - Similarity-Based Classification")
print("=" * 80)

# Data: Fruit classification based on Weight and Color intensity
X_knn = np.array([
    [150, 7], [160, 8], [170, 7.5], [180, 8.5],  # Apples (0)
    [120, 4], [130, 4.5], [125, 3.8], [135, 4.2],  # Oranges (1)
    [200, 2], [210, 2.5], [220, 2.2], [215, 1.8]   # Bananas (2)
])
y_knn = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.25, random_state=42)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train_knn, y_train_knn)
y_pred_knn = model_knn.predict(X_test_knn)

print(f"Number of Neighbors (K): 3")
print(f"Accuracy: {accuracy_score(y_test_knn, y_pred_knn) * 100:.1f}%")

# Example prediction
fruit = [[155, 7.2]]
fruit_name = ["Apple", "Orange", "Banana"][model_knn.predict(fruit)[0]]
print(f"\nFruit: 155g weight, 7.2 color intensity → Predicted: {fruit_name}")

# Show nearest neighbors
distances, indices = model_knn.kneighbors(fruit)
print(f"3 Nearest Neighbors: {indices[0]}")
print(f"Distances: {distances[0]}")

# ============================================================================
# VISUAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATION...")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Machine Learning Algorithms Comparison', fontsize=16, fontweight='bold')

# 1. Linear Regression
axes[0, 0].scatter(X_linear, y_linear, color='blue', label='Data')
axes[0, 0].plot(X_linear, model_lr.predict(X_linear), color='red', linewidth=2, label='Linear Fit')
axes[0, 0].set_title('1. Linear Regression')
axes[0, 0].set_xlabel('House Size (sq ft)')
axes[0, 0].set_ylabel('Price ($1000s)')
axes[0, 0].legend()

# 2. Polynomial Regression
axes[0, 1].scatter(X_poly, y_poly, color='blue', label='Data')
axes[0, 1].plot(X_poly, y_pred_poly, color='red', linewidth=2, label='Polynomial Fit')
axes[0, 1].set_title('2. Polynomial Regression')
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].set_ylabel('Ice Cream Sales')
axes[0, 1].legend()

# 3. Logistic Regression - Decision Boundary
axes[0, 2].scatter(X_log[y_log==0, 0], X_log[y_log==0, 1], color='red', label='Fail', alpha=0.6)
axes[0, 2].scatter(X_log[y_log==1, 0], X_log[y_log==1, 1], color='green', label='Pass', alpha=0.6)
axes[0, 2].set_title('3. Logistic Regression')
axes[0, 2].set_xlabel('Study Hours')
axes[0, 2].set_ylabel('Previous Score')
axes[0, 2].legend()

# 4. Decision Tree
axes[1, 0].scatter(X_tree[y_tree==0, 0], X_tree[y_tree==0, 1], color='red', label='Rejected', alpha=0.6)
axes[1, 0].scatter(X_tree[y_tree==1, 0], X_tree[y_tree==1, 1], color='green', label='Approved', alpha=0.6)
axes[1, 0].set_title('4. Decision Tree')
axes[1, 0].set_xlabel('Income ($1000s)')
axes[1, 0].set_ylabel('Credit Score')
axes[1, 0].legend()

# 5. SVM
colors_svm = ['red', 'blue', 'green']
for i in range(3):
    mask = y_svm == i
    axes[1, 1].scatter(X_svm[mask, 0], X_svm[mask, 1], 
                       color=colors_svm[i], 
                       label=['Low', 'Medium', 'High'][i], 
                       alpha=0.6)
axes[1, 1].set_title('5. Support Vector Machine')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Spending ($1000s)')
axes[1, 1].legend()

# 6. KNN
colors_knn = ['red', 'orange', 'yellow']
for i in range(3):
    mask = y_knn == i
    axes[1, 2].scatter(X_knn[mask, 0], X_knn[mask, 1], 
                       color=colors_knn[i], 
                       label=['Apple', 'Orange', 'Banana'][i], 
                       alpha=0.6, s=100)
axes[1, 2].set_title('6. K-Nearest Neighbors')
axes[1, 2].set_xlabel('Weight (g)')
axes[1, 2].set_ylabel('Color Intensity')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('/mnt/c/Users/gopur/OneDrive/Documents/Agentic_ai/ML/ml_algorithms_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'ml_algorithms_comparison.png'")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("ALGORITHM COMPARISON SUMMARY")
print("=" * 80)

summary = """
┌─────────────────────────┬──────────────┬─────────────────────────────────┐
│ Algorithm               │ Type         │ Best Used For                   │
├─────────────────────────┼──────────────┼─────────────────────────────────┤
│ Linear Regression       │ Regression   │ Straight-line relationships     │
│ Polynomial Regression   │ Regression   │ Curved relationships            │
│ Multiple Linear Reg.    │ Regression   │ Multiple input features         │
│ Logistic Regression     │ Classification│ Binary yes/no decisions        │
│ Decision Tree           │ Both         │ Interpretable rules             │
│ Random Forest           │ Both         │ High accuracy, less overfitting │
│ SVM                     │ Both         │ Complex boundaries              │
│ KNN                     │ Both         │ Similarity-based predictions    │
└─────────────────────────┴──────────────┴─────────────────────────────────┘
"""
print(summary)

print("\n" + "=" * 80)
print("KEY CONCEPTS:")
print("=" * 80)
print("""
1. REGRESSION: Predicting continuous numbers (price, temperature, sales)
2. CLASSIFICATION: Predicting categories (pass/fail, spam/not spam)
3. OVERFITTING: Model memorizes training data, fails on new data
4. UNDERFITTING: Model too simple, misses patterns
5. TRAIN/TEST SPLIT: Always test on unseen data!

LEARNING PATH:
Linear → Polynomial → Multiple → Logistic → Tree → Forest → SVM → KNN
""")

print("\n✓ All examples completed successfully!")
print("=" * 80)