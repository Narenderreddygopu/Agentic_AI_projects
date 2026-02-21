import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# X = [age, income_k]
X = np.array([
    [22, 30], [25, 35], [28, 40], [35, 80], [40, 90],
    [45, 95], [50, 110], [23, 25], [55, 120], [60, 130]
])
# y = loan approved? (1/0)
y = np.array([0,0,0,1,1,1,1,0,1,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)

pred = mlp.predict(X_test)
print("MLP tabular accuracy:", round(accuracy_score(y_test, pred), 4))