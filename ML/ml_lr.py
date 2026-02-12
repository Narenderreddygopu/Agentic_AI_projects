import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

X = np.array([30, 40, 150, 160, 120, 15, 25, 20, 90, 10]).reshape(-1, 1)
y = np.array(['y','y','y','y','n','n','y','y','n','n'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=['y','n'])
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", round(acc,2))
print(classification_report(y_test, y_pred, zero_division=0))
