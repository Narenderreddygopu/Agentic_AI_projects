import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Sample data
animals = ['dog', 'fish', 'cat', 'snake', 'frog', 'eel', 'horse', 'snail', 'rabbit', 'turtle']
actual = ['legs', 'no legs', 'legs', 'no legs', 'legs', 'no legs', 'legs', 'no legs', 'legs', 'no legs']
predicted = ['legs', 'no legs', 'legs', 'legs', 'no legs', 'no legs', 'legs', 'no legs', 'legs', 'no legs']

# Create confusion matrix
cm = confusion_matrix(actual, predicted, labels=['legs', 'no legs'])
print(f"Confusion Matrix:\n{cm}")
print(f"Confusion Matrix:{cm}")
accuracy = np.trace(cm) / np.sum(cm) # (True Positives + True Negatives) / Total
print(f"Accuracy: {accuracy:.2f}")
precision = cm[0, 0] / (cm[0, 0] + cm[1, 0]) # True Positives / (True Positives + False Positives)
print(f"Precision: {precision:.2f}")
recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # True Positives / (True Positives + False Negatives)
print(f"Recall (Sensitivity): {recall:.2f}")
score = cm[0, 0] / (cm[0, 0] + cm[1, 0])  # True Positives / (True Positives + False Positives)
print(f"Precision: {score:.2f}")
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"F1 Score: {f1_score:.2f}")


tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

print(f"True Positives (TP): {tp}")
print(f"False Negatives (FN): {fn}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")


# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['legs', 'no legs'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Animal Legs Classification")
plt.show()