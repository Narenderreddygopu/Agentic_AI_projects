import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# ----- DATA (Regression) -----
#Data Predict house_price from sqft (continuous number)
# X = square feet, y = price in $1000s
X = np.array([800, 1000, 1200, 1500, 1800, 2000, 2300, 2500, 2800, 3000]).reshape(-1, 1)
y = np.array([200, 240, 280, 330, 370, 410, 460, 500, 560, 600])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("Predict 1500 sqft ($k):", model.predict([[1500]])[0])

# Plot regression fit
plt.figure()
plt.scatter(X, y)
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.plot(x_line, model.predict(x_line))
plt.title("Linear Regression: House Price vs Sqft")
plt.xlabel("Sqft")
plt.ylabel("Price ($1000s)")
plt.show()
