import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# x : 1,2,3
# y : 2,4,8

#observed x = 1, and observed y = 2 -> predicted formula - y = x * 2
# observed x = 2, and observed y = 4 -> predicted formula - y = x * 2
# observed x = 3, and observed y = 8 -> predicted formula - y = x * 2 -> failed 
# 1. Prepare the data
X = np.array([1,2,3,4,5,6,7,8,9,11]) #input parameter
Y = np.array([3,4,2,5,6,7,8,9,10,11]) #output parameter

X = X.reshape(-1, 1) # reshape the data to be in the form of a column vector
# 2. Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("training data (Y_train): ", Y_train)
# test_size = 0.2 -> 20% of the data will be used for testing, and the remaining 80% will be used for training

# example : 1000 data points -> since test size is 0.2 -> 20% should be test size -> 20% of 1000 = 200 data points will be used for testing.
# the other 800 will be used for training

#3. Choose the model and apply the model to the training data

model = LinearRegression()

#4. fit the model to the training data
model.fit(X_train, Y_train) # 80% of data is used for training

print("Slope (m): ", model.coef_)
print("Intercept (c): ", model.intercept_)

# 5. Make predictions using the testing data
Y_pred = model.predict(X_test) # 20% of data is used for testing because this data was not seen by the model and since we also have the actual values of Y_test, we can compare the predicted values with the actual values to evaluate the performance of the model.
print("Testing data (X_test): ", X_test)
print("Predicted values: ", Y_pred)

print("\n spotting the difference between the actual values and the predicted values")
print("Actual values (Y_test): ", Y_test)

# 6. Evaluate the model
# confusion matrix is used for classification problems, but since this is a regression problem, we will use mean squared error to evaluate the performance of the model.

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error: ", mse)

# if mse is high -> the model is not performing well and if mse is low -> the model is performing well.
r2 = r2_score(Y_test, Y_pred)
print(f"R² Score: {r2:.3f}")  # 1.0 = perfect, 0.0 = poor

# 2. Show residuals (errors) for each prediction
residuals = Y_test - Y_pred
print("\nResiduals:", residuals)
#plot 
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
#plt.legend()    
plt.show()
