"""
 - Input layer
 - One hidden Layer
 - Output Layer
 - Activations Functions
 - Forward Propagation
 - Back Propagation
 - Loss function
 - Epoch training data
 
 
 0 -> False
1 -> True

OR :
0 or 0 = 0
0 or 1 = 1
1 or 0 = 1
1 or 1 = 1

AND #

0 AND 0 = 0
0 AND 1 = 0
1 AND 0 = 0
1 AND 1 = 1

XNOR Gate
0 XNOR 0 = 1
1 XNOR 0 = 0
0 XNOR 1 = 0
1 XNOR 1 = 1

XOR Gate
0 XOR 0 = 0
1 XOR 0 = 1
0 XOR 1 = 1
1 XOR 1 = 0


XOR Gate
0 XOR 0 = 0
1 XOR 0 = 1
0 XOR 1 = 1
1 XOR 1 = 0
# input layer = 2 (no. of neurons)
# output layer = 1
# hidden layer = 4

# activation function = sigmoid function
# Loss = Mean Squared error -> error
"""



import numpy as np

#1 Dataset (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# 2 Activation Function :
def sigmoid(x) :
  return 1/(1+np.exp(-x))
def sigmoid_derivative(x) :
  return x*(1-x)


# 3 Initialize weights
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

w1 = np.random.randn(input_neurons,hidden_neurons)
#b1 = np.zeros(())
b1 = np.zeros((1, hidden_neurons))

w2 = np.random.randn(hidden_neurons,output_neurons)
#b2 = np.zeros(())
b2 = np.zeros((1, output_neurons))

learning_rate=0.10
epochs = 1000

for epoch in range(epochs):
  hidden_input = np.dot(X,w1) + b1 # prediction = input*slope(Weight) + coeff(bias)
  hidden_output = sigmoid(hidden_input)

  final_input = np.dot(hidden_output,w2) + b2
  predicted_output = sigmoid(final_input)

  # loss  :
  loss = np.mean((Y-predicted_output)**2)

  # error
  error = Y - predicted_output #error = actual - predicted or predicted - target
  d_output = error*sigmoid_derivative(predicted_output) 

  error_hidden = d_output.dot(w2.T)
  d_hidden = error_hidden*sigmoid_derivative(hidden_output)
  b1 = b1 + np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
  w1 = w1 + X.T.dot(d_hidden) * learning_rate
  w2 = w2 + hidden_output.T.dot(d_output)*learning_rate
  #b2 = b2 + np.sum(d_hidden,axis = 0,keepdims = True)*learning_rate
  b2 = b2 + np.sum( d_output, axis=0, keepdims=True) * learning_rate

  if epoch%100 == 0:
    print(f"Epoch is  {epoch}, Loss is : {loss}, Predicted Output is : {predicted_output}, Error is : {error}, Weights are : {w1}, {w2}, Biases are : {b1}, {b2} , Learning Rate is : {learning_rate}")

print("final predictions",predicted_output)