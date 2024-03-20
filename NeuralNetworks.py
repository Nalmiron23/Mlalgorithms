# Import packages 

import numpy as np
print("This is a neural network from scratch using one hidden layer")

# Define the layer size for the network

def layer_size(X,Y):
    input_size = X.shape[0]
    hidden_size = 3
    output_size = Y.shape[0]
    
    return input_size, hidden_size, output_size

 #Initialise the weights and biases randomnly to enable symmetry breaking
def initialise_parameters(input_size, hidden_size, output_size):

    W1 = np.random.randn(hidden_size, input_size) * 0.01
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((output_size, 1))

    # store in dict for access
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

# Sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Derivative
def d_sigmoid(z):
    return z*(1-z)
# Forward prop
def forward(X, parameters):

    # Get parameters from dict
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # Calculate the forward using the activation function
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    # Store the values in a dict
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache

# Cross entropy loss function
def compute_cost(A2, Y):
    
    cost = (-1/Y.shape[1]) * np.sum(Y*np.log(A2) + (1-Y) * np.log(1-A2))
    cost = float(np.squeeze(cost))
    
    return cost

