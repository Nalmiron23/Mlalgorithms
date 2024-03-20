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
