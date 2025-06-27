"""
Feedforward Neural Network Classifier for Diabetes Dataset
==========================================================

Author: Jackson Wurzer  
Created: May 18, 2025  
Updated Last: June 25, 2025  
Contact: jacksonwurzer@outlook.com

Description:
------------
This script implements a feedforward neural network from scratch. 
The network classifies diabetes outcomes using the dataset from the "Pima Indians Diabetes Database".
The network architecture is 8–15–15–2, with ReLU activations, mean squared error loss, 
and trained using gradient descent. 

Main Features:
--------------
- Data loading and preprocessing with scikit-learn
- Manual forward and backward propagation
- Training loop with adjustable learning rate and epoch count
- Final evaluation using a confusion matrix

Libraries:
-------------
- numpy
- scipy
- scikit-learn

Usage:
------
Make sure `diabetes.mat` is in the same directory. Run the script to train the model and 
output a confusion matrix to evaluate its performance on the test set.

"""


# Load necessary packages
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load in data
data = scipy.io.loadmat('diabetes.mat')
X = data['P']  # Shape: (768, 8)
T = data['T']  # Shape: (768, 2)

# Scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, T, test_size=0.3, random_state=42)

# Defining ReLU as the activation function
def relu(x):
    return np.maximum(0, x)

# Defining derivative of ReLU
def d_relu(x):
    return (x > 0).astype(float)

# Initialize weights and biases for each layer of the network
def initialize_weights(input_size, hidden_size1, hidden_size2, output_size):
    # Initialization is sensitive — small values help stabilize learning
    W1 = np.random.randn(hidden_size1, input_size) * 0.1
    b1 = np.zeros((hidden_size1, 1))
    W2 = np.random.randn(hidden_size2, hidden_size1) * 0.1
    b2 = np.zeros((hidden_size2, 1))
    W3 = np.random.randn(output_size, hidden_size2) * 0.1
    b3 = np.zeros((output_size, 1))
    return W1, b1, W2, b2, W3, b3

# Perform forward pass through the network
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    P1 = W1 @ X.T + b1
    S1 = relu(P1)
    P2 = W2 @ S1 + b2
    S2 = relu(P2)
    P3 = W3 @ S2 + b3
    S3 = P3  # No activation on output layer (linear)
    return P1, S1, P2, S2, P3, S3

# Compute mean squared error loss
def compute_loss(Y_true, Y_pred):
    return np.mean((Y_true.T - Y_pred) ** 2)

# Backpropagation to compute gradients
def backward_pass(X, Y, P1, S1, P2, S2, P3, S3, W2, W3):
    m = X.shape[0]

    # Output layer gradient
    dP3 = (S3 - Y.T) * d_relu(P3)  # This works for ReLU or linear output
    dW3 = (1/m) * dP3 @ S2.T
    db3 = (1/m) * np.sum(dP3, axis=1, keepdims=True)

    # Hidden layer 2 gradient
    dP2 = (W3.T @ dP3) * d_relu(P2)
    dW2 = (1/m) * dP2 @ S1.T
    db2 = (1/m) * np.sum(dP2, axis=1, keepdims=True)

    # Hidden layer 1 gradient
    dP1 = (W2.T @ dP2) * d_relu(P1)
    dW1 = (1/m) * dP1 @ X
    db1 = (1/m) * np.sum(dP1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Training loop — tweak learning rate and epochs as needed
def train(X_train, Y_train, input_size=8, hidden_size1=15, hidden_size2=15, output_size=2, alpha=0.1, epochs=200):
    W1, b1, W2, b2, W3, b3 = initialize_weights(input_size, hidden_size1, hidden_size2, output_size)
    errors = []

    for epoch in range(epochs):
        # Forward pass
        P1, S1, P2, S2, P3, S3 = forward_pass(X_train, W1, b1, W2, b2, W3, b3)

        # Compute and record loss
        loss = compute_loss(Y_train, S3)
        errors.append(loss)

        # Backward pass
        dW1, db1, dW2, db2, dW3, db3 = backward_pass(X_train, Y_train, P1, S1, P2, S2, P3, S3, W2, W3)

        # Update weights and biases
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3

    return W1, b1, W2, b2, W3, b3, errors

# Train the network — sensitivity to alpha/epochs may require tuning
W1, b1, W2, b2, W3, b3, errors = train(X_train, Y_train, alpha=0.1, epochs=200)

# Predict on test set
P1, S1, P2, S2, P3, Y_pred_test = forward_pass(X_test, W1, b1, W2, b2, W3, b3)
Y_pred_labels = np.argmax(Y_pred_test, axis=0)
Y_true_labels = np.argmax(Y_test, axis=1)

# Display confusion matrix 
conf_mat = confusion_matrix(Y_true_labels, Y_pred_labels)
print("\n Confusion Matrix:\n", conf_mat)

accuracy = np.mean(Y_pred_labels == Y_true_labels)
print(f"\n Test Accuracy: {accuracy:.4f}")

