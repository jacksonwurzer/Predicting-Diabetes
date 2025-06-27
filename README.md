# Predicting Diabetes Using a Custom Neural Net

This project implements a feedforward neural network from scratch in NumPy. My goal was to use the neural net to classify diabetes outcomes of people from the Pima Indians Database dataset. The network uses ReLU as the activation function in the two hidden layers and a linear activation in the output layer. 


## Model Architecture

- Input Layer: 8 neurons (features)
- Hidden Layer 1: 15 neurons, ReLU activation
- Hidden Layer 2: 15 neurons, ReLU activation
- Output Layer: 2 neurons, linear output
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Gradient descent
- Learning Rate: 0.1
- Epochs: 200
