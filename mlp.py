#activate virtual environment -> .\myenv\Scripts\Activate
#deactivate

'''
Catherine Rodriquez
Project 1 - CSC 4700
'''

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x: np.ndarray, train_y: np.ndarray, batch_size: int):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :yield tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    samples = train_x.shape[0]
    for i in range(0, samples, batch_size):
        yield train_x[i:i + batch_size], train_y[i:i + batch_size]


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        tanh_x = self.forward(x)
        return 1 - np.square(tanh_x)


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        jacobian = np.zeros((x.shape[0], s.shape[-1], s.shape[-1]))  # shape (batch_size, num_classes, num_classes)

        # Compute the Jacobian matrix for each input vector
        for i in range(x.shape[0]):  
            softmax_vector = s[i].reshape(-1, 1)  # reshape the i-th softmax vector into a column
            jacobian[i] = np.diagflat(softmax_vector) - np.dot(softmax_vector, softmax_vector.T)

        return jacobian



class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)



class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y_true - y_pred) / y_true.size


class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15 # Add small epsilon to avoid log(0) -> undefined
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Ensures y prediction never gets too close to 0 or 1
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # Formula
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15  # Add small value to prevent division by 0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid division by 0
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)



class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # Initialize weights and biaes
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))  # weights from Glorot distribution between (-limit, limit)
        self.b = np.zeros(fan_out)  # biases

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        # Linear Transformation: pre-activation (a) = W dot h + b
        a = np.dot(h, self.W) + self.b
        if self.activation_function is not None:
            self.activations = self.activation_function.forward(a)
        else:
            self.activations = a  # No activation for regression

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """

        # print("delta shape:", delta.shape)
        # print("activation function derivative shape:", self.activation_function.derivative(self.activations).shape)

        # for Softmax -> Jacobian
        if isinstance(self.activation_function, Softmax):
            jacobian = self.activation_function.derivative(self.activations)  # Jacobian (batch_size, num_classes, num_classes)

            # Ensure the delta is properly shaped by multiplying with the Jacobian
            self.delta = np.matmul(delta, jacobian)  

            # Reduce the delta to correct dimensions for weight update (batch_size, num_classes)
            self.delta = np.sum(self.delta, axis=1)  
        elif self.activation_function is not None:
            self.delta = delta * self.activation_function.derivative(self.activations)
        else:
            # No activation function (for linear regression)
            self.delta = delta

        dL_dW = np.dot(h.T, self.delta) # gradient for weight
        dL_db = np.sum(self.delta, axis=0) #gradient for bias

        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x) # apply forward propagation for each layer

        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []

        delta = loss_grad

        for i in reversed(range(len(self.layers))):
            # Check current layer to be the input layer or not 
            if i == 0:
                h = input_data
            else:
                h = self.layers[i-1].activations

            # Call backward method for current layer 
            # Compute weight and bias gradients given h and delta from next layer
            dL_dw, dL_db = self.layers[i].backward(h, delta)

            # Ensure gradients are listed in order -> from input to output layer
            dl_dw_all.insert(0, dL_dw)
            dl_db_all.insert(0, dL_db)

            # Update delta by multiplying current layer's delta with transpose of the weights
            delta = np.dot(self.layers[i].delta, self.layers[i].W.T)

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: 'LossFunction', learning_rate: float = 1E-3, batch_size: int = 16, epochs: int = 32) -> tuple:
        """
        Train the multilayer perceptron.
        :param train_x: training set input (n x d)
        :param train_y: training set output (n x q)
        :param val_x: validation set input
        :param val_y: validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: batch size
        :param epochs: number of epochs
        :return: (training losses, validation losses)
        """
        training_losses = []
        validation_losses = []
        batches_num = int(np.ceil(train_x.shape[0] / batch_size))

        # Determine task type: regression if train_y is 1D or has one column; classification otherwise.
        is_regression = (train_y.ndim == 1) or (train_y.ndim == 2 and train_y.shape[1] == 1)

        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_metric = 0.0

            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                predictions = self.forward(batch_x)
                loss = loss_func.loss(predictions, batch_y)
                epoch_train_loss += loss

                loss_gradient = loss_func.derivative(batch_y, predictions)
                dL_dW_all, dL_db_all = self.backward(loss_gradient, batch_x)

                for layer, dL_dW, dL_db in zip(self.layers, dL_dW_all, dL_db_all):
                    layer.W -= learning_rate * dL_dW
                    layer.b -= learning_rate * dL_db

                if is_regression:
                    # For regression: mean absolute error (MAE)
                    batch_metric = np.mean(np.abs(predictions - batch_y))
                else:
                    # For classification: accuracy
                    batch_metric = np.sum(np.argmax(predictions, axis=1) == np.argmax(batch_y, axis=1))
                epoch_metric += batch_metric

            epoch_train_loss /= batches_num
            epoch_metric /= train_x.shape[0]  # average per sample

            # Validation pass
            val_predictions = self.forward(val_x)
            val_loss = loss_func.loss(val_predictions, val_y)
            if is_regression:
                val_metric = np.mean(np.abs(val_predictions - val_y))
            else:
                val_metric = np.sum(np.argmax(val_predictions, axis=1) == np.argmax(val_y, axis=1)) / val_x.shape[0]

            training_losses.append(epoch_train_loss)
            validation_losses.append(val_loss)

            if is_regression:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_metric:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val MAE: {val_metric:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_metric:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metric:.4f}")

        return np.array(training_losses), np.array(validation_losses)