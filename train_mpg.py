'''
Catherine Rodriguez
Project 1 - CSC 4700
'''

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from mpg_dataloader import dataloader
from mlp import MultilayerPerceptron, Layer, Relu, SquaredError

# Plot Graph
def plot_loss_curves(training_losses, validation_losses, epochs):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), training_losses, label="Training Loss")
    plt.plot(range(1, epochs+1), validation_losses, label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load the data using the dataloader
    X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std = dataloader()

    # Reshape targets to be 2D: (n_samples, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)


    # Define the MLP architecture
    layer1 = Layer(fan_in=X_train.shape[1], fan_out=784, activation_function=Relu())
    layer2 = Layer(fan_in=784, fan_out=128, activation_function=Relu())
    output_layer = Layer(fan_in=128, fan_out=1, activation_function=None)  # No activation for regression

    loss_function = SquaredError()  # Use squared error for regression
    mlp_model = MultilayerPerceptron(layers=(layer1, layer2, output_layer))

    # Set training hyperparameters
    learning_rate = 1e-3
    batch_size = 32
    epochs = 10

    print("Starting training...")
    training_losses, validation_losses = mlp_model.train(
        X_train, y_train, X_val, y_val,
        loss_func=loss_function,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )

    # Plot the loss curves
    plot_loss_curves(training_losses, validation_losses, epochs)

if __name__ == "__main__":
    main()
