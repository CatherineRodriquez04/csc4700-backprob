'''
Catherine Rodriquez
Project 1 - CSC 4700
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpg_dataloader import dataloader
from mlp import MultilayerPerceptron, Layer, Relu, SquaredError

# Plot Graph
def plotLoss(training_losses, validation_losses, epochs):
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

# Report testing loss and 10 sample predictions
def reportTesting(mlp_model, X_test, y_test, loss_func, y_mean, y_std):
    """Compute total testing loss and report predictions for 10 random test samples."""
    # Manually forward pass through the model to get predictions
    output = X_test
    for layer in mlp_model.layers:  # Assuming layers attribute exists
        output = layer.forward(output)

    predictions = output

    # Compute total testing loss using the loss function
    test_loss = loss_func.loss(predictions, y_test)
    print(f"Total testing loss (MSE): {test_loss:.4f}")

    # Select 10 random test samples
    indices = np.random.choice(range(len(X_test)), 10, replace=False)
    selected_samples = X_test.iloc[indices]
    true_mpg = y_test[indices]

    # Make predictions for the selected samples
    output_samples = selected_samples
    for layer in mlp_model.layers:  # Forward pass for selected samples
        output_samples = layer.forward(output_samples)

    predicted_mpg = output_samples

    # Reverse standardization to get original MPG values
    true_mpg_original = (true_mpg * y_std) + y_mean  # Convert back to original scale
    predicted_mpg_original = (predicted_mpg * y_std) + y_mean  # Convert back to original scale

    # Create a table for the true vs predicted MPG
    results = pd.DataFrame({
        'True MPG': true_mpg_original.flatten(),
        'Predicted MPG': predicted_mpg_original.flatten()
    })

    print("\n10 Random Test Samples - Predicted vs True MPG:")
    print(results)


def main():
    # Load the data using the dataloader
    X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std = dataloader()

    # Reshape targets to be 2D: (n_samples, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Define the MLP architecture
    layer1 = Layer(fan_in=X_train.shape[1], fan_out=64, activation_function=Relu())
    layer2 = Layer(fan_in=64, fan_out=32, activation_function=Relu())
    output_layer = Layer(fan_in=32, fan_out=1, activation_function=None)  # No activation for regression

    loss_function = SquaredError()  # Use squared error for regression
    mlp_model = MultilayerPerceptron(layers=(layer1, layer2, output_layer))

    # Set training hyperparameters
    learning_rate = 1e-3
    batch_size = 32
    epochs = 20

    print("Starting training...")
    training_losses, validation_losses = mlp_model.train(
        X_train, y_train, X_val, y_val,
        loss_func=loss_function,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )

    # Plot the loss curves
    plotLoss(training_losses, validation_losses, epochs)

    # Report testing loss and 10 sample predictions
    reportTesting(mlp_model, X_test, y_test, loss_function, y_mean, y_std)

if __name__ == "__main__":
    main()
