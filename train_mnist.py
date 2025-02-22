'''
Catherine Rodriquez
Project 1 - CSC 4700
'''

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from mlp import MultilayerPerceptron, Layer, Relu, Softmax, CrossEntropy
from mnist_dataloader import MnistDataloader

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
    # File path
    mnist_folder = "mnist"
    train_images_path = join(mnist_folder, "train-images.idx3-ubyte")
    train_labels_path = join(mnist_folder, "train-labels.idx1-ubyte")
    test_images_path = join(mnist_folder, "t10k-images.idx3-ubyte")
    test_labels_path = join(mnist_folder, "t10k-labels.idx1-ubyte")

    # Load the data
    dataloader = MnistDataloader(train_images_path, train_labels_path, test_images_path, test_labels_path)
    (train_images, train_labels), (test_images, test_labels) = dataloader.load_data()

    # Flatten images (28x28 -> 784) and normalize pixel values to [0, 1]
    train_x = np.array(train_images).reshape(-1, 28*28).astype(np.float32) / 255.0
    test_x = np.array(test_images).reshape(-1, 28*28).astype(np.float32) / 255.0

    train_y = np.eye(10)[train_labels]  # Using np.eye for one-hot encoding
    test_y = np.eye(10)[test_labels]

    # Split into 80% training and 20% validation
    num_train = train_x.shape[0]
    split_index = int(0.8 * num_train)
    val_x = train_x[split_index:]
    val_y = train_y[split_index:]
    train_x = train_x[:split_index]
    train_y = train_y[:split_index]

    # Define the MLP architecture
    layer1 = Layer(fan_in=784, fan_out=128, activation_function=Relu())
    layer2 = Layer(fan_in=128, fan_out=64, activation_function=Relu())
    output_layer = Layer(fan_in=64, fan_out=10, activation_function=Softmax())

    mlp_model = MultilayerPerceptron(layers=(layer1, layer2, output_layer))
    loss_function = CrossEntropy()  # Use cross-entropy for multi-class classification

    # Set training hyperparameters
    learning_rate = 1e-4
    batch_size = 64
    epochs = 10

    print("Starting training...")
    training_losses, validation_losses = mlp_model.train(
        train_x, train_y, val_x, val_y,
        loss_func=loss_function,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )

    # Plot the loss curves
    plot_loss_curves(training_losses, validation_losses, epochs)

if __name__ == "__main__":
    main()
