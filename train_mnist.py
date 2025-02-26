'''
Catherine Rodriquez
Project 1 - CSC 4700
'''

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from mlp import MultilayerPerceptron, Layer, Linear, Relu, Softmax, CrossEntropy
from mnist_dataloader import MnistDataloader

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

# Report accuracy
def accuracy(mlp_model, test_x, test_y):
    """Compute accuracy on the test set."""
    correct = 0
    total = len(test_x)
    
    for i in range(total):
        output = test_x[i]
        for layer in mlp_model.layers:
            output = layer.forward(output)

        predicted_label = np.argmax(output)
        true_label = np.argmax(test_y[i])
        
        if predicted_label == true_label:
            correct += 1
    
    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Show samples for each class
def sampleImages(mlp_model, test_x, test_y, num_classes=10):
    """Select one sample from each class and show the image with its predicted label."""
    plt.figure(figsize=(15, 10))
    for class_id in range(num_classes):
        class_indices = np.where(np.argmax(test_y, axis=1) == class_id)[0]  
        sample_index = np.random.choice(class_indices)  # Randomly select one image from the class
        sample_image = test_x[sample_index].reshape(28, 28)  # Reshape to 28x28 image
        true_label = class_id

        output = test_x[sample_index]
        for layer in mlp_model.layers:
            output = layer.forward(output)

        predicted_label = np.argmax(output)

        # Display the image
        plt.subplot(3, 4, class_id + 1)
        plt.imshow(sample_image, cmap="gray")
        plt.title(f"True: {true_label}, Pred: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
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
    epochs = 50

    print("Starting training...")
    training_losses, validation_losses = mlp_model.train(
        train_x, train_y, val_x, val_y,
        loss_func=loss_function,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )

    # Plot the loss curves
    plotLoss(training_losses, validation_losses, epochs)

    # Report accuracy on the full test dataset
    accuracy(mlp_model, test_x, test_y)

    # Show one sample from each class and display predicted labels
    sampleImages(mlp_model, test_x, test_y)

if __name__ == "__main__":
    main()
