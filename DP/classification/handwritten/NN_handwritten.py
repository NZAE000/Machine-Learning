import matplotlib.pyplot as plt
import numpy as np
from read_mnist import load_mnist

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

class FeedforwardNN:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list with the number of neurons per layer (including input and output).
        Example: [784, 64, 10].
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        # Initialize weights and biases with small values.
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward propagation, returns activations and z-sums.
        """
        activations = [X]
        zs = []  # Sums before activation
        for i in range(self.num_layers - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)
        # Output layer with softmax (multiclass classification).
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        zs.append(z)
        activations.append(a)
        return activations, zs

    def compute_loss(self, y_true, y_pred):
        """
        Cross entropy loss for multiclass classification.
        y_true: one hot encoding.
        y_pred: probabilidades softmax.
        """
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred + 1e-9) * y_true
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, activations, zs, y_true):
        """
        Backpropagation to calculate gradients.
        """
        m = y_true.shape[0]
        grads_w = [0] * (self.num_layers - 1)
        grads_b = [0] * (self.num_layers - 1)

        # Output layer (softmax + crossentropy).
        delta = (activations[-1] - y_true) / m  # Derivative of loss with respect to z output.
        grads_w[-1] = np.dot(activations[-2].T, delta)
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Hidden layers back.
        for l in range(self.num_layers - 3, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * sigmoid_derivative(zs[l])
            grads_w[l] = np.dot(activations[l].T, delta)
            grads_b[l] = np.sum(delta, axis=0, keepdims=True)

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def train(self, X, y, epochs=10, batch_size=64, learning_rate=0.01, X_val=None, y_val=None):
        """
        Training with mini-batch gradient descent.
        'Y' must be in one-hot encoding.
        """
        n = X.shape[0]
        history = {"loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # Shuffle data.
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            epoch_loss = 0
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                activations, zs = self.forward(X_batch)
                loss = self.compute_loss(y_batch, activations[-1])
                epoch_loss += loss * X_batch.shape[0]

                grads_w, grads_b = self.backward(activations, zs, y_batch)
                self.update_params(grads_w, grads_b, learning_rate)

            epoch_loss /= n
            history["loss"].append(epoch_loss)

            # Evaluate validation.
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, self.forward(X_val)[0][-1])
                val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        return history

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)


# Load Data.
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
#im = X_train[10000, :].reshape(28,28)
#print(y_train[10000])
#plt.imshow(im)
#plt.show()

# Encode labels to one-hot (10 classes).
num_classes = 10
y_train_oh  = one_hot_encode(y_train, num_classes)
y_val_oh    = one_hot_encode(y_val, num_classes)
y_test_oh   = one_hot_encode(y_test, num_classes)

# Define hyperparameters to evaluate.
learning_rates = [0.1, 0.01, 0.001]
layers_configs = [
    [784, 32, 10],
    [784, 64, 10],
    [784, 128, 10],
    [784, 64, 64, 10]
]

# Store results.
results = []

for lr in learning_rates:
    for layers in layers_configs:
        print(f"\nEntrenando red con LR={lr}, capas={layers}")
        nn = FeedforwardNN(layers)
        history = nn.train(X_train, y_train_oh, epochs=50, batch_size=128, learning_rate=lr, X_val=X_val, y_val=y_val_oh)
        val_acc = history["val_acc"][-1]
        test_acc = nn.accuracy(X_test, y_test_oh)
        print(f"Test Accuracy: {test_acc:.4f}")
        results.append({
            "learning_rate": lr,
            "layers": layers,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "history": history,
            "model": nn
        })

# Select the best model by accuracy in validation.
best_model = max(results, key=lambda r: r["val_acc"])
print("\nBest model:")
print(f"Layers: {best_model['layers']}")
print(f"Learning Rate: {best_model['learning_rate']}")
print(f"Accuracy en validación: {best_model['val_acc']:.4f}")
print(f"Accuracy en test: {best_model['test_acc']:.4f}")

# Plot the training curve of the best model.
plt.plot(best_model["history"]["loss"], label="Train Loss")
plt.plot(best_model["history"]["val_loss"], label="Validation Loss")
plt.title("Best model loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


