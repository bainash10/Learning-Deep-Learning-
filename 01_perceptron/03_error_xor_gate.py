import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=50):
        """Initialize perceptron"""
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # +1 for bias

    def activation(self, x):
        """Step function"""
        return np.where(x >= 0, 1, 0)

    def predict(self, x):
        """Predict output for one sample"""
        x_with_bias = np.insert(x, 0, 1)
        z = np.dot(x_with_bias, self.weights)
        return self.activation(z)

    def fit(self, X, y):
        """Train perceptron and return error history"""
        error_history = []

        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                x_with_bias = np.insert(xi, 0, 1)
                y_pred = self.activation(np.dot(x_with_bias, self.weights))
                # Update weights
                self.weights += self.lr * (target - y_pred) * x_with_bias
                # Compute squared error
                total_error += (target - y_pred) ** 2
            error_history.append(total_error)
            print(f"Epoch {epoch+1}: Total Error = {total_error}")
        
        return error_history

# XOR data
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Train perceptron
perceptron_xor = Perceptron(input_size=2, learning_rate=0.1, epochs=50)
error_history = perceptron_xor.fit(X_xor, y_xor)

# Test predictions
print("\nXOR Gate Predictions:")
for xi in X_xor:
    print(f"{xi} -> {perceptron_xor.predict(xi)}")

# Plot error over epochs
plt.plot(range(1, 51), error_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Total Squared Error")
plt.title("Perceptron Learning Error for XOR Gate")
plt.grid(True)
plt.show()
