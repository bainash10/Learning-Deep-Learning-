import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        """Initialize perceptron"""
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # +1 for bias

    def activation(self, x):
        """Step function"""
        return np.where(x >= 0, 1, 0)

    def predict(self, x):
        """Predict output for one sample"""
        x_with_bias = np.insert(x, 0, 1)  # add bias
        z = np.dot(x_with_bias, self.weights)
        return self.activation(z)

    def fit(self, X, y):
        """Train perceptron"""
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            for xi, target in zip(X, y):
                x_with_bias = np.insert(xi, 0, 1)
                z = np.dot(x_with_bias, self.weights)
                y_pred = self.activation(z)

                print(f"\n Input: {xi}, Target: {target}, Pred: {y_pred}")

                # Update weights
                for i in range(len(self.weights)):
                    old_w = self.weights[i]
                    self.weights[i] += self.lr * (target - y_pred) * x_with_bias[i]
                    print(f"  w{i}: {old_w:.3f} -> {self.weights[i]:.3f} "
                          f"(update = {self.lr}*({target}-{y_pred})*{x_with_bias[i]})")

    def plot_decision_boundary(self, X, y):
        """Plot decision boundary for single input"""
        for xi, label in zip(X, y):
            if label == 0:
                plt.scatter(xi[0], 0, color='red', marker='o', label='0' if '0' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(xi[0], 0, color='blue', marker='x', label='1' if '1' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Decision boundary: w0 + w1*x = 0 -> x = -w0/w1
        if self.weights[1] != 0:
            x_boundary = -self.weights[0] / self.weights[1]
            plt.axvline(x=x_boundary, color='k', linestyle='-', label='Decision boundary')

        plt.xlabel("x")
        plt.title("Decision Boundary of NOT Gate")
        plt.yticks([])  # Hide y-axis as it's single input
        plt.legend()
        plt.grid(True)
        plt.show()


# ---- NOT Gate ----
X_not = np.array([[0],
                  [1]])
y_not = np.array([1, 0])  # NOT gate outputs

# Train perceptron
perceptron_not = Perceptron(input_size=1, learning_rate=0.1, epochs=10)
perceptron_not.fit(X_not, y_not)

# Test perceptron
print("\nTesting NOT Gate:")
for xi in X_not:
    print(f"{xi} -> {perceptron_not.predict(xi)}")

# Plot decision boundary
perceptron_not.plot_decision_boundary(X_not, y_not)
