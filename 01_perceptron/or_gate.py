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
        x_with_bias = np.insert(x, 0, 1)
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
        """Plot 2D decision boundary"""
        for xi, label in zip(X, y):
            if label == 0:
                plt.scatter(xi[0], xi[1], color='red', marker='o',
                            label='0' if '0' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(xi[0], xi[1], color='blue', marker='x',
                            label='1' if '1' not in plt.gca().get_legend_handles_labels()[1] else "")

        x_vals = np.linspace(-0.5, 1.5, 100)
        if self.weights[2] != 0:
            y_vals = -(self.weights[0] + self.weights[1] * x_vals) / self.weights[2]
            plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Decision Boundary of OR Gate")
        plt.legend()
        plt.grid(True)
        plt.show()


# ---- OR Gate ----
X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y_or = np.array([0, 1, 1, 1])  # OR gate outputs

# Train perceptron
perceptron_or = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron_or.fit(X_or, y_or)

# Test perceptron
print("\nTesting OR Gate:")
for xi in X_or:
    print(f"{xi} -> {perceptron_or.predict(xi)}")

# Plot decision boundary
perceptron_or.plot_decision_boundary(X_or, y_or)
