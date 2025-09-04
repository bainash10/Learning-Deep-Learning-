import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        """
        Initialize the perceptron model.
        
        Parameters:
        input_size (int): Number of input features
        learning_rate (float): Step size for updating weights
        epochs (int): Number of times to iterate over the training data
        """
        self.lr = learning_rate
        self.epochs = epochs
        # Initialize weights (including bias as the first weight) to zeros
        # Shape = input_size + 1 (extra 1 for bias)
        self.weights = np.zeros(input_size + 1)  

    def activation(self, x):
        """
        Step activation function.
        Returns 1 if x >= 0, else 0.
        """
        return np.where(x >= 0, 1, 0)

    def predict(self, x):
        """
        Predict the output for a single input sample.
        
        Parameters:
        x (array): Input feature vector (without bias)
        
        Returns:
        int: Predicted class (0 or 1)
        """
        # Add bias term (1) at the start of the input vector
        x_with_bias = np.insert(x, 0, 1)
        # Compute linear combination of inputs and weights
        z = np.dot(x_with_bias, self.weights)
        # Apply activation function
        return self.activation(z)

    def fit(self, X, y):
        """
        Train the perceptron using the perceptron learning rule.
        
        Parameters:
        X (2D array): Training input samples
        y (1D array): Training output labels
        """
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            # Loop over each training sample
            for xi, target in zip(X, y):
                # Add bias term to the input
                x_with_bias = np.insert(xi, 0, 1)
                # Compute linear combination
                z = np.dot(x_with_bias, self.weights)
                # Apply activation function
                y_pred = self.activation(z)

                # Print input, target, and prediction
                print(f"\n Input: {xi}, Target: {target}, Pred: {y_pred}")

                # Update weights according to perceptron learning rule:
                # wi = wi + learning_rate * (target - prediction) * xi
                for i in range(len(self.weights)):
                    old_w = self.weights[i]  # store old weight for printing
                    self.weights[i] = old_w + self.lr * (target - y_pred) * x_with_bias[i]
                    print(f"  w{i}: {old_w:.3f} -> {self.weights[i]:.3f} "
                          f"(update = {self.lr}*({target}-{y_pred})*{x_with_bias[i]})")

    def plot_decision_boundary(self, X, y):
        """
        Plot the decision boundary learned by the perceptron for 2D data.
        
        Parameters:
        X (2D array): Input samples
        y (1D array): Corresponding labels
        """
        # Plot training points
        for xi, label in zip(X, y):
            if label == 0:
                # Red circle for class 0
                plt.scatter(xi[0], xi[1], color='red', marker='o',
                            label='0' if '0' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                # Blue X for class 1
                plt.scatter(xi[0], xi[1], color='blue', marker='x',
                            label='1' if '1' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # Calculate decision boundary: w0 + w1*x1 + w2*x2 = 0 -> x2 = -(w0 + w1*x1)/w2
        x_vals = np.linspace(-0.5, 1.5, 100)
        if self.weights[2] != 0:  # Avoid division by zero
            y_vals = -(self.weights[0] + self.weights[1] * x_vals) / self.weights[2]
            plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')

        # Labels and grid
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Decision Boundary of Perceptron")
        plt.legend()
        plt.grid(True)
        plt.show()


# ---- Example: AND gate ----
# Input samples for AND logic
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output labels for AND logic
y = np.array([0, 0, 0, 1])

# Create perceptron instance
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)

# Train the perceptron
perceptron.fit(X, y)

# Test perceptron predictions
print("\nTesting AND gate:")
for xi in X:
    print(f"{xi} -> {perceptron.predict(xi)}")

# Plot the final decision boundary
perceptron.plot_decision_boundary(X, y)
