import ml_cpp
import numpy as np
import matplotlib.pyplot as plt

# Initialize C++ model
model = ml_cpp.LinearRegression()

# Train with numpy data
X_train = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.double)
y_train = np.array([5, 8, 11], dtype=np.double)

# Train the model for all algorithms and store results
results = {}
for algo in [
    ml_cpp.AlgorithmType.BatchGradientDescent,
    ml_cpp.AlgorithmType.StochasticGradientDescent,
    ml_cpp.AlgorithmType.MinibatchGradientDescent,
    ml_cpp.AlgorithmType.NormalEquation
]:
    model.clear()
    model.train(algo, 2, X_train.tolist(), y_train.tolist())
    weights = model.get_weights()
    bias = model.get_bias()
    results[algo] = (weights, bias)  # Store weights and bias for each algorithm

# Plot the training data for Feature 1
X_train_feature1 = X_train[:, 0]  # First feature
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # Create a subplot for Feature 1
plt.scatter(X_train_feature1, y_train, color='blue', label='Training Data (Feature 1)')

# Plot the best-fitting function for Feature 1
X_line_feature1 = np.linspace(min(X_train_feature1), max(X_train_feature1), 100)
for algo, (weights, bias) in results.items():
    y_line = weights[0] * X_line_feature1 + bias  # Use the first feature's weight
    plt.plot(X_line_feature1, y_line, label=str(algo).split('.')[-1])

plt.title("Linear Regression (Feature 1)")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()

# Plot the training data for Feature 2
X_train_feature2 = X_train[:, 1]  # Second feature
plt.subplot(1, 2, 2)  # Create a subplot for Feature 2
plt.scatter(X_train_feature2, y_train, color='blue', label='Training Data (Feature 2)')

# Plot the best-fitting function for Feature 2
X_line_feature2 = np.linspace(min(X_train_feature2), max(X_train_feature2), 100)
for algo, (weights, bias) in results.items():
    y_line = weights[1] * X_line_feature2 + bias  # Use the second feature's weight
    plt.plot(X_line_feature2, y_line, label=str(algo).split('.')[-1])

plt.title("Linear Regression (Feature 2)")
plt.xlabel("Feature 2")
plt.ylabel("Target")
plt.legend()

# Show the plots
plt.tight_layout()
plt.gcf().canvas.manager.set_window_title("Training Result (Feature 1 and Feature 2)")
plt.show()
