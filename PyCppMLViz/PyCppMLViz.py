import ml_cpp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting module

# Initialize C++ model
model = ml_cpp.LinearRegression()

# Train with numpy data
X_train = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.double)
y_train = np.array([5, 8, 11], dtype=np.double)

# Test data
X_test = np.array([[4, 5], [5, 6]], dtype=np.double)

# Train the model for all algorithms and store results
results = {}
predictions = {}
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
    y_test_pred = model.predict(X_test.tolist())  # Predict for X_test
    results[algo] = (weights, bias)  # Store weights and bias
    predictions[algo] = y_test_pred  # Store predictions for X_test

# 3D Plotting
fig = plt.figure(figsize=(12, 10))
for i, (algo, (weights, bias)) in enumerate(results.items(), start=1):
    ax = fig.add_subplot(2, 2, i, projection='3d')  # Create a 3D subplot

    # Plot the training data
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='blue', label='Training Data')

    # Plot the test data predictions
    y_test_pred = predictions[algo]
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test_pred, color='green', label='Test Predictions')

    # Create a grid for the plane
    x1_min = min(np.min(X_train[:, 0]), np.min(X_test[:, 0]))
    x1_max = max(np.max(X_train[:, 0]), np.max(X_test[:, 0]))
    x2_min = min(np.min(X_train[:, 1]), np.min(X_test[:, 1]))
    x2_max = max(np.max(X_train[:, 1]), np.max(X_test[:, 1]))
    x1 = np.linspace(x1_min, x1_max, 10)
    x2 = np.linspace(x2_min, x2_max, 10)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    y_grid = weights[0] * x1_grid + weights[1] * x2_grid + bias  # Hypothesis plane

    # Plot the hypothesis plane
    ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color='red', label='Hypothesis Plane')

    ax.set_title(f"Hypothesis and Predictions ({str(algo).split('.')[-1]})")
    ax.set_xlabel("Feature 1 (x1)")
    ax.set_ylabel("Feature 2 (x2)")
    ax.set_zlabel("Target (y)")
    ax.legend()

# Show the plots
plt.tight_layout()
plt.gcf().canvas.manager.set_window_title("3D Hypothesis and Predictions for Each Algorithm")
plt.show()
