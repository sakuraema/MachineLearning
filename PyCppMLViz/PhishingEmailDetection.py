import ml_cpp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting module

# Load the dataset
data = pd.read_csv(r"DataSet\email_phishing_data.csv")

# Extract features and target
features = data[["num_words", "num_unique_words"]].values  # Example features
labels = data["label"].values  # Target (1 for phishing, 0 for not phishing)

# Split the data into training and test sets
train_features, test_features = features[-300:], features[-600:-300]  # Last 300 rows for training, 300 before last for testing
train_labels, test_labels = labels[-300:], labels[-600:-300]

# Initialize C++ model
model = ml_cpp.LogisticRegression(1e-7, 100000)

# Train the model for all algorithms and store results
model.clear()
model.train(train_features.shape[1], train_features.tolist(), train_labels.tolist())
weights = model.get_weights()
bias = model.get_bias()
test_predictions = model.predict(test_features.tolist())  # Predict for test_features

# 3D Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')  # Create a single 3D plot

# Plot the training data
ax.scatter(train_features[:, 0], train_features[:, 1], train_labels, color='blue', label='Training Data')

# Plot the test data predictions
ax.scatter(test_features[:, 0], test_features[:, 1], test_predictions, color='green', label='Test Predictions')

# Create a grid for the plane
x1_min = min(np.min(train_features[:, 0]), np.min(test_features[:, 0]))
x1_max = max(np.max(train_features[:, 0]), np.max(test_features[:, 0]))
x1 = np.linspace(x1_min, x1_max, 10)

# Calculate x2 and z (probability = 0.5) for the decision boundary
x1_grid, x2_grid = np.meshgrid(x1, x1)  # Create a grid for x1 and x2
z_grid = -(weights[0] * x1_grid + weights[1] * x2_grid + bias) / weights[1]  # Decision boundary

# Plot the separation plane
ax.plot_surface(x1_grid, x2_grid, z_grid, alpha=0.5, color='red', label='Decision Boundary')

ax.set_title("3D Separation Plane for Logistic Regression")
ax.set_xlabel("Number of Words (Feature 1)")
ax.set_ylabel("Number of Unique Words (Feature 2)")
ax.set_zlabel("Phishing Probability (Target)")
ax.legend()

# Show the plot
plt.tight_layout()
plt.gcf().canvas.manager.set_window_title("3D Separation Plane for Logistic Regression")
plt.show()
