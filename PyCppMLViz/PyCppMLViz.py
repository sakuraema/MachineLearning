import ml_cpp
import numpy as np
import matplotlib.pyplot as plt

# Initialize C++ model
model = ml_cpp.LinearRegression()

# Train with numpy data
X_train = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.double)
y_train = np.array([5, 8, 11], dtype=np.double)
model.train(ml_cpp.AlgorithmType.BatchGradientDescent, 2, X_train.tolist(), y_train.tolist())  # Convert to C++ types
weights = model.get_weights()
bias = model.get_bias()

# Print weights and bias
print(f"Weights: {weights}")
print(f"Bias: {bias}")

# Predict and plot
X_test = np.array([[4, 5], [5, 6]], dtype=np.double)
predictions = model.predict(X_test.tolist())
print(f"Predictions: {predictions}\n")
plt.plot(predictions, label="BatchGradientDescent")
plt.legend()

model.clear()
model.train(ml_cpp.AlgorithmType.StochasticGradientDescent, 2, X_train.tolist(), y_train.tolist())
weights = model.get_weights()
bias = model.get_bias()
print(f"Weights: {weights}")
print(f"Bias: {bias}")
predictions = model.predict(X_test.tolist())
print(f"Predictions: {predictions}\n")
plt.plot(predictions, label="StochasticGradientDescent")
plt.legend()

model.clear()
model.train(ml_cpp.AlgorithmType.MinibatchGradientDescent, 2, X_train.tolist(), y_train.tolist())
weights = model.get_weights()
bias = model.get_bias()
print(f"Weights: {weights}")
print(f"Bias: {bias}")
predictions = model.predict(X_test.tolist())
print(f"Predictions: {predictions}\n")
plt.plot(predictions, label="MinibatchGradientDescent")
plt.legend()

model.clear()
model.train(ml_cpp.AlgorithmType.NormalEquation, 2, X_train.tolist(), y_train.tolist())
weights = model.get_weights()
bias = model.get_bias()
print(f"Weights: {weights}")
print(f"Bias: {bias}")
predictions = model.predict(X_test.tolist())
print(f"Predictions: {predictions}\n")
plt.plot(predictions, label="NormalEquation")
plt.legend()

plt.title("Linear Regression")

plt.gcf().canvas.manager.set_window_title("Training Result")

plt.show()
