#include "pch.h"

#include "LinearRegression.h"

int main()
{
    // Sample dataset: y = 2*x1 + 3*x2 + 1
    std::vector<std::vector<double>> X = {
        {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}
    };
    std::vector<double> y = { 9, 14, 19, 24, 29 }; // 2*1 + 3*2 + 1 = 9, etc.

    // Create and train model
    LinearRegression model(2, 0.01, 1e-6, 10000);
    model.Train(AlgorithmType::BatchGradientDescent, X, y);
    // Output results
    std::cout << "Final parameters:\n";
	for (size_t i = 0; i < model.GetWeights().size(); ++i)
		std::cout << "Weight " << i + 1 << ": " << model.GetWeights()[i] << "\n";
    std::cout << "Bias: " << model.GetBias() << "\n\n";

    model.Clear();
    model.Train(AlgorithmType::StochasticGradientDescent, X, y);
    // Output results
    std::cout << "Final parameters:\n";
    for (size_t i = 0; i < model.GetWeights().size(); ++i)
        std::cout << "Weight " << i + 1 << ": " << model.GetWeights()[i] << "\n";
    std::cout << "Bias: " << model.GetBias() << "\n\n";

    model.Clear();
	// Sample dataset is too small causing batch size to be 1, which is same as SGD
    model.Train(AlgorithmType::MinibatchGradientDescent, X, y);
    // Output results
    std::cout << "Final parameters:\n";
    for (size_t i = 0; i < model.GetWeights().size(); ++i)
        std::cout << "Weight " << i + 1 << ": " << model.GetWeights()[i] << "\n";
    std::cout << "Bias: " << model.GetBias() << "\n\n";

    model.Clear();
	// Data has linear dependency, causing X^T*X to be singular
	// Implement Ridge Regression or Pseudoinverse later
	model.Train(AlgorithmType::NormalEquation, X, y);
    // Output results
    std::cout << "Final parameters:\n";
    for (size_t i = 0; i < model.GetWeights().size(); ++i)
        std::cout << "Weight " << i + 1 << ": " << model.GetWeights()[i] << "\n";
    std::cout << "Bias: " << model.GetBias() << "\n\n";

    std::cin.get();

    return 0;
}
