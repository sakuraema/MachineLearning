#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

#include "LinearRegression.h"

int main()
{
    // Sample dataset: y = 2*x1 + 3*x2 + 1
    std::vector<std::vector<double>> X = {
        {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}
    };
    std::vector<double> y = { 9, 14, 19, 24, 29 }; // 2*1 + 3*2 + 1 = 9, etc.

    // Create and train model
    LinearRegression model(2, 0.01, 1e-8, 3000);
    model.Train(AlgorithmType::BatchGradientDescent, X, y);
    // Output results
    std::cout << "Final parameters:\n";
	for (size_t i = 0; i < model.GetWeights().size(); ++i)
		std::cout << "Weight " << i + 1 << ": " << model.GetWeights()[i] << "\n";
    std::cout << "Bias: " << model.GetBias() << "\n";

    model.Clear();
	// Data has linear dependency, causing X^T*X to be singular
	// Implement Ridge Regression or Pseudoinverse later
	model.Train(AlgorithmType::NormalEquation, X, y);
    // Output results
    std::cout << "Final parameters:\n";
    for (size_t i = 0; i < model.GetWeights().size(); ++i)
        std::cout << "Weight " << i + 1 << ": " << model.GetWeights()[i] << "\n";
    std::cout << "Bias: " << model.GetBias() << "\n";

    std::cin.get();

    return 0;
}
