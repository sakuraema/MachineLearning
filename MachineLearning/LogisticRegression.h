#pragma once

#include "Algorithms.h"

class LogisticRegression
{
public:
    LogisticRegression(double lr = 0.01, int n_iter = 1000);

    // Train the model using gradient descent
    void Train(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    // Predict probability for single sample
    double PredictProbability(const std::vector<double>& x) const;

    // Predict class label (0 or 1)
    int Predict(const std::vector<double>& x) const;

    inline std::vector<double> GetWeights() const { return m_dWeights; }

private:
    std::vector<double> m_dWeights;  // Includes bias term as first element
    double m_dLearningRate;
    int m_iMaxIterations;

    // Sigmoid activation function
    double ComputeSigmoid(double z) const;
    
    std::vector<double> ComputeSoftmax(const std::vector<double>& z) const;

    // Add bias term (column of 1s) to feature matrix
    std::vector<std::vector<double>> AddBias(const std::vector<std::vector<double>>& X) const;
};
