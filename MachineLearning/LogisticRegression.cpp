#include "pch.h"
#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(double lr, int n_iter)
    : m_dLearningRate(lr), m_iMaxIterations(n_iter)
{
}

// Train the model using gradient descent
void LogisticRegression::Train(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{
    // Add bias term and initialize weights
    auto X_bias = AddBias(X);
    m_dWeights = std::vector<double>(X_bias[0].size(), 0.0);

    // Gradient descent
    for (int iter = 0; iter < m_iMaxIterations; ++iter)
    {
        std::vector<double> gradients(m_dWeights.size(), 0.0);

        // Calculate gradients
        for (size_t i = 0; i < X_bias.size(); ++i)
        {
            double predicted = PredictProbability(X[i]);
            double error = predicted - y[i];

            for (size_t j = 0; j < m_dWeights.size(); ++j)
            {
                gradients[j] += error * X_bias[i][j];
            }
        }

        // Update weights
        for (size_t j = 0; j < m_dWeights.size(); ++j)
        {
            m_dWeights[j] -= m_dLearningRate * gradients[j] / X.size();
        }
    }
}

// Predict probability for single sample
double LogisticRegression::PredictProbability(const std::vector<double>& x) const
{
    double z = m_dWeights[0];  // Bias term
    for (size_t i = 0; i < x.size(); ++i)
    {
        z += m_dWeights[i + 1] * x[i];
    }
    return ComputeSigmoid(z);
}

// Predict class label (0 or 1)
int LogisticRegression::Predict(const std::vector<double>& x) const
{
    return PredictProbability(x) >= 0.5 ? 1 : 0;
}

// Sigmoid activation function
double LogisticRegression::ComputeSigmoid(double z) const
{
    return 1.0 / (1.0 + exp(-z));
}

std::vector<double> LogisticRegression::ComputeSoftmax(const std::vector<double>& z) const
{
	std::vector<double> softmax(z.size());
	double max_z = *std::max_element(z.begin(), z.end());
	double sum_exp = 0.0;
	for (size_t i = 0; i < z.size(); ++i)
	{
		softmax[i] = exp(z[i] - max_z);
		sum_exp += softmax[i];
	}
	for (size_t i = 0; i < z.size(); ++i)
	{
		softmax[i] /= sum_exp;
	}
	return softmax;
}

// Add bias term (column of 1s) to feature matrix
std::vector<std::vector<double>> LogisticRegression::AddBias(const std::vector<std::vector<double>>& X) const
{
    std::vector<std::vector<double>> X_bias;
    for (const auto& row : X)
    {
        std::vector<double> new_row = { 1.0 };  // Bias term
        new_row.insert(new_row.end(), row.begin(), row.end());
        X_bias.push_back(new_row);
    }
    return X_bias;
}
