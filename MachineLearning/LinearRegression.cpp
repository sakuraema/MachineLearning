#include "LinearRegression.h"
#include <iostream>
#include <cmath>
#include <limits>

LinearRegression::LinearRegression(int nFeatures, double dLearningRate, double dTolerance, int iMaxIterations)
    : m_dWeights(nFeatures, 0.0), m_dBias(0), m_dLearningRate(dLearningRate), m_dTolerance(dTolerance), m_iMaxIterations(iMaxIterations), m_nFeatures(nFeatures)
{
}

// Hypothesis: h(x) = w1*x1 + w2*x2 + ... + wn*xn + b
double LinearRegression::ComputeHypothesis(const std::vector<double>& x) const
{
    double dPredict = 0.0;
    for (int j = 0; j < m_nFeatures; ++j)
        dPredict += m_dWeights[j] * x[j];
    return dPredict + m_dBias;
}

// Mean Squared Error loss: L(w, b) = (1/2n) * £U(h(xi) - yi)^2
double LinearRegression::ComputeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const
{
    double dTotalLoss = 0.0;
    const size_t n = X.size();

    for (size_t i = 0; i < n; ++i)
    {
        double dHypothesis = ComputeHypothesis(X[i]);
        double dLoss = dHypothesis - y[i];
        dTotalLoss += dLoss * dLoss;
    }
    return dTotalLoss / (2 * n);
}

// Griadient: dL/dw = (h(x) - y) * x
void LinearRegression::ComputeGradients(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::vector<double>& dDeltaWeights, double& dDeltaBias) const
{
    const size_t n = X.size();

    for (size_t i = 0; i < n; ++i)
    {
        double dError = ComputeHypothesis(X[i]) - y[i];
        for (int j = 0; j < m_nFeatures; ++j)
        {
            dDeltaWeights[j] += dError * X[i][j];
        }
        dDeltaBias += dError;
    }

    for (int j = 0; j < m_nFeatures; ++j)
		dDeltaWeights[j] /= n; // Normalize by number of samples
    dDeltaBias /= n;
}

void LinearRegression::Fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    if (X.empty() || y.empty())
    {
        std::cerr << "Error: The input data X and y must not be empty.\n";
        return;
    }

    if (X.size() != y.size())
    {
        std::cerr << "Error: The size of X and y must be the same.\n";
        return;
    }

    if (m_nFeatures != X[0].size())
    {
        std::cerr << "Error: The number of features in X does not match the expected number.\n";
        return;
    }

    double dPreviousLoss = std::numeric_limits<double>::infinity();

    for (int i = 0; i < m_iMaxIterations; ++i)
    {
        std::vector<double> dDeltaWeights(m_nFeatures, 0.0);
        double dDeltaBias = 0.0;
        ComputeGradients(X, y, dDeltaWeights, dDeltaBias);

        // Update parameters
        for (int j = 0; j < m_nFeatures; ++j)
            m_dWeights[j] -= m_dLearningRate * dDeltaWeights[j];
        m_dBias -= m_dLearningRate * dDeltaBias;

        // Check convergence
        double dCurrentLoss = ComputeLoss(X, y);
        if (fabs(dPreviousLoss - dCurrentLoss) < m_dTolerance)
        {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
        dPreviousLoss = dCurrentLoss;

		if (i == m_iMaxIterations - 1)
			std::cout << "Reached maximum iterations without convergence.\n";
    }
}
