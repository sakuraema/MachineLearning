#include "pch.h"
#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(double dLearningRate, int iMaxIterations)
    : m_dBias(0), m_dLearningRate(dLearningRate), m_iMaxIterations(iMaxIterations), m_nFeatures(0)
{
}

void LogisticRegression::Train(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{
    m_nFeatures = X[0].size();
    m_dWeights.assign(m_nFeatures, 0.0);

    std::vector<double> dDeltaWeights(m_nFeatures, 0.0);
    double dDeltaBias = 0.0;

    for (int iter = 0; iter < m_iMaxIterations; ++iter)
    {
        ComputeGradients(X, y, dDeltaWeights, dDeltaBias);

        // Update weights and bias
        for (int j = 0; j < m_nFeatures; ++j)
            m_dWeights[j] -= m_dLearningRate * dDeltaWeights[j];
        m_dBias -= m_dLearningRate * dDeltaBias;
    }
}

double LogisticRegression::PredictProbability(const std::vector<double>& x) const
{
    double z = m_dWeights[0];  // Bias term
    for (size_t i = 0; i < x.size(); ++i)
        z += m_dWeights[i + 1] * x[i];

    return ComputeSigmoid(z);
}

int LogisticRegression::Predict(const std::vector<double>& x) const
{
    return PredictProbability(x) >= 0.5 ? 1 : 0;
}

double LogisticRegression::ComputeSigmoid(double z) const
{
    return 1.0 / (1.0 + exp(-z));
}

std::vector<double> LogisticRegression::ComputeSoftmax(const std::vector<double>& z) const
{
	std::vector<double> softmax(z.size());
	double dMaxZ = *std::max_element(z.begin(), z.end());
	double dSumExp = 0.0;
	for (size_t i = 0; i < z.size(); ++i)
	{
		softmax[i] = exp(z[i] - dMaxZ);
		dSumExp += softmax[i];
	}
	for (size_t i = 0; i < z.size(); ++i)
		softmax[i] /= dSumExp;

	return softmax;
}

void LogisticRegression::ComputeGradients(const std::vector<std::vector<double>>& X, const std::vector<int>& y, std::vector<double>& dDeltaWeights, double& dDeltaBias) const
{
    const size_t n = X.size();

    for (size_t i = 0; i < n; ++i) {
        // Modified to use sigmoid activation
        double dPrediction = ComputeSigmoid(Eigen::Map<const Eigen::VectorXd>(X[i].data(), X[i].size()).dot(Eigen::Map<const Eigen::VectorXd>(m_dWeights.data(), m_dWeights.size())) + m_dBias);
        double dError = dPrediction - y[i];

        for (int j = 0; j < m_nFeatures; ++j)
            dDeltaWeights[j] += dError * X[i][j];
        dDeltaBias += dError;
    }

    const double dInvSampleCount = 1.0 / n;
    for (int j = 0; j < m_nFeatures; ++j)
        dDeltaWeights[j] *= dInvSampleCount;
    dDeltaBias *= dInvSampleCount;
}
