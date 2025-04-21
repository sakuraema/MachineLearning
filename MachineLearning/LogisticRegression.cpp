#include "pch.h"
#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(double dLearningRate, int iMaxIterations)
    : m_dBias(0), m_dLearningRate(dLearningRate), m_iMaxIterations(iMaxIterations), m_nFeatures(0), m_nClasses(0)
{
}

void LogisticRegression::Train(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    m_nFeatures = X[0].size();
    m_dWeights.assign(m_nFeatures, 0.0);
    m_nClasses = *std::max_element(y.begin(), y.end()) + 1; // Determine the number of classes

    Eigen::VectorXd dDeltaWeights = Eigen::VectorXd::Zero(m_nFeatures);
    double dDeltaBias = 0.0;

    Eigen::MatrixXd X_eigen(X.size(), m_nFeatures);
    for (size_t i = 0; i < X.size(); ++i)
    {
        for (size_t j = 0; j < m_nFeatures; ++j)
            X_eigen(i, j) = X[i][j];
    }
    const Eigen::VectorXd y_eigen = Eigen::Map<const Eigen::VectorXd>(y.data(), y.size());

    for (int iter = 0; iter < m_iMaxIterations; ++iter)
    {
        ComputeGradients(X_eigen, y_eigen, dDeltaWeights, dDeltaBias);

        // Update weights and bias
        Eigen::VectorXd dWeights_eigen = Eigen::Map<Eigen::VectorXd>(m_dWeights.data(), m_dWeights.size());
        dWeights_eigen -= m_dLearningRate * dDeltaWeights;
        m_dBias -= m_dLearningRate * dDeltaBias;

        // Copy back to std::vector
        Eigen::VectorXd::Map(&m_dWeights[0], m_nFeatures) = dWeights_eigen;
    }
}

double LogisticRegression::PredictProbability(const std::vector<double>& x) const
{
    Eigen::VectorXd x_eigen = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
    Eigen::VectorXd dWeights_eigen = Eigen::Map<const Eigen::VectorXd>(m_dWeights.data(), m_dWeights.size());

    double z = m_dBias + dWeights_eigen.dot(x_eigen);
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

void LogisticRegression::ComputeGradients(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, Eigen::VectorXd& dDeltaWeights, double& dDeltaBias) const
{
    const size_t n = X.rows();

    if (m_nClasses > 2) // Multiclass classification
    {
        return;
#if 0
        Eigen::MatrixXd logits = X * Eigen::Map<const Eigen::VectorXd>(m_dWeights.data(), m_dWeights.size()).replicate(1, m_nClasses);
        logits.rowwise() += Eigen::VectorXd::Constant(m_nClasses, m_dBias).transpose();

        Eigen::MatrixXd probabilities = logits.unaryExpr([this](double z) { return ComputeSigmoid(z); });

        Eigen::MatrixXd errors = probabilities;
        for (size_t i = 0; i < n; ++i)
            errors.row(i) -= Eigen::VectorXd::Unit(m_nClasses, y[i]);

        dDeltaWeights = X.transpose() * errors.colwise().sum();
        dDeltaBias = errors.sum();
#endif
    }
    else // Binary classification
    {
        Eigen::VectorXd predictions = (X * Eigen::Map<const Eigen::VectorXd>(m_dWeights.data(), m_dWeights.size()) + Eigen::VectorXd::Constant(X.rows(), m_dBias)).unaryExpr([this](double z) { return ComputeSigmoid(z); });
        Eigen::VectorXd errors = predictions - y;

        dDeltaWeights = X.transpose() * errors;
        dDeltaBias = errors.sum();
    }

    // Normalize gradients
    dDeltaWeights /= n;
    dDeltaBias /= n;
}
