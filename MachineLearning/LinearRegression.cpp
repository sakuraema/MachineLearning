#include "pch.h"

#include "LinearRegression.h"

LinearRegression::LinearRegression(double dLearningRate, double dTolerance, int iMaxIterations)
	: m_dBias(0), m_dLearningRate(dLearningRate), m_dTolerance(dTolerance), m_iMaxIterations(iMaxIterations), m_nFeatures(0)
{
}

void LinearRegression::Train(AlgorithmType eType, int nFeatures, const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
	m_nFeatures = nFeatures;

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

	m_dWeights.assign(m_nFeatures, 0.0);

    switch (eType)
    {
    case AlgorithmType::BatchGradientDescent:
        BatchGradientDescent(X, y);
	    std::cout << "Batch Gradient Descent completed.\n";
        break;
    case AlgorithmType::StochasticGradientDescent:
        StochasticGradientDescent(X, y);
	    std::cout << "Stochastic Gradient Descent completed.\n";
        break;
    case AlgorithmType::MinibatchGradientDescent:
    {
        int nBatchSize = std::max(1, static_cast<int>(X.size() / 10)); // Set batch size to 10% of the sample size, minimum 1
        MinibatchGradientDescent(X, y, nBatchSize);
        std::cout << "Minibatch Gradient Descent completed with batch size: " << nBatchSize << "\n";
        break;
    }
    case AlgorithmType::NormalEquation:
        NormalEquation(X, y);
	    std::cout << "Normal Equation completed.\n";
        break;
    default:
        break;
    }
}

std::vector<double> LinearRegression::Predict(const std::vector<std::vector<double>>& X) const
{
	std::vector<double> yPred(X.size(), 0.0);
	for (size_t i = 0; i < X.size(); ++i)
		yPred[i] = ComputeHypothesis(X[i]);
	return yPred;
}

double LinearRegression::ComputeHypothesis(const std::vector<double>& x) const
{
    double dPredict = 0.0;
    // Hypothesis: h(x) = w1*x1 + w2*x2 + ... + wn*xn + b
    for (int j = 0; j < m_nFeatures; ++j)
        dPredict += m_dWeights[j] * x[j];
    return dPredict + m_dBias;
}

double LinearRegression::ComputeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const
{
    double dTotalLoss = 0.0;
    const size_t n = X.size();
    // Mean Squared Error loss: L(w, b) = (1/2n) * Σ(h(xi) - yi)^2
    for (size_t i = 0; i < n; ++i)
    {
        double dHypothesis = ComputeHypothesis(X[i]);
        double dLoss = dHypothesis - y[i];
        dTotalLoss += dLoss * dLoss;
    }
    return dTotalLoss / (2 * n);
}

void LinearRegression::ComputeGradients(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::vector<double>& dDeltaWeights, double& dDeltaBias) const
{
    const size_t n = X.size();
	// Griadient of the loss function: ∂L/∂w = (1/n) * Σ(h(xi) - yi) * xi
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

void LinearRegression::BatchGradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
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

void LinearRegression::StochasticGradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    const size_t n = X.size();
    double dPreviousLoss = std::numeric_limits<double>::infinity();

    for (int i = 0; i < m_iMaxIterations; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            // Compute gradients for a single sample
            std::vector<double> dDeltaWeights(m_nFeatures, 0.0);
            double dDeltaBias = 0.0;

            double dError = ComputeHypothesis(X[j]) - y[j];
            for (int k = 0; k < m_nFeatures; ++k)
            {
                dDeltaWeights[k] = dError * X[j][k];
            }
            dDeltaBias = dError;

            // Update parameters
            for (int k = 0; k < m_nFeatures; ++k)
            {
                m_dWeights[k] -= m_dLearningRate * dDeltaWeights[k];
            }
            m_dBias -= m_dLearningRate * dDeltaBias;
        }

        // Check convergence after each epoch
        double dCurrentLoss = ComputeLoss(X, y);
        if (fabs(dPreviousLoss - dCurrentLoss) < m_dTolerance)
        {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
        dPreviousLoss = dCurrentLoss;

        if (i == m_iMaxIterations - 1)
        {
            std::cout << "Reached maximum iterations without convergence.\n";
        }
    }
}

void LinearRegression::MinibatchGradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int batchSize)
{
    const size_t n = X.size();
    double dPreviousLoss = std::numeric_limits<double>::infinity();

    for (int i = 0; i < m_iMaxIterations; ++i)
    {
        for (size_t start = 0; start < n; start += batchSize)
        {
            size_t end = std::min(start + batchSize, n);
            std::vector<double> dDeltaWeights(m_nFeatures, 0.0);
            double dDeltaBias = 0.0;

            // Compute gradients for the minibatch
            for (size_t j = start; j < end; ++j)
            {
                double dError = ComputeHypothesis(X[j]) - y[j];
                for (int k = 0; k < m_nFeatures; ++k)
                {
                    dDeltaWeights[k] += dError * X[j][k];
                }
                dDeltaBias += dError;
            }

            // Normalize gradients by minibatch size
            size_t batchSizeActual = end - start;
            for (int k = 0; k < m_nFeatures; ++k)
            {
                dDeltaWeights[k] /= batchSizeActual;
            }
            dDeltaBias /= batchSizeActual;

            // Update parameters
            for (int k = 0; k < m_nFeatures; ++k)
            {
                m_dWeights[k] -= m_dLearningRate * dDeltaWeights[k];
            }
            m_dBias -= m_dLearningRate * dDeltaBias;
        }

        // Check convergence after each epoch
        double dCurrentLoss = ComputeLoss(X, y);
        if (fabs(dPreviousLoss - dCurrentLoss) < m_dTolerance)
        {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
        dPreviousLoss = dCurrentLoss;

        if (i == m_iMaxIterations - 1)
        {
            std::cout << "Reached maximum iterations without convergence.\n";
        }
    }
}

void LinearRegression::NormalEquation(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    // Convert X and y to Eigen matrices
    Eigen::MatrixXd matX(X.size(), X[0].size() + 1); // Add a column for the bias term
    Eigen::VectorXd vecY(y.size());

    for (size_t i = 0; i < X.size(); ++i)
    {
        matX(i, 0) = 1.0; // Bias term
        for (size_t j = 0; j < X[0].size(); ++j)
            matX(i, j + 1) = X[i][j];
        vecY(i) = y[i];
    }

    // Compute weights using the normal equation: w = (X^T * X)^(-1) * X^T * y
    Eigen::VectorXd weights = (matX.transpose() * matX).ldlt().solve(matX.transpose() * vecY);

    // Update the model's weights and bias
    m_dBias = weights(0); // First element is the bias
    for (int i = 0; i < m_nFeatures; ++i)
        m_dWeights[i] = weights(i + 1); // Remaining elements are the weights
}
