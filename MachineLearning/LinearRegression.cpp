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
    Eigen::MatrixXd X_eigen(X.size(), m_nFeatures);
    for (size_t i = 0; i < X.size(); ++i)
    {
        for (size_t j = 0; j < m_nFeatures; ++j)
            X_eigen(i, j) = X[i][j];
    }
    const Eigen::VectorXd y_eigen = Eigen::Map<const Eigen::VectorXd>(y.data(), y.size());

    switch (eType)
    {
    case AlgorithmType::BatchGradientDescent:
        BatchGradientDescent(X_eigen, y_eigen);
	    std::cout << "Batch Gradient Descent completed.\n";
        break;
    case AlgorithmType::StochasticGradientDescent:
        StochasticGradientDescent(X_eigen, y_eigen);
	    std::cout << "Stochastic Gradient Descent completed.\n";
        break;
    case AlgorithmType::MinibatchGradientDescent:
    {
        int nBatchSize = std::max(1, static_cast<int>(X.size() / 10)); // Set batch size to 10% of the sample size, minimum 1
        MinibatchGradientDescent(X_eigen, y_eigen, nBatchSize);
        std::cout << "Minibatch Gradient Descent completed with batch size: " << nBatchSize << "\n";
        break;
    }
    case AlgorithmType::NormalEquation:
        NormalEquation(X_eigen, y_eigen);
	    std::cout << "Normal Equation completed.\n";
        break;
    default:
        break;
    }
}

std::vector<double> LinearRegression::Predict(const std::vector<std::vector<double>>& X) const
{
	if (X.size() == 0)
	{
		std::cerr << "Error: The input data X must not be empty.\n";
		return {};
	}

	if (X[0].size() != m_nFeatures)
	{
		std::cerr << "Error: The number of features in X does not match the trained model.\n";
		return {};
	}

	std::vector<double> yPred(X.size(), 0.0);
	for (size_t i = 0; i < X.size(); ++i)
		yPred[i] = ComputeHypothesis(X[i]);
	return yPred;
}

double LinearRegression::ComputeHypothesis(const std::vector<double>& x) const
{
    // Hypothesis: h(x) = w1*x1 + w2*x2 + ... + wn*xn + b
    double dPredict = m_dBias;
	Eigen::VectorXd x_eigen = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
	Eigen::VectorXd dWeights_eigen = Eigen::Map<const Eigen::VectorXd>(m_dWeights.data(), m_dWeights.size());
	dPredict += dWeights_eigen.dot(x_eigen);
    return dPredict;
}

void LinearRegression::BatchGradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(m_nFeatures);
    double bias = m_dBias;

    double dPreviousLoss = std::numeric_limits<double>::infinity();

    for (int i = 0; i < m_iMaxIterations; ++i)
    {
        // Compute predictions: h(x) = X * weights + bias
        Eigen::VectorXd predictions = X * weights + Eigen::VectorXd::Constant(X.rows(), bias);

        // Compute errors: error = h(x) - y
        Eigen::VectorXd errors = predictions - y;

        // Compute gradients
        Eigen::VectorXd gradientWeights = (X.transpose() * errors) / X.rows();
        double gradientBias = errors.mean();

        // Update parameters
        weights -= m_dLearningRate * gradientWeights;
        bias -= m_dLearningRate * gradientBias;

        // Compute loss
        double dCurrentLoss = (errors.array().square().sum()) / (2 * X.rows());

        // Check convergence
        if (fabs(dPreviousLoss - dCurrentLoss) < m_dTolerance)
        {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
        dPreviousLoss = dCurrentLoss;

        if (i == m_iMaxIterations - 1)
            std::cout << "Reached maximum iterations without convergence (" + std::to_string(m_iMaxIterations) + ").\n";
    }

    // Update the model's weights and bias
    m_dWeights.assign(weights.data(), weights.data() + weights.size());
    m_dBias = bias;
}

void LinearRegression::StochasticGradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(m_nFeatures);
    double bias = m_dBias;

    double dPreviousLoss = std::numeric_limits<double>::infinity();

    for (int i = 0; i < m_iMaxIterations; ++i)
    {
        for (int j = 0; j < X.rows(); ++j)
        {
            // Compute prediction for a single sample
            double prediction = X.row(j).dot(weights) + bias;

            // Compute error
            double error = prediction - y(j);

            // Compute gradients
            Eigen::VectorXd gradientWeights = error * X.row(j).transpose();
            double gradientBias = error;

            // Update parameters
            weights -= m_dLearningRate * gradientWeights;
            bias -= m_dLearningRate * gradientBias;
        }

        // Compute loss
        Eigen::VectorXd predictions = X * weights + Eigen::VectorXd::Constant(X.rows(), bias);
        Eigen::VectorXd errors = predictions - y;
        double dCurrentLoss = (errors.array().square().sum()) / (2 * X.rows());

        // Check convergence
        if (fabs(dPreviousLoss - dCurrentLoss) < m_dTolerance)
        {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
        dPreviousLoss = dCurrentLoss;

        if (i == m_iMaxIterations - 1)
            std::cout << "Reached maximum iterations without convergence (" + std::to_string(m_iMaxIterations) + ").\n";
    }

    // Update the model's weights and bias
    m_dWeights.assign(weights.data(), weights.data() + weights.size());
    m_dBias = bias;
}

void LinearRegression::MinibatchGradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int batchSize)
{
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(m_nFeatures);
    double bias = m_dBias;

    double dPreviousLoss = std::numeric_limits<double>::infinity();

    for (int i = 0; i < m_iMaxIterations; ++i)
    {
        for (int start = 0; start < X.rows(); start += batchSize)
        {
            int end = std::min(start + batchSize, static_cast<int>(X.rows()));
            Eigen::MatrixXd batchX = X.middleRows(start, end - start);
            Eigen::VectorXd batchY = y.segment(start, end - start);

            // Compute predictions for the minibatch
            Eigen::VectorXd predictions = batchX * weights + Eigen::VectorXd::Constant(batchX.rows(), bias);

            // Compute errors
            Eigen::VectorXd errors = predictions - batchY;

            // Compute gradients
            Eigen::VectorXd gradientWeights = (batchX.transpose() * errors) / batchX.rows();
            double gradientBias = errors.mean();

            // Update parameters
            weights -= m_dLearningRate * gradientWeights;
            bias -= m_dLearningRate * gradientBias;
        }

        // Compute loss
        Eigen::VectorXd predictions = X * weights + Eigen::VectorXd::Constant(X.rows(), bias);
        Eigen::VectorXd errors = predictions - y;
        double dCurrentLoss = (errors.array().square().sum()) / (2 * X.rows());

        // Check convergence
        if (fabs(dPreviousLoss - dCurrentLoss) < m_dTolerance)
        {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
        dPreviousLoss = dCurrentLoss;

        if (i == m_iMaxIterations - 1)
            std::cout << "Reached maximum iterations without convergence (" + std::to_string(m_iMaxIterations) + ").\n";
    }

    // Update the model's weights and bias
    m_dWeights.assign(weights.data(), weights.data() + weights.size());
    m_dBias = bias;
}

void LinearRegression::NormalEquation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    // Add a column for the bias term
    Eigen::MatrixXd matXWithBias(X.rows(), X.cols() + 1);
    matXWithBias.col(0) = Eigen::VectorXd::Ones(X.rows()); // Bias term
    matXWithBias.block(0, 1, X.rows(), X.cols()) = X;

    // Compute weights using the normal equation: w = (X^T * X)^(-1) * X^T * y
    Eigen::VectorXd weights = (matXWithBias.transpose() * matXWithBias).ldlt().solve(matXWithBias.transpose() * y);

    // Update the model's weights and bias
    m_dBias = weights(0); // First element is the bias
    m_dWeights.assign(weights.data() + 1, weights.data() + weights.size()); // Remaining elements are the weights
}
