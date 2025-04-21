#pragma once

#include "Algorithms.h"
#include <vector>
#include <Eigen/Dense>

class LinearRegression
{
public:
    LinearRegression(double dLearningRate = 0.01, double dTolerance = 1e-6, int iMaxIterations = 100000);

    void Train(AlgorithmType eType, int nFeatures, const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<double> Predict(const std::vector<std::vector<double>>& X) const;
	void Clear() { std::fill(m_dWeights.begin(), m_dWeights.end(), 0); m_dBias = 0.0; }

    inline std::vector<double> GetWeights() const { return m_dWeights; }
    inline double GetBias() const { return m_dBias; }

private:
    double ComputeHypothesis(const std::vector<double>& x) const;
	void BatchGradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
	void StochasticGradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
	void MinibatchGradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int batchSize);
	void NormalEquation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

private:
    std::vector<double> m_dWeights;
    double m_dBias;
    double m_dLearningRate;
    double m_dTolerance;
    int m_nFeatures;
    int m_iMaxIterations;
};
