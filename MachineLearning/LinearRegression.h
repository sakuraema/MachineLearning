#pragma once

#include "Algorithms.h"
#include <vector>

class LinearRegression
{
public:
    LinearRegression(double dLearningRate = 0.01, double dTolerance = 1e-6, int iMaxIterations = 1000);

    void Train(AlgorithmType eType, int nFeatures, const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<double> Predict(const std::vector<std::vector<double>>& X) const;
	void Clear() { std::fill(m_dWeights.begin(), m_dWeights.end(), 0); m_dBias = 0.0; }

    inline std::vector<double> GetWeights() const { return m_dWeights; }
    inline double GetBias() const { return m_dBias; }

private:
    double ComputeHypothesis(const std::vector<double>& x) const;
    double ComputeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;
    void ComputeGradients(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::vector<double>& dDeltaWeights, double& dDeltaBias) const;
	void BatchGradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
	void StochasticGradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
	void MinibatchGradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int batchSize);
	void NormalEquation(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

private:
    std::vector<double> m_dWeights;
    double m_dBias;
    double m_dLearningRate;
    double m_dTolerance;
    int m_nFeatures;
    int m_iMaxIterations;
};
