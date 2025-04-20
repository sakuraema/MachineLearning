#pragma once

#include "Algorithms.h"

class LogisticRegression
{
public:
    LogisticRegression(double lr = 0.01, int n_iter = 1000);
    void Train(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    int Predict(const std::vector<double>& x) const;
    void Clear() { std::fill(m_dWeights.begin(), m_dWeights.end(), 0); m_dBias = 0.0; }

    inline std::vector<double> GetWeights() const { return m_dWeights; }
    inline double GetBias() const { return m_dBias; }

private:
    double PredictProbability(const std::vector<double>& x) const;
    double ComputeSigmoid(double z) const;    
    std::vector<double> ComputeSoftmax(const std::vector<double>& z) const;
    void ComputeGradients(const std::vector<std::vector<double>>& X, const std::vector<int>& y, std::vector<double>& dDeltaWeights, double& dDeltaBias) const;

private:
    std::vector<double> m_dWeights;  // Includes bias term as first element
    double m_dBias;
    double m_dLearningRate;
    int m_nFeatures;
    int m_iMaxIterations;
	int m_nClasses;
};
