#pragma once

#include <vector>

class LinearRegression
{
public:
    LinearRegression(int nFeatures, double dLearningRate = 0.01, double dTolerance = 1e-6, int iMaxIterations = 1000);

    double ComputeHypothesis(const std::vector<double>& x) const;
    double ComputeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;
    void ComputeGradients(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::vector<double>& dDeltaWeights, double& dDeltaBias) const;
    void Fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    inline std::vector<double> GetWeights() const { return m_dWeights; }
    inline double GetBias() const { return m_dBias; }

private:
    std::vector<double> m_dWeights;
    double m_dBias;
    double m_dLearningRate;
    double m_dTolerance;
    int m_nFeatures;
    int m_iMaxIterations;
};
