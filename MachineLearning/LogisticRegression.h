#pragma once

#include "Algorithms.h"
#include <vector>
#include <Eigen/Dense>

class LogisticRegression
{
public:
    LogisticRegression(double lr = 0.01, int n_iter = 1000);
    void Train(int nFeatures, const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<int> Predict(const std::vector<std::vector<double>>& X) const;
    void Clear() { std::fill(m_dWeights.begin(), m_dWeights.end(), 0); m_dBias = 0.0; }

    inline std::vector<double> GetWeights() const { return m_dWeights; }
    inline double GetBias() const { return m_dBias; }

private:
    double ComputeSigmoid(double z) const;    
    void ComputeGradients(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, Eigen::VectorXd& dDeltaWeights, double& dDeltaBias) const;

private:
    std::vector<double> m_dWeights;
    double m_dBias;
    double m_dLearningRate;
    int m_nFeatures;
    int m_iMaxIterations;
	int m_nClasses;
};
