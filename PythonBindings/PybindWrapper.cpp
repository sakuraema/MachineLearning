#include <LinearRegression.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(ml_cpp, m)
{
	py::class_<LinearRegression>(m, "LinearRegression")
		.def(py::init<>())
		.def("train", &LinearRegression::Train)
		.def("predict", &LinearRegression::Predict)
		.def("clear", &LinearRegression::Clear)
		.def("get_weights", &LinearRegression::GetWeights)
		.def("get_bias", &LinearRegression::GetBias);
	py::enum_<AlgorithmType>(m, "AlgorithmType")
		.value("BatchGradientDescent", AlgorithmType::BatchGradientDescent)
		.value("StochasticGradientDescent", AlgorithmType::StochasticGradientDescent)
		.value("MinibatchGradientDescent", AlgorithmType::MinibatchGradientDescent)
		.value("NormalEquation", AlgorithmType::NormalEquation);
}
