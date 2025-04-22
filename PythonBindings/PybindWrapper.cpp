#include <LinearRegression.h>
#include <LogisticRegression.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(ml_cpp, m)
{
    py::enum_<AlgorithmType>(m, "AlgorithmType")
        .value("BatchGradientDescent", AlgorithmType::BatchGradientDescent)
        .value("StochasticGradientDescent", AlgorithmType::StochasticGradientDescent)
        .value("MinibatchGradientDescent", AlgorithmType::MinibatchGradientDescent)
        .value("NormalEquation", AlgorithmType::NormalEquation);

    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<double, double, int>(), 
             py::arg("learning_rate") = 0.01, 
             py::arg("tolerance") = 1e-6, 
             py::arg("max_iterations") = 100000)
        .def("train", &LinearRegression::Train)
        .def("predict", &LinearRegression::Predict)
        .def("clear", &LinearRegression::Clear)
        .def("get_weights", &LinearRegression::GetWeights)
        .def("get_bias", &LinearRegression::GetBias);

	py::class_<LogisticRegression>(m, "LogisticRegression")
		.def(py::init<double, int>(),
			py::arg("learning_rate") = 0.01,
			py::arg("max_iterations") = 100000)
		.def("train", &LogisticRegression::Train)
		.def("predict", &LogisticRegression::Predict)
		.def("clear", &LogisticRegression::Clear)
		.def("get_weights", &LogisticRegression::GetWeights)
		.def("get_bias", &LogisticRegression::GetBias);
}
