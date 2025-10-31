#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "linreg.hpp"
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Hybrid Linear Regression Module using C++ and pybind11";

    py::class_<LinearRegression> cls(m, "LinearRegression");

    cls.def(py::init<const std::string &, double, double, int, bool, int>(),
            py::arg("method") = "normal_equation", py::arg("l2") = 0.0, py::arg("learning_rate") = 0.01,
            py::arg("epochs") = 1000, py::arg("fit_intercept") = true, py::arg("batch_size") = 0);

    cls.def("fit",
            [](LinearRegression &self,
               const Eigen::Ref<const Eigen::MatrixXd> &X,
               const Eigen::Ref<const Eigen::VectorXd> &y) -> LinearRegression &
               {
                py::gil_scoped_release release; //Python GIL released
                self.fit(X, y); //C++ Computation happens here
                return self; //Return self for method chaining
               },
            py::arg("X"),
            py::arg("y"),
        py::return_value_policy::reference_internal);

    cls.def("predict",
            &LinearRegression::predict,
            py::arg("X"),
        "Predict target values for sample matrix X.");
    
    cls.def("coef_",
            &LinearRegression::coef_,
        "Returned learned coefficient vector w.");

    cls.def("loss_curve_",
            &LinearRegression::loss_curve_,
        "Returns the loss values recorded at every 100 epochs during training.");

    cls.def("rmse_", &LinearRegression::rmse_, py::arg("X"), py::arg("y"));

    cls.def("mae_", &LinearRegression::mae_, py::arg("X"), py::arg("y"));

    cls.def("r2_score_", &LinearRegression::r2_score_, py::arg("X"), py::arg("y"));
}
