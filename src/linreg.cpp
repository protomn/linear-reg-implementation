#include "linreg.hpp"
#include <iostream>
#include <string>
#include <stdexcept>

//Defining the constructor.

LinearRegression::LinearRegression(const std::string &method,
                                   double l2,
                                   double learning_rate,
                                   int epochs,
                                   bool fit_intercept)
                :w(),
                 b(0.0),
                 method(method),
                 l2(l2),
                 learning_rate(learning_rate),
                 epochs(epochs),
                 fit_intercept(fit_intercept),
                 fit_flag(false)
{
    std::string lower = method;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (learning_rate <= 0.0 && method == "gradient_descent")
    {
        throw std::invalid_argument("Learning rate must always be positive.");
    }

    if (method != "normal_equation" && method != "gradient_descent")
    {
        throw std::invalid_argument("Method must be either 'normal_equation' or 'gradient_descent'.");
    }

    if (l2 < 0.0)
    {
        throw std::invalid_argument("L2 regularization cannot be negative.");
    }

    if (epochs <= 0 && method == "gradient_descent") 
    {
        throw std::invalid_argument("Epochs must be positive for gradient descent.");
    }
}

//Defining the fit method algorithm.

void LinearRegression::fit(const Eigen::Ref<const Eigen::MatrixXd> &X,
                           const Eigen::Ref<const Eigen::VectorXd> &y)
{

}

// Defining the predict method algorithm.

Eigen::VectorXd LinearRegression::predict(const Eigen::Ref<const Eigen::MatrixXd> &X) const
{

}

// Defining the coef_ method to return the weights.

Eigen::VectorXd LinearRegression::coef_() const
{

}
