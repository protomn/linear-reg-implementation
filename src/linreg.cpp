#include "linreg.hpp"

//Defining the constructor.

LinearRegression::LinearRegression(const std::string &method,
                                   double l2,
                                   double learning_rate,
                                   int epochs,
                                   bool fit_intercept)
{

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
