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
    if (X.rows() == 0 || X.cols() == 0)
    {
        throw std::invalid_argument("Feature matrix cannot be empty.");
    }

    if (X.rows() != y.size())
    {
        throw std::invalid_argument("Number of samples in X should match size of y.");
    }

    Eigen::MatrixXd X_aug;

    if (fit_intercept)
    {
        X_aug.resize(X.rows(), X.cols() + 1);
        X_aug.leftCols(X.cols()) = X;
        X_aug.rightCols(1).setOnes();
    }
    else
    {
        X_aug = X; //shallow copy.
    }

    Eigen::MatrixXd XtX = X_aug.transpose() * X_aug;
    Eigen::VectorXd Xty = X_aug.transpose() * y;

    //Ridge Regularization

    if (l2 > 0.0)
    {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());

        //Skip last element if fit_intercept = true

        if (fit_intercept)
        {
            I(I.rows() - 1, I.cols() - 1) = 0.0;
        }
        
        XtX += l2 * I;
    }

    Eigen::LDLT<Eigen::MatrixXd> solver(XtX);
    Eigen::VectorXd w_solved = solver.solve(Xty);

    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("LDLT decomposition failed: possibly singular matrix.");
    }

    if(!w_solved.allFinite())
    {
        throw std::runtime_error("Weights contain NaN/Inf values.");
    }

    //Separating the feature weights and bias term.

    if (fit_intercept)
    {
        w = w_solved.head(w_solved.size() - 1);
        b = w_solved[w_solved.size() - 1];
    }
    else
    {
        w = w_solved;
        b = 0.0;
    }

    fit_flag = true;
    std::cout << "Training complete. Coefficients: " << w.transpose() << ", bias: " << b << std::endl;
}

// Defining the predict method algorithm.

Eigen::VectorXd LinearRegression::predict(const Eigen::Ref<const Eigen::MatrixXd> &X) const
{

}

// Defining the coef_ method to return the weights.

Eigen::VectorXd LinearRegression::coef_() const
{

}
