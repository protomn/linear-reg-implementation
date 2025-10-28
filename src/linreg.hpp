/*
This header file covers the design descisions made in the implementation of linear regression algorithms.
It outlines the structure, key components, and rationale behind the choices to facilitate understanding and maintenance.

1. Intercept Policy
    - C++ will maintain an internal scalar b.
    - During fit, the algorithm does not add columns of ones to X, instead estimates both w and b.
    - During predict, output is Xw + b.
    - When L2 regularization is used, b is excluded from the penalty.

2. Training Modes
    - Supports two training paths: Normal Equation (ridge optional) and Batch Gradient Descent.
    - Normal Equation is the default due to its efficiency for small to medium datasets.
    - Batch Gradient Descent is included for larger datasets to benchmark superiority of C++ over Python.
    Decisions:
        - Use of Double Precision.
        - Normal Equation Solver: LDLT Decomp (good for SPD matrices); no explicit inverse.
        - Gradient Descent Defaults: Learning Rate = 0.01, Epochs = 1000 (to be tuned later).
    - Gradient Descent uses full-batch updates.
    - Gradient includes L2 term if l2>0.

3. Data Shapes and Ownership
    - Python will pass numpy.npdarrays to C++. These will be stored in Eigen::VectorXd for efficient access without copying.
    - All inputs must have dtype float64 and must be contiguous.
    - Shape Expectations:
        - X: (n_samples, n_features)
        - y: (n_samples,)
    - Internally stored as:
        - X: Eigen::MatrixXd (n_samples, n_features)
        - y: Eigen::VectorXd (n_samples)

4. Public API
    - Constructor Hyperparameters:
        - "method": "normal_equation" or "gradient_descent"
        - "l2": double (default 0.0)
        - "learning_rate": double (default: 0.0; only for gradient descent).
        - epochs: int (default: 1000; only for gradient descent).
        - fit_intercept: bool (default: true).
    - Methods:
        - fit(X, y) -> void (will retunn self in Python bindings for chaining).
        - predict(X) -> vector.
        - coef_() -> const ref or a copy of w.
    - Python properties:
        - Expose _coeff as a Numpy array (copy/view).
    - fit overwrites w.
    - predict requires w to be set.
    - coef_ returns current weights.
    - Throws on errors (shape mismatches, unfit model, etc.).

5. Error Handling
    - If X.rows() != y.size(): throw std::invalid_argument("X and y size mismatch")
    - If X.cols() == 0: throw std::invalid_argument("Empty feature matrix")
    - If predict() called before fit(): throw std::runtime_error("Model not fitted")
    - If X.cols() != w.size(): throw std::invalid_argument("Feature dimension mismatch")

6. Numeric Defaults and Reproducibility
    - Deterministics calculations (no internal RNG).
    - For large or poorlu conditioned datasets, ridge (L2>0) improves stability.; scaling features is recommended but not enforced.
*/

#pragma once
#ifndef LINREG_HPP
#define LINREG_HPP

#include <Eigen/Dense>
#include <string>
#include <stdexcept>

class LinearRegression{
    public:
        Eigen::VectorXd w;
        double b;
        std::string method;
        double l2;
        double learning_rate;
        int epochs;
        bool fit_intercept;
        bool fit_flag; //indicates if fit() has been called.

        LinearRegression(std::string method = "normal_equation",
                         double l2 = 0.0,
                         double leanring_rate = 0.01,
                         int epochs = 1000,
                         bool fit_intercept = true);
        
        void fit(const Eigen::Ref<const Eigen::MatrixXd> &X,
                 const Eigen::Ref<const Eigen::VectorXd> &y);

        Eigen::VectorXd predict(const Eigen::Ref<const Eigen::MatrixXd> &X) const;
        Eigen::VectorXd coef_() const;
        
    private:

};

#endif