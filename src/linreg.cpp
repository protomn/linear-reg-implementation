#include "linreg.hpp"
#include <iostream>
#include <string>
#include <stdexcept>
#include <numeric> //for std::iota
#include <random> // for std::mt19937
#include <cmath> //for std::sqrt

//A learning point:
/*
    Unlike Numpy, Eigen does not support broadcasting in matrix operations.
    Therefore, to add a bias term to each row of a matrix, we need to
    explicitly replicate the bias vector across all rows or use other techniques.
*/

//Defining the constructor.

LinearRegression::LinearRegression(const std::string &method,
                                   double l2,
                                   double learning_rate,
                                   int epochs,
                                   bool fit_intercept,
                                   int batch_size,
                                   double tol,
                                   int patience,
                                   double momentum)
                :w(),
                 b(0.0),
                 method(method),
                 l2(l2),
                 learning_rate(learning_rate),
                 epochs(epochs),
                 fit_intercept(fit_intercept),
                 fit_flag(false),
                 batch_size(batch_size),
                 tol(tol),
                 patience(patience),
                 momentum(momentum)
{
    //Standardize all methods to lower case.
    std::string lower = method;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    //All validation checks.

    if (learning_rate <= 0.0 && method == "gradient_descent")
    {
        throw std::invalid_argument("Learning rate must always be positive.");
    }

    if (batch_size < 0)
    {
        throw std::invalid_argument("Batch size cannot be negative.");
    }

    if (learning_rate > 1.0)
    {
        std::cerr << "Warning: learning_rate is unusually high (" << learning_rate << ")\n";
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

    if (tol <= 0.0)
    {
        throw std::invalid_argument("Tolerance must be positive.");
    }

    if (patience <= 0)
    {
        throw std::invalid_argument("Patience must be positive.");
    }

    if (momentum <0.0 || momentum >= 1.0)
    {
        throw std::invalid_argument("Momentum must be in the range [0.0, 1.0).]");
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

    if (method == "normal_equation")
    {
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
    
    else if (method == "gradient_descent")
    {
        const auto n_samples = X.rows();
        const auto n_features = X.cols();
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 rng(rd()); //Fixed random seed for reproducibility (same as python seed).

        w = Eigen::VectorXd::Zero(X.cols());
        b = 0.0;
        loss_curve.clear(); //Clearing previous loss values if any.

        //Initializing momentum terms (velocity vectors).
        Eigen::VectorXd v_w = Eigen::VectorXd::Zero(w.size());
        double v_b = 0.0;

        if (batch_size > 0 && batch_size < n_samples)
        {
            Eigen::MatrixXd X_batch(batch_size, n_features);
            Eigen::VectorXd y_batch(batch_size);

            double prev_loss = std::numeric_limits<double>::infinity();
            int no_improvement = 0;

            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                std::shuffle(indices.begin(), indices.end(), rng);

                for (int start = 0; start < n_samples; start += batch_size)
                {
                    int end = std::min(start + static_cast<Eigen::Index>(batch_size), n_samples);
                    int curr_batch = end - start;

                    //Allocate buffers once per epoch, then reuse them for every mini-batch.
                    //To reduce heap churn.
                    X_batch.block(0,0, curr_batch, n_features) = X.middleRows(start, curr_batch);
                    y_batch.head(curr_batch) = y.segment(start, curr_batch);

                    Eigen::VectorXd y_pred = X_batch * w;

                    if (fit_intercept)
                    {
                        y_pred.array() += b;
                    }

                    Eigen::VectorXd error = y_pred - y_batch;

                    Eigen::VectorXd grad_w = (X_batch.transpose() * error)/curr_batch;

                    if (l2 >0.0)
                    {
                        grad_w += (l2/curr_batch) * w;
                    }

                    double grad_b = fit_intercept ? error.mean() : 0.0;

                    v_w = momentum * v_w +(1.0 - momentum) * grad_w;
                    w -= learning_rate * v_w;

                    if (fit_intercept)
                    {
                        v_b = momentum * v_b + (1.0 - momentum) * grad_b;
                        b -= learning_rate * v_b;
                    }

                }

                Eigen::VectorXd bias_vec = Eigen::VectorXd::Constant(X.rows(), b);
                double loss = (X * w + (fit_intercept ? bias_vec : Eigen::VectorXd::Zero(n_samples)) - y).squaredNorm() / (2.0 * n_samples);

                if (epoch%100 == 0)
                {  
                    loss_curve.push_back(loss);
                    std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
                }

                //Early stopping logic.

                if (std::abs(prev_loss - loss) < tol)
                {
                    ++no_improvement;
                    if (no_improvement >= patience)
                    {
                        std::cout << "Early stopping triggered at epoch " << epoch
                                  << " (Δloss < " << tol
                                  << " for " << patience << " consecutive checks)\n";
                        break;
                    }
                }
                else
                {
                    no_improvement = 0; //resets if improvement is seen.
                }

                prev_loss = loss;
            }
            
        }
        else
        {
            batch_size = n_samples;
            double prev_loss = std::numeric_limits<double>::infinity();
            int no_improvement = 0;


            for (int i = 0; i < epochs; ++i)
            {
                Eigen::VectorXd y_pred = X * w;
                if (fit_intercept)
                {
                    y_pred.array() += b;
                }

                Eigen::VectorXd error = y_pred - y;
                Eigen::VectorXd grad_w = (X.transpose() * error)/n_samples;

                if (l2 > 0.0)
                {
                    grad_w += (l2/n_samples) * w;
                }

                double grad_b = 0.0;
                if (fit_intercept)
                {
                    grad_b = error.mean();
                }

                v_w = momentum * v_w +(1.0 - momentum) * grad_w;
                w -= learning_rate * v_w;

                if (fit_intercept)
                {
                    v_b = momentum * v_b + (1.0 - momentum) * grad_b;
                    b -= learning_rate * v_b;
                }

                double loss = (error.squaredNorm()/(2 * n_samples));

                if (i%100 == 0)
                {
                    loss_curve.push_back(loss); //Storing loss for this epoch.
                    std::cout << "Epoch: " << i << " Loss: " << loss << std::endl;
                }

                //Early stopping logic.

                if (std::abs(prev_loss - loss) < tol)
                {
                    ++no_improvement;
                    if (no_improvement >= patience)
                    {
                        std::cout << "Early stopping triggered at epoch " << i
                                  << " (Δloss < " << tol
                                  << " for " << patience << " consecutive checks)\n";
                        break;
                    }
                }
                else
                {
                    no_improvement = 0; //resets if improvement is seen.
                }

                prev_loss = loss;
            }
        }
        
        fit_flag = true;
        std::cout << "Training complete. Coefficients: " << w.transpose() << ", bias: " << b << std::endl;
    }

    else
    {
        throw std::invalid_argument("Unsupported training method: " + method);
    }
}

// Defining the predict method algorithm.

Eigen::VectorXd LinearRegression::predict(const Eigen::Ref<const Eigen::MatrixXd> &X) const
{
    if (!fit_flag)
    {
        throw std::runtime_error("Model has not been fitted yet.");
    }

    if (X.cols() != w.size())
    {
        throw std::invalid_argument("Input feature dimension does not match trained weights.");
    }
    
    Eigen::VectorXd prediction;

    if (fit_intercept)
    {   
        
        prediction = X * w + Eigen::VectorXd::Ones(X.rows()) * b;
    }
    else
    {
        prediction = X * w;
    }

    return prediction;

}

// Implementing Evaluation Methods

double LinearRegression::rmse_(const Eigen::Ref<const Eigen::MatrixXd> &X,
                               const Eigen::Ref<const Eigen::VectorXd> &y) const
{
    Eigen::VectorXd y_pred = predict(X);
    Eigen::VectorXd difference = y_pred - y;
    return std::sqrt(difference.array().square().mean());
}

double LinearRegression::mae_(const Eigen::Ref<const Eigen::MatrixXd> &X,
                              const Eigen::Ref<const Eigen::VectorXd> &y) const
{
    Eigen::VectorXd y_pred = predict(X);
    Eigen::VectorXd difference = (y_pred - y).array().abs();
    return difference.mean();
}

double LinearRegression::r2_score_(const Eigen::Ref<const Eigen::MatrixXd> &X,
                                   const Eigen::Ref<const Eigen::VectorXd> &y) const
{
    Eigen::VectorXd y_pred = predict(X);
    double ss_res = (y_pred - y).squaredNorm();
    double ss_total = (y.array() - y.mean()).matrix().squaredNorm();
    return 1.0 - (ss_res/ss_total);
}


// Defining the coef_ method to return the weights.

Eigen::VectorXd LinearRegression::coef_() const
{
    if (!fit_flag)
    {
        throw std::runtime_error("Model has not been fitted yet."); 
    }
    else
    {
        return w;
    }
}
