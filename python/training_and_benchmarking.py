import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(root_directory)

from _core import LinearRegression as lr

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LinReg, SGDRegressor as GDReg

np.random.seed(42)

n_samples, n_features = 200_000, 3
X = np.random.randn(n_samples, n_features)
true_w = np.array([1.5, -2.0, 0.5])
true_b = 3.0
y = X @ true_w + true_b

#Benchmarking

#1. CPP Normal Equation
start = time.perf_counter()
cpp_ne = lr(method = "normal_equation", fit_intercept = True)
cpp_ne.fit(np.ascontiguousarray(X, np.float64), np.ascontiguousarray(y, np.float64))
cpp_ne_time = time.perf_counter() - start

#2. CPP Gradient Descent
start = time.perf_counter()
cpp_gd = lr(method = "gradient_descent", learning_rate = 0.01, epochs = 3000)
cpp_gd.fit(np.ascontiguousarray(X, np.float64), np.ascontiguousarray(y, np.float64))
cpp_gd_time = time.perf_counter() - start

#Benchmarking Scikit-learn Equivalents

#3. Scikit-learn Normal Equation
start = time.perf_counter()
sk_lr = LinReg(fit_intercept = True)
sk_lr.fit(X, y)
sk_lr_time = time.perf_counter() - start

#4. Scikit-learn Gradient Descent
start = time.perf_counter()
sk_gd = GDReg(alpha = 0.0, max_iter = 3000, tol = 1e-8, eta0 = 0.01, learning_rate = "constant")
sk_gd.fit(X, y)
sk_gd_time = time.perf_counter() - start

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

print(f"C++ Normal Equation vs scikit-learn OLS RMSE: {rmse(cpp_ne.coef_(), sk_lr.coef_):.6f}")
print(f"C++ Gradient Descent vs SGDRegressor RMSE:     {rmse(cpp_gd.coef_(), sk_gd.coef_):.6f}")
print(f"C++ Normal Equation: {cpp_ne_time:.4f}s")
print(f"Python OLS:          {sk_lr_time:.4f}s")
print(f"C++ Gradient Descent:{cpp_gd_time:.4f}s")
print(f"Python SGD:          {sk_gd_time:.4f}s")


plt.plot(cpp_gd.loss_curve_())
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Gradient Descent Convergence Plot.")
plt.grid(True)
plt.show()