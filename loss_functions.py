import numpy as np

## Linear Regression ##

def MSE(X, y, w):
    """
    This function calculates the (in-sample) mean squared error per iteration / after each weight update.
    The mean squared error can also be written as the cost J.
    """
    n = X.shape[0]
    f = X @ w
    J = np.sum(np.power((y - f), 2)) / n
    return J

def RMSE(X, y, w):
    """
    This function calculates the (in-sample) root mean squared error per iteration / after each weight update.
    The mean squared error can also be written as the cost J.
    """
    n = X.shape[0]
    f = X @ w
    J = np.sqrt(np.sum(np.power((y - f), 2)) / n)
    return J


## Logistic Regression ##

def log_error(X, y, w):
    """
    This function computes the (in-sample) logistic log-likelihood after each weight update.
    The logistic in-sample error can also be written as the cost J
    """
    N = X.shape[0]
    J = np.sum(np.log(1.0 + np.exp(-y * (X @ w)))) / N
    return J