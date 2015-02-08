"""
Here you will need to implement a checkgrad method.
"""
import numpy as np

def checkgrad(fun, params, tol, *varargs):
    """
    Given a function handle for an objective that computes a value/gradient,
    a vector of parameters, and a error tolerance for each dimension of computed
    gradient (between 1e-6 and 1e-3), along with any extra necessary arguments;
    returns an estimate of the error in the gradient.
    """
    # make weights a vector
    weights = np.copy(params)
    weights.shape = (params.size, 1)

    # nexamples x ndimensions
    X = varargs[0]

    Y = varargs[1]
    # make Y a vector
    Y.shape = (Y.size, 1)

    # parameters is a dictionary. Has keys:
    #     'learning_rate', 'weight_regularization', 'num_iterations'
    parameters = varargs[2]

    grad_err = np.zeros((weights.size, 1))

    for i in range(grad_err.size):
        error = np.zeros(grad_err.shape);
        error[i] = tol

        log_likelihood_plus, _, _ = fun(weights + error, X, Y, parameters)
        log_likelihood_minus, _, _ = fun(weights - error, X, Y, parameters)
        grad_err[i] = (log_likelihood_plus - log_likelihood_minus) / (2 * tol)

    return grad_err

