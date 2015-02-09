import numpy as np

def sigmoid(array):
    """ array should be a numpy array
    Return a numpy array which is the sigmoid of the input
    """
    return 1.0 / (1.0 + np.exp(-array))

def one_minus_sigmoid(array):
    """ array should be a numpy array
    Return a numpy array which is the sigmoid of the input
    """
    exponential = np.exp(-array)
    return exponential / (1.0 + exponential)

def logistic_err(weights, data, labels, parameters):
    """
    Computes the logistic regression negative log-likelihood, gradient,
    and the fraction of correctly predicted cases.

    Given:
        weights    - a num_dimensions length vector of model parameters.
        data       - a num_cases by num_dimensions matrix of data examples.
        labels     - a num_cases length vector of binary labels.
        parameters - a dict that contains a weight_regularization entry.

    Returns: negative log-likelihood, gradient, fraction of correct cases
    """
    # weights: dim x 1
    # data:    cases x dim

    # make these vectors
    weights.shape = (weights.size, 1)
    labels.shape = (labels.size, 1)

    # cases x 1
    p_label_zero = sigmoid(data.dot(weights))
    p_label_one = one_minus_sigmoid(data.dot(weights))
    # labels is cases x 1

    # log_likelihood is a real number
    log_likelihood = -( labels.T.dot(np.log( p_label_one )) +
                    (1.0 - labels.T).dot( np.log(p_label_zero) ) )



    # dim x 1
    gradient =  data.T.dot(labels) - data.T.dot(p_label_one)
    gradient[weights.size - 1] = labels.sum() - p_label_one.sum()

    if parameters['weight_regularization'] != 0:
        gradient = gradient + parameters['weight_regularization'] * weights

    predicted = p_label_one

    wrong = np.sum(np.round(np.abs(predicted - labels)))
    frac_correct = (labels.size - wrong) / labels.size

    weights.shape = (weights.size, )
    labels.shape = (labels.size, )
    gradient.shape = (gradient.size, )

    return log_likelihood, gradient, frac_correct





