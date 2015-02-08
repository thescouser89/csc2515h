"""
Here you will need to implement the naive Bayes decision rule.
"""
import numpy as np
def nb_decision(X,p_y,p_x_given_y):
    """
    Computes a vector of binary predictions, where each entry corresponds to the predicted
    label for it's associated row in X.

    Given:
        X           - a num_cases by num_dims array of binary inputs.
        p_y         - an array with two elements corresponding to the prior class
                      probabilities.
        p_x_given_y - A num_dims by 2 matrix corresponding to the naive Bayes
                      probabilities for each attribute.

    Returns:
        A vector of class predictions for each example in X.
    """
    cases = X.shape[0]
    dims = X.shape[1]

    bayes_prob = np.zeros((cases, 2))
    result = np.zeros((cases, 1))

    y_values = p_y.size

    # find the probability for each y value
    for y_value in range(y_values):
        # iterate through each case and calculate the probability
        for case in range(cases):
            prob = p_y[y_value]

            for dim in range(dims):
                if X[case][dim] == 1:
                    prob = prob * p_x_given_y[dim][y_value]
                elif X[case][dim] == 0:
                    prob = prob * (1 - p_x_given_y[dim][y_value])

            bayes_prob[case][y_value] = prob
        print y_value

    # now find out which class has the biggest probability
    for case in range(cases):
        if bayes_prob[case][0] > bayes_prob[case][1]:
            result[case] = 0
        else:
            result[case] = 1

    return result










    # X: num_cases x dim
    # p_y: prior class probabiliy
    # p_x_given_y: num_dims x 2

# need to return a vector of binary predictions, each entry corresponds to the
# predicted label for its associated row in X

