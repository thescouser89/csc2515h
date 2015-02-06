"""
Here you will need to implement the naive Bayes decision rule.
"""
import numpy as np
def nb_decision(X,p_y,p_x_given_y):
	"""
	Computes a vector of binary predictions, where each entry corresponds to the predicted
	label for it's associated row in X.

	Given:
		X   		- a num_cases by num_dims array of binary inputs.
		p_y  	 	- an array with two elements corresponding to the prior class
					  probabilities.
		p_x_given_y - A num_dims by 2 matrix corresponding to the naive Bayes
		              probabilities for each attribute.
    
    Returns:
    	A vector of class predictions for each example in X.
	"""
    raise Exception('Implement this function.')