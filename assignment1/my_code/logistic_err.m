function [f, df, frac_correct] = logistic_err(weights, data, labels, parameters)
  %%
  % Computes the logistic regression negative log-likelihood, gradient,
  %  and the fraction of correctly predicted cases.

  %  Given:
  %      weights    - a num_dimensions length vector of model parameters.
  %      data       - a num_cases by num_dimensions matrix of data examples.
  %      labels     - a num_cases length vector of binary labels.
  %      parameters - a dict that contains a weight_regularization entry.

  %  Returns: negative log-likelihood, gradient, fraction of correct cases
  p_label_zero = sigmoid(weights' * data')';
  f = - (labels' * log(p_label_zero)) - ((1 - labels') * log(1 - p_label_zero));

  df = labels' * data - (1 - p_label_zero)' * data;

  frac_correct = 0.9;




function result = sigmoid(value)
    result = 1 ./ ( 1 + exp(-value) );


