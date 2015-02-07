%%%
% Here you will need to implement a checkgrad method.
%%%
function grad_err = checkgrad(f, weights, e, X, Y, parameters)

  %  Given:
  %      f          - a function handle for computing objective value/gradient
  %      weights    - a vector of weights
  %      e          - error tolerance for each dimension of computed gradient(between 1e-6 and 1e-3)
  %      X          - a matrix containing given data
  %      Y          - a vector for given labels
  %      parameters - a structure that contains a weight_regularization entry.

  %  Returns: an estimate of the error in the gradient.

  % checkgrad: df/dtheta_i ~= (f(theta_1, ... , theta_i + e, ..., theta_n) - f(theta_1, ... , theta_i - e, ..., theta_n)) / 2e

  grad_err = zeros(size(weights, 1), 1);

  for i = 1:size(weights, 1)
    plus_e = weights;
    minus_e = weights;
    plus_e(i) = plus_e(i) + e;
    minus_e(i) = minus_e(i) - e;
    [log_likelihood_plus, gradient, frac_correct] = f(plus_e, X, Y, parameters);
    [log_likelihood_minus, gradient, frac_correct] = f(minus_e, X, Y, parameters);
    grad_err(i) = (log_likelihood_plus - log_likelihood_minus) / (2 * e);
  end
