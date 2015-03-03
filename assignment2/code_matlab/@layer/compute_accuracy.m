% Navdeep Jaitly, Kevin Swersky, 2013
% ndjaitly@gmail.com

function results = compute_accuracy(self, outputs, label_mat)
% Called only for softmax output layer
   [y, corr_label] = max(label_mat, [], 1);
   [y, pred_label] = max(outputs, [], 1);
   num_correct = sum(corr_label == pred_label);
   log_prob = sum(sum(log(outputs+1e-8) .* label_mat));
   results = struct('num_correct', num_correct, 'log_prob', log_prob);
