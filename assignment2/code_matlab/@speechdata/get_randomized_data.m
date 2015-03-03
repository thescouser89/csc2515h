function [batch_data, label_mat] =  get_randomized_data(self, start_index, end_index)
   num_pts = end_index-start_index+1;
   indices = reshape(self.indices(start_index:end_index), 1, num_pts);
   indices_with_ctxt = repmat(indices, self.num_frames_per_pt, 1)  + ...
                          repmat([0:(self.num_frames_per_pt-1)]', 1, num_pts);
   indices_with_ctxt = reshape(indices_with_ctxt, 1, numel(indices_with_ctxt));
   batch_data = reshape(self.data(:,indices_with_ctxt), self.data_dim, num_pts);
   labels = self.targets(indices + floor(self.num_frames_per_pt/2));
   label_mat = self.label_eye(:, labels+1);
