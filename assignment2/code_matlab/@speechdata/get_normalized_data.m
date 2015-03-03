function data_norm = get_normalized_data(self)
   % Function returns normalized data.
   % Keep in mind that it does not change the original. You will 
   % have to do that yourself.
   mean_rep = repmat(self.data_mean, 1, size(self.data,2));
   std_rep = repmat(self.data_std, 1, size(self.data, 2));
   data_norm = (self.data - mean_rep) ./ std_rep;
end
