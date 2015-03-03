% Navdeep Jaitly, Kevin Swerssky, 2013.
% ndjaitly@gmail.com, kswerss@gmail.com
%
function results =get_utterance_data(self, ...
                                     utterance_num, get_labels)
   s_f = self.utt_indices(utterance_num,1);
   e_f = self.utt_indices(utterance_num,2);
   data = self.data(:, s_f:e_f-1);
   left_context = int32(self.num_frames_per_pt/2);
   right_context = self.num_frames_per_pt - left_context;
   
   left = repmat(data(:,1),1,left_context);
   right = repmat(data(:,end),1,right_context);
   data = [left,data,right];
   data_stacked = zeros(self.data_dim,e_f-s_f);
   for i=1:self.num_frames_per_pt
      s = (i-1)*self.frame_dim + 1;
      e = i*self.frame_dim;
      data_stacked(s:e,:) = data(:,i:(e_f-s_f+i-1));
   end

   results = struct();
   results.data = data_stacked;
   if get_labels
      eye_mat = eye(self.label_dim);
      label_mat = eye_mat(:,self.targets(s_f:e_f-1)+1);
      results.label_mat = label_mat;
   end
end
