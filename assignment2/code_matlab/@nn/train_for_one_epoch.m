function self = train_for_one_epoch(self, data_src, eps, momentum, ...
                                        l2, batch_size);
   % Work horse of the learning for one epoch. As long as the other
   % functions are working correctly, and satisfy the interface, 
   % there should be no need to change this function. 

 
   num_pts = 0;
   classif_err_sum = 0;
   lg_p_sum = 0;
   batch = 0;

   data_src = data_src.randomize_data_indices();
   num_training_pts = numel(data_src.indices);
   num_layers = length(self.lst_layers);

   for start_index=1:batch_size:num_training_pts
      end_index = min(start_index + batch_size-1, num_training_pts);
      [data, label_mat] = data_src.get_randomized_data(start_index, end_index);
      num_pts = num_pts + (end_index-start_index+1);
      lst_layer_outputs = fwd_prop(self, data);
      results = self.lst_layers(num_layers).compute_accuracy(...
                                  lst_layer_outputs{num_layers}, label_mat);
      num_correct = results.num_correct;
      log_prob = results.log_prob;

      classif_err_sum = classif_err_sum + (end_index-start_index+1- num_correct);
      lg_p_sum = lg_p_sum + log_prob;
      
      self = back_prop(self, lst_layer_outputs, data, label_mat);
      self = apply_gradients(self, momentum, eps, l2);
      batch = batch + 1;

   end
   
   classif_err = classif_err_sum*100./num_pts;
   lg_p = lg_p_sum/num_pts;
   fprintf(1, 'Batch = %d, Classif Err = %.3f lg(p) %.4f\n', ...
           batch, classif_err, lg_p);

