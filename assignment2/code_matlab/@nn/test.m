function results = test(self, data_src)
   num_pts = 0;
   classif_err_sum = 0;
   lg_p_sum = 0;

   for utt_num=1:data_src.num_utterances
      results = data_src.get_utterance_data(utt_num, true);
      data = results.data ;
      label_mat = results.label_mat;
      num_pts = num_pts + size(data, 2);
      lst_layer_outputs = fwd_prop(self, data);
      results = self.lst_layers(self.num_layers).compute_accuracy(...
                     lst_layer_outputs{self.num_layers}, label_mat);
      classif_err_sum =  classif_err_sum + size(data,2) - results.num_correct;
      lg_p_sum = lg_p_sum + results.log_prob;

      classif_err = classif_err_sum*100.0/num_pts;
      lg_p = lg_p_sum / num_pts;
   end
   fprintf(1, 'TESTING Classif Err = %.3f, lg(p) %.4f\n', classif_err, lg_p);
   acc =  100 - classif_err;
   results = struct('accuracy', acc, 'lg_p', lg_p);
