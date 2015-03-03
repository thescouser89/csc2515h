function results = create_predictions(self, data_src)
   num_pts = 0;
   classif_err_sum = 0;
   lg_p_sum = 0;

   results = struct('pred_lst', {cell(1, data_src.num_utterances)}, 'num_output_frames', 0);
   num_output_frames = 0;
   for utt_num = 1:data_src.num_utterances
      cur_results = data_src.get_utterance_data(utt_num, false);
      data = cur_results.data ;
      lst_layer_outputs = fwd_prop(self, data);
      lg_p = log(1e-32+lst_layer_outputs{self.num_layers});
      results.pred_lst{utt_num} = lg_p;
      results.num_output_frames = results.num_output_frames + size(data,2);
   end
