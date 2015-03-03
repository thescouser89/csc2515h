% Code by Navdeep Jaitly, 2013
% Email: ndjaitly@gmail.com

% FILL IN THESE
test_db_path   = '../../data/test.mat';
param_file = 'model.mat';
predictions_file = 'predictions.mat';

% Number of contiguous frames to predict phoneme labels. Leave at 15
num_frames_per_pt = 15;

test_src = speechdata(test_db_path, num_frames_per_pt);
% APPLY PREPROCESSING HERE THAT WAS USED IN THE TRAINING.
 
nnet = nn();
nnet = nnet.load(param_file);


results = nnet.create_predictions(test_src);
pred_lst = results.pred_lst;
num_output_frames = results.num_output_frames;

predictions_mat = zeros(size(pred_lst{1},1), num_output_frames);
utt_indices = zeros(length(pred_lst), 2);

num_so_far = 1;
for index=1:length(pred_lst)
    predictions = pred_lst{index};
    predictions_mat(:, num_so_far:(num_so_far+size(predictions, 2)-1)) = predictions;
    utt_indices(index, 1) = num_so_far;
    num_so_far = num_so_far + size(predictions, 2) ;
    utt_indices(index, 2) = num_so_far;
 end

params_dict = struct();
params_dict.predictions = predictions_mat;
params_dict.utt_indices = utt_indices;
save(predictions_file, '-struct', 'params_dict');
