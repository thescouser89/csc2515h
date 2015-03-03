% Navdeep Jaitly, Kevin Swersky, 2013.
% email: ndjaitly@gmail.com
%
%
% FILL IN THESE
max_epochs    = 30;
train_db_path = '../../data/train.mat';
dev_db_path   = '../../data/dev.mat';
param_file = 'model.mat';

% Number of contiguous frames to predict phoneme labels. Leave at 15
num_frames_per_pt = 15;


% DECIDE ON MEANINGFUL VALUES FOR THESE VARIABLES
batch_size = 0;
eps        = 0;
momentum   = 0;
l2         = 0;

train_src = speechdata(train_db_path, num_frames_per_pt);
train_src.data = train_src.get_normalized_data();


layer1_def = layer_definition('Layer1', 'SIGMOID_LAYER', ...
                              train_src.get_data_dim(), 100, 0.01);
layer2_def = layer_definition('Layer2', 'SOFTMAX_LAYER', layer1_def.num_units,...
                              train_src.get_target_dim(), 0.01);


lst_def = [layer1_def, layer2_def];
nn_train = nn(lst_def);

fprintf(1, 'Will save output params to file: %s\n', param_file)

%%% MEANING FULL PREPROCESSING GOES HERE

for i=1:max_epochs
    nn_train = nn_train.train_for_one_epoch(train_src, eps, momentum, l2, ...
                                            batch_size);
     %%%% CROSS VALIDATION GOES HERE
end
