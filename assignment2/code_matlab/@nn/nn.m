function self = nn(lst_def)
   if nargin > 0
      obj = create_nnet_from_def(lst_def);
      self = class(obj, 'nn');
   else
      obj = struct('num_layers', 0, 'data_dim', 0, 'cur_epoch', 0, ...
                 'tot_batch', 0, 'lst_num_hid', [], 'lst_layer_names', cell(1,1), ...
                 'lst_layer_type', cell(1,1), 'lst_layers', []);
      self = class(obj, 'nn');
   end

function obj = create_nnet_from_def(lst_def)
   obj = struct();
   obj.num_layers = length(lst_def);

   obj.data_dim = lst_def(1).input_dim;
   obj.cur_epoch = 0;
   obj.tot_batch = 0;

   obj.lst_num_hid = [];
   obj.lst_layer_type = cell(1, obj.num_layers);
   obj.lst_layers = [];
   obj.lst_layer_names = cell(1, obj.num_layers);

   for layer_num=1:obj.num_layers
      layer_def = lst_def(layer_num);
      obj.lst_layer_names{layer_num} = layer_def.name;
      obj.lst_num_hid = [obj.lst_num_hid layer_def.num_units];
      obj.lst_layer_type{layer_num} = layer_def.layer_type;
      obj.lst_layers = [obj.lst_layers layer(layer_def)];
   end
