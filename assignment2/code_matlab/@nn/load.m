function self = load(self, file_name)
   params_dict = builtin('load', file_name)
   self.lst_layer_type = params_dict.lst_layer_type;
   self.lst_layer_names = params_dict.lst_layer_names;
   self.lst_num_hid = params_dict.lst_num_hid;
   self.data_dim = params_dict.data_dim;
   self.num_layers = length(self.lst_num_hid);

   if true | isfield(self, 'lst_layers')
      fprintf(1, 'Creating new layers from parameters in file: %s', file_name);
      lst_layers = [];
      input_dim = self.data_dim;
      for layer_num=1:self.num_layers
         layer_type = params_dict.lst_layer_type(layer_num);
         layer_name = params_dict.lst_layer_names{layer_num};
         layer_def = layer_definition(layer_name, layer_type, input_dim, ...
                                      self.lst_num_hid(layer_num), 0);
         lyr = layer(layer_def);
         lyr.wts = getfield(params_dict, strcat(layer_name, '_wts'));
         lyr.b = getfield(params_dict, strcat(layer_name, '_b'));
         lst_layers = [lst_layers lyr];
         input_dim = self.lst_num_hid(layer_num)
      end
      self.lst_layers = lst_layers;
   else
      fprintf(1, 'Updating layer parameters using file: %s\n', file_name);
      for layer_num=1:self.num_layers
         layer_name = self.lst_layer_names{layer_num};
         self.lst_layers(layer_num).wts = getfield(params_dict, ...
                                                   strcat(layer_name, '_wts'));
         self.lst_layers(layer_num).b = getfield(params_dict, ...
                                                 strcat(layer_name, '_b'));
      end
   end

