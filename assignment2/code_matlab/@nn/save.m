function out = save(self, file_name)
    params_dict = struct();
    params_dict.data_dim = self.data_dim;

    num_layers = length(self.lst_layers);
    layer_names = cell(1, num_layers);

    for layer_num=1:num_layers
       layer = self.lst_layers(layer_num);
       layer_names{layer_num} = layer.name;
       params_dict.(strcat(layer.name, '_wts')) = layer.wts;
       params_dict.(strcat(layer.name, '_b')) = layer.b;
    end
    params_dict.lst_layer_names = layer_names;
    params_dict.lst_layer_type = self.lst_layer_type;
    params_dict.lst_num_hid = self.lst_num_hid;

    save(file_name, '-struct', 'params_dict') ;
