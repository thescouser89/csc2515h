function self = layer(layer_def)
   obj = create_nnet_layer(layer_def);
   self = class(obj, 'layer');
end

function obj = create_nnet_layer(layer_def)
   input_dim  = layer_def.input_dim;
   num_units = layer_def.num_units;
   wt_sigma   = layer_def.wt_sigma;

   wts      = randn(input_dim, num_units) .* wt_sigma;
   b        = zeros(num_units,1);
   wts_grad = zeros(size(wts));
   wts_inc  = zeros(size(wts));
   b_grad   = zeros(num_units,1);
   b_inc    = zeros(num_units,1);

   obj = struct('wts', wts, 'b', b, 'wts_grad', wts_grad, 'b_grad', b_grad, ...
                'wts_inc', wts_inc, 'b_inc', b_inc, 'name', layer_def.name,...
                'layer_type', layer_def.layer_type);

end
