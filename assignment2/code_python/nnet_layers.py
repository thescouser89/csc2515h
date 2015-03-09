# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

from numpy import sqrt, isnan, Inf, dot, zeros, exp, log, sum
from numpy.random import randn

SIGMOID_LAYER = 0
SOFTMAX_LAYER = 1

class layer_definition(object):
    def __init__(self, name, layer_type, input_dim, num_units, wt_sigma):
        self.name, self.layer_type, self.input_dim, self.num_units, \
          self.wt_sigma  =  name, layer_type, input_dim, num_units, wt_sigma


class layer(object):
    def __init__(self, name):
        self.name = name

    @property
    def shape(self):
        return self._wts.shape

    @property
    def num_hid(self):
        return self._wts.shape[1]

    @property
    def num_dims(self):
        return self._wts.shape[0]

    def create_params(self, layer_def):
        input_dim, output_dim, wt_sigma = layer_def.input_dim, \
                        layer_def.num_units, layer_def.wt_sigma

        self._wts = randn(input_dim, output_dim) * wt_sigma
        self._b = zeros((output_dim, 1))

        self._wts_grad = zeros(self._wts.shape)
        self._wts_inc = zeros(self._wts.shape)

        self._b_grad = zeros(self._b.shape)
        self._b_inc = zeros(self._b.shape)

        self.__num_params = input_dim*output_dim

    def add_params_to_dict(self, params_dict):
        params_dict[self.name + "_wts"] = self._wts.copy()
        params_dict[self.name + "_b"] = self._b.copy()

    def copy_params_from_dict(self, params_dict):
        self._wts = params_dict[self.name + "_wts"].copy()
        self._b = params_dict[self.name + "_b"].copy()
        self.__num_params = self._wts.shape[0] * self._wts.shape[1]
        self._wts_inc = zeros(self._wts.shape)
        self._b_inc = zeros(self._b.shape)

    def apply_gradients(self, momentum, eps, l2=.0001):
        """ NEED TO IMPLEMENT

        update wts_inc(b_inc) and use wts_inc(b_inc) to update the weight
        (bias). You may want the gradient wts_grad(b_grad) as well as momentum
        and learning rate.

        TODO: apply linear regularizer later
        """
        # ==== IMPLEMENTED =====================================================
        self._b_inc = eps * (self._b_grad) - momentum * (self._b_inc)
        self._wts_inc = eps * (self._wts_grad) - momentum * (self._wts_inc)
        # now that we have found the wts_grad, let's update the bias and weight
        # itself
        self._b = self._b - self._b_inc
        self._wts = self._wts - self._wts_inc
        # ======================================================================

    # ==========================================================================
    # act_grad      :: gradient wrt activation function of this layer
    # input_grad    :: gradients wrt the input of this layer
    # ==========================================================================
    def back_prop(self, act_grad, data):
        '''
        NEED TO IMPLEMENT.
        Feel free to add member variables.
        back prop activation grad, and compute gradients.

        Back propagate activation gradients and compute gradients for one layer.
        The output is a struct consisting of 3 parts, wts_grad, b_grad,
        input_grad

        NEED TO FIND WTS_GRAD HERE and update the self object.

        data is the layer input

        input grad is dE_dyi
        act_grad is dE_dxj
        '''
        dE_dxj = act_grad
        dE_wij = data.dot(act_grad.T)
        self._wts_grad = dE_wij
        self._b_grad = dE_dxj

        input_grad = self._wts.dot(dE_dxj)

        return input_grad


# ==============================================================================
# self.wts :: weights for each layer
# b        :: bias for each layer
# wts_grad :: gradient for weights you calculated from back_prop for each layer
# wts_inc  :: actual update you will do for wts in a SGD step for each layer
# b_grad   :: gradient for bias you calculated from back_prop for each layer
# b_inc    :: actual update you will do for b in a SGD for each layer
# ==============================================================================
class sigmoid_layer(layer):
    pass

    def fwd_prop(self, data):
        """ NEED TO IMPLEMENT

        Perform a forward pass
        """
        # data is (345, 1)
        # _wts is (345, 300)
        # z = data.T * _wts => (1, 300)
        print data.T.shape
        z = data.T.dot(self._wts)
        z = self._wts.T.dot(data) + self._b

        # sigmoid :: use logistic regression
        sigmoid = 1 / (1 + exp(-z))
        # we want to return a column vector, not a row vector
        return sigmoid

    def compute_act_grad_from_output_grad(self, output, output_grad):
        """ NEED TO IMPLEMENT

        Compute the gradients wrt activations of sigmoid layer, the input are
        the current activations of this layer and the gradients wrt outputs of
        the sigmoid.
        """
        yj = output
        dE_dyj = output_grad

        act_grad = yj * (1 - yj) * dE_dyj
        return act_grad


class softmax_layer(layer):
    pass

    def fwd_prop(self, data):
        """ NEED TO IMPLEMENT

        Perform a forward pass

        weight is (300, 44)
        data is (300, 1)
        """
        z = self._wts.T.dot(data) + self._b
        top_part = exp(z)
        bottom_part = sum(top_part)
        result = top_part / bottom_part
        return result

    def compute_act_gradients_from_targets(self, targets, output):
        """ NEED TO IMPLEMENT

        Compute the gradients wrt activations of the softmax layer, given the
        targets and the outputs of the softmax, the inputs are the current
        activations of this layer and the target.
        """
        act_grad = output * (1 - output) * (output - targets)
        return act_grad


    @staticmethod
    def compute_accuraccy(probs, label_mat):
        num_correct = sum(probs.argmax(axis=0) == label_mat.argmax(axis=0))
        log_probs = sum(log(probs) * label_mat)
        return num_correct, log_probs

def create_empty_nnet_layer(name, layer_type):
    if layer_type == SIGMOID_LAYER:
        layer = sigmoid_layer(name)
    elif layer_type == SOFTMAX_LAYER:
        layer = softmax_layer(name)
    else:
        raise Exception, "Unknown layer type"
    return layer

def create_nnet_layer(layer_def):
    layer = create_empty_nnet_layer(layer_def.name, layer_def.layer_type)
    layer.create_params(layer_def)
    return layer
