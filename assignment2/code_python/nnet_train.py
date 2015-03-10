# Code by Navdeep Jaitly, 2013.
# Email: ndjaitly@gmail.com

from nnet_layers import *
import sys, logging, os, scipy.io
from numpy import zeros, savez, log

# create logger
logger = logging.getLogger('nnet_train')
logger.setLevel(logging.INFO)
# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


class nn(object):
    def __init__(self):
        pass

    def save(self, file_name):
        params_dict = {}
        params_dict['lst_layer_names'] = [layer.name for layer in self._lst_layers]
        params_dict['lst_layer_type'] = self._lst_layer_type
        params_dict['lst_num_hid'] = self._lst_num_hid
        params_dict['data_dim'] = self._data_dim

        for layer in self._lst_layers:
            layer.add_params_to_dict(params_dict)

        scipy.io.savemat(file_name, params_dict)
        #util.save(file_name, " ".join(params_dict.keys()), params_dict)

    def load(self, file_name):
        params_dict = scipy.io.loadmat(file_name)
        self._lst_layer_type, self._lst_num_hid, self._data_dim = \
                                           params_dict['lst_layer_type'], \
                                           params_dict['lst_num_hid'], \
                                           params_dict['data_dim']

        if not hasattr(self, '_lst_layers'):
            logging.info("Creating new layers from parameters in file: %s"%file_name)
            self._lst_layers = []
            for (layer_name, layer_type) in zip(params_dict['lst_layer_names'],
                                                self._lst_layer_type[0]):
                layer = create_empty_nnet_layer(layer_name, layer_type)
                layer.copy_params_from_dict(params_dict)
                self._lst_layers.append(layer)
        else:
            logging.info("Updating layer parameters using file: %s"%file_name)
            for layer in self._lst_layers:
                layer.copy_params_from_dict(params_dict)

        self.num_layers = len(self._lst_layers)


    def get_num_layers(self):
        return len(self._lst_layers)

    def get_code_dim(self):
        return self._lst_num_hid[-1]

    def create_nnet_from_def(self, lst_def):
        self._layers = []
        self.num_layers = len(lst_def)

        self._data_dim = lst_def[0].input_dim

        self._lst_num_hid = []
        self._lst_layer_type = []
        self._lst_layers = []

        for layer_num, layer_def in enumerate(lst_def):
            self._lst_num_hid.append(layer_def.num_units)
            self._lst_layer_type.append(layer_def.layer_type)
            self._lst_layers.append(create_nnet_layer(layer_def))


    def fwd_prop(self, data):
        """
        NEED TO IMPLEMENT.
        Return list of outputs per layer.

        Do a pass over all layers, and return a list of activations for each
        layer. You may want to call layer.fwd_prop for each layer
        """
        lst_layer_outputs = []
        # data is (345, 1)

        previous_activation = data
        lst_layer_outputs.append(previous_activation)
        for layer in self._lst_layers:
            previous_activation = layer.fwd_prop(previous_activation)
            lst_layer_outputs.append(previous_activation)

        return lst_layer_outputs


    def back_prop(self, lst_layer_outputs, data, targets):
        """
        NEED TO IMPLEMENT

        Perform a backpropagation, return 'self' with updated gradient of
        weights and biases for all layers. You may want to call layer.back_prop
        for each layer.

        lst_layer_outputs is from fwd_prop output
        """

        input_grad = 0

        for index, layer in enumerate(reversed(self._lst_layers), start=1):
            last_layer_output = lst_layer_outputs[-index]
            last_layer_input = lst_layer_outputs[-(index + 1)]
            # consider act_grad as dE_dxj
            if isinstance(layer, softmax_layer):    # only for last layer
                act_grad = layer.compute_act_gradients_from_targets(targets, last_layer_output)
            else:   # for everything else
                act_grad = layer.compute_act_grad_from_output_grad(last_layer_output, input_grad)
            input_grad = layer.back_prop(act_grad, last_layer_input)


    def apply_gradients(self, eps, momentum, l2=0):
        """
        NEED TO IMPLEMENT

        Perform stochastic gradient descent step. You may want to call
        layer.apply_gradients for each layer.
        """
        for layer in self._lst_layers:
            layer.apply_gradients(momentum, eps, l2)


    def create_predictions(self, data_src):
        """
        Function used to create predictions from acoustics.
        """
        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0

        pred_lst = []
        num_output_frames = 0
        for utt_num  in range(data_src.get_num_utterances()):
            data = data_src.get_utterance_data(utt_num,
                                               get_labels=False)
            lst_layer_outputs = self.fwd_prop(data)
            pred_lst.append(log(1e-32 + lst_layer_outputs[-1]))
            num_output_frames += pred_lst[-1].shape[1]


        return pred_lst, num_output_frames


    def test(self, data_src):
        """
        Function used to test accuracy.
        """
        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0

        for utt_num  in range(data_src.get_num_utterances()):
            data, label_mat = data_src.get_utterance_data(utt_num)
            num_pts += data.shape[1]
            lst_layer_outputs = self.fwd_prop(data)
            num_correct, log_prob = self._lst_layers[-1].compute_accuraccy(\
                                            lst_layer_outputs[-1], label_mat)
            classif_err_sum += (data.shape[1] - num_correct)
            lg_p_sum += log_prob

        classif_err = classif_err_sum*100./num_pts
        logging.info("TESTING Classif Err = %.3f, lg(p) %.4f\n"%(\
                         classif_err, lg_p_sum*1./num_pts))
        sys.stderr.write("TESTING Classif Err = %.3f, lg(p) %.4f\n"%(\
                         classif_err, lg_p_sum*1./num_pts))
        sys.stderr.flush()
        ch.flush()

        return 100 - classif_err, lg_p_sum*1./num_pts

    def train_for_one_epoch(self, data_src, eps, momentum, l2, batch_size):
        '''
        Work horse of the learning for one epoch. As long as the other
        functions are working correctly, and satisfy the interface,
        there should be no need to change this function.
        '''
        try:
            self.__cur_epoch += 1
        except AttributeError:
            self.__cur_epoch = 1

        try:
            self._tot_batch
        except AttributeError:
            self._tot_batch = 0

        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0
        batch = 0

        for  (data, label_mat) in data_src.get_iterator(batch_size):
            batch += 1
            num_pts += batch_size
            lst_layer_outputs = self.fwd_prop(data)
            num_correct, log_prob = self._lst_layers[-1].compute_accuraccy(\
                                            lst_layer_outputs[-1], label_mat)
            classif_err_sum += (data.shape[1] - num_correct)
            lg_p_sum += log_prob

            self.back_prop(lst_layer_outputs, data, label_mat)
            self.apply_gradients(eps, momentum, l2)
            self._tot_batch += 1


        classif_err = classif_err_sum*100./num_pts
        logging.info("Epoch = %d, batch = %d, Classif Err = %.3f, lg(p) %.4f"%(\
                   self.__cur_epoch, batch, classif_err, lg_p_sum*1./num_pts))
        sys.stderr.write("Epoch = %d, batch = %d, Classif Err = %.3f, lg(p) %.4f\n"%(\
                   self.__cur_epoch, batch, classif_err, lg_p_sum*1./num_pts))
        sys.stderr.flush()
        ch.flush()

