# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

import numpy
numpy.random.seed(42) # This seed is meaningful. :).

import nnet_train, nnet_layers, sys, os, argparse, speech_data
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--seed', dest='seed', type=int,
                    default=-1, help='Seed for random number generators')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='Maximum number of epcohs to train for')
parser.add_argument('train_db_path', help='Path to training database')
parser.add_argument('dev_db_path', help='Path to validation database')
parser.add_argument('output_fldr', type=str, help='output folder')

arguments = parser.parse_args()
if not os.path.exists(arguments.output_fldr):
    os.makedirs(arguments.output_fldr)
logPath = os.path.join(arguments.output_fldr, "log.txt")
if os.path.exists(logPath): os.remove(logPath)

# create logger
logging.basicConfig(filename=logPath, level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("python " + " ".join(sys.argv))

# Number of contiguous frames to predict phoneme labels. Leave at 15
num_frames_per_pt = 15


# DECIDE ON MEANINGFUL VALUES FOR THESE VARIABLES
# ==============================================================================
# eps: learning rate for SGD
# l2:  coefficient for L2 regularizer
# momentum: momentum for SGD
# batch_size: batch size for SGD
# ==============================================================================
batch_size, eps, momentum, l2 = 10, 0.3, 0.2, 0


train_src = speech_data.speech_data(arguments.train_db_path, num_frames_per_pt)
validation_src = speech_data.speech_data(arguments.dev_db_path, 
                                         num_frames_per_pt)
layer1_def = nnet_layers.layer_definition("Layer1", nnet_layers.SIGMOID_LAYER,
                                         train_src.get_data_dim(), 500, 0.01)
layer2_def = nnet_layers.layer_definition("Layer2", nnet_layers.SOFTMAX_LAYER,
                                         500, train_src.get_target_dim(), 0.01)

## FILL IN SOME PRE-PROCESS  INSTRUCTIONS HERE
train_src.normalize_data()
validation_src.normalize_data()



# Definition of multi layer neural network
lst_def = [layer1_def, layer2_def] 

nn_train = nnet_train.nn()
nn_train.create_nnet_from_def(lst_def)

# where the model gets written to.
param_file = os.path.join(arguments.output_fldr, "model.mat")

best_dev_accuracy = 0
for i in range(arguments.max_epochs):
    nn_train.train_for_one_epoch(train_src, eps, momentum, l2, batch_size)
    accuracy, lg_p = nn_train.test(validation_src)
    # CODE FOR CONTROLLING OVERFITTING...GOES SOMEWHERE AROUND HERE..
    if best_dev_accuracy + 0.01 > accuracy:
        break

    best_dev_accuracy = accuracy

#nn_train.save(param_file)
