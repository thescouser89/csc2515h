import numpy as np
import scipy.io

def euclidean_distance(vec_a, vec_b):
    return numpy.linalg.norm(vec_a - vec_b)

def find_neighbors(vec_to_find, k, train_data, train_target):
    """ Find target of vec_to_find
    vec_to_find: (23, ) vector, where we are supposed to find its target
    k: [int] number of neighbors to check
    train_data: (23, x) array
    train_target: (x, 1) array

    1. Given vec_to_find, calculate the distance between it and all the
       train_data
    2. Sort the distances
    3. Select the 'k' shortest distances
    4. Use the mode to find the most-voted target
    """
    pass


mat = scipy.io.loadmat('../knn_subset.mat')

# I'm assuming each row is a dimension, column is the number of data
train_data = mat['train_data']          # (23, 4400)
train_target = mat['train_targets']     # (4400, 1)

test_data = mat['test_data']            # (23, 2200)
test_targets = mat['test_targets']      # (2200, 1)
