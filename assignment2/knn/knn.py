import numpy as np
import scipy.io
from scipy.stats import mode

def euclidean_distance(vec_a, vec_b):
    return np.linalg.norm(vec_a - vec_b)

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

    return most-voted target[1]
    """
    train_data_T = train_data.T

    number_train_data = train_data.shape[1]

    train_data_distance = []

    for i in range(number_train_data):
    	distance = euclidean_distance(vec_to_find, train_data_T[i])
    	# Add to the list
    	# (distance, target)
    	train_data_distance.append((distance, train_target[i]))

    # sort the data
    train_data_distance.sort(key=lambda tup: tup[0])

    # k shortest distances
    k_shortest_distances = train_data_distance[:k]
    k_targets = [target[1] for target in k_shortest_distances]
    return mode(k_targets)[0][0]


mat = scipy.io.loadmat('../knn_subset.mat')

# I'm assuming each row is a dimension, column is the number of data
train_data = mat['train_data']          # (23, 4400)
train_target = mat['train_targets']     # (4400, 1)

test_data = mat['test_data']            # (23, 2200)
test_targets = mat['test_targets']      # (2200, 1)


number_test_data = test_data.shape[1]

test_data_T = test_data.T

guessed_right = 0

print number_test_data
for i in range(number_test_data):

	guessed_target = find_neighbors(test_data_T[i], 1, train_data, train_target)
	if guessed_target == test_targets[i]:
		guessed_right += 1

print guessed_right