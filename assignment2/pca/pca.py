import numpy as np
import scipy.io

from matplotlib.mlab import PCA
import sys
sys.path.append('../knn')
import knn

def pca(matrix):

    matrix_T = matrix.T
    normalized_matrix = (matrix_T - matrix_T.mean(axis=0)) / matrix_T.std(axis=0)
    eigen_val, eigen_vec = np.linalg.eig(np.cov(normalized_matrix.T))
    eigen_vec = eigen_vec[:, np.argsort(eigen_val)[::-1]].T
    return eigen_vec

if __name__ == '__main__':
    mat = scipy.io.loadmat('../knn_subset.mat')

    train_data = mat['train_data']          # (23, 4400)
    train_target = mat['train_targets']     # (4400, 1)

    test_data = mat['test_data']            # (23, 2200)
    test_targets = mat['test_targets']      # (2200, 1)

    V = pca(train_data)

    new_data = V[:20].dot(train_data)
    print new_data.shape

    number_test_data = test_data.shape[1]
    guessed_right = 0

    test_data = V[:20].dot(test_data)
    print test_data.shape
    test_data_T = test_data.T
    for i in range(number_test_data):
        guessed_target = knn.find_neighbors(test_data_T[i], 1, new_data, train_target)
        if guessed_target == test_targets[i]:
            guessed_right += 1

    print guessed_right
