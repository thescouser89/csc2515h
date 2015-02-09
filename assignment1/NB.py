"""
Starter code for running naive Bayes.
This code trains a naive Bayes model, you are responsible
for coding the decision rule.

To run from the command line: python NB.py
"""
import numpy as np
from scipy.io import loadmat
from my_code.nb_decision import nb_decision

def train_nb(X,y):
    X = np.vstack((X,np.zeros((2,X.shape[1])),np.ones((2,X.shape[1]))))
    y = np.hstack((y,np.array([0,1,0,1])))
    p_x_given_y = np.zeros((X.shape[1],2))
    p_y = np.zeros(2)

    p_y[1] = np.mean(y)
    p_y[0] = 1 - p_y[1]

    n1 = y.sum()
    n0 = y.shape[0] - n1

    p_x_given_y[:,0] = (X*(y==0)[:,np.newaxis]).sum(0) / n0
    p_x_given_y[:,1] = (X*(y==1)[:,np.newaxis]).sum(0) / n1

    return p_y, p_x_given_y

if __name__ == '__main__':
    D = loadmat('data/a1spam.mat')
    D['labels_train'] = D['labels_train'].ravel()
    D['labels_valid']  = D['labels_valid'].ravel()
    D['feature_names'] = D['feature_names'].ravel()

    p_y,p_x_given_y = train_nb(D['data_train'],D['labels_train'])

    probability = p_y * p_x_given_y

    k = np.sum(probability, 1)
    print 'max'
    print [str(s[0]) for s in D['feature_names'][k.argsort()[::-1][:10]]]
    print 'min'
    print [str(s[0]) for s in D['feature_names'][k.argsort()[:10]]]

    yhat_train = nb_decision(D['data_train'],p_y,p_x_given_y)
    yhat_valid = nb_decision(D['data_valid'],p_y,p_x_given_y)


    train_acc = np.mean(yhat_train == D['labels_train'])
    valid_acc = np.mean(yhat_valid == D['labels_valid'])

    print 'TRAIN ACC:%4.2f   VALID ACC:%4.2f' % (train_acc,valid_acc)

