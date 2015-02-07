"""
Starter code for running/plotting logistic regression.

To run from the command line: python LR.py
"""
import numpy as np
import pylab
from scipy.io import loadmat
from my_code.checkgrad import checkgrad
from my_code.logistic_err import logistic_err

def check_lr_grad():
    nexamples   = 20
    ndimensions = 10

    parameters = {
            'learning_rate'         : 1,
            'weight_regularization' : 0,
            'num_iterations'        : 2000}

    diff = checkgrad(
            logistic_err,
            np.random.randn(ndimensions),
            1e-3,
            np.random.randn(nexamples,ndimensions),
            np.random.rand(nexamples).round(),
            parameters)
    print 'Checkgrad gives: %s' % diff

def train_lr(X_train,y_train,X_valid,y_valid,weights,parameters):
    train_history = []
    valid_history = []

    [f, df, frac_correct_train] = logistic_err(
                    weights,
                    X_train,
                    y_train,
                    parameters)
    train_history.append(frac_correct_train)

    [temp,temp2,frac_correct_valid] = logistic_err(
                    weights,
                    X_valid,
                    y_valid,
                    parameters)
    valid_history.append(frac_correct_valid)

    for t in xrange(parameters['num_iterations']):
        [f, df, frac_correct_train] = logistic_err(
                    weights,
                    X_train,
                    y_train,
                    parameters)
        [temp,temp2,frac_correct_valid] = logistic_err(
                    weights,
                    X_valid,
                    y_valid,
                    parameters)

        train_history.append(frac_correct_train)
        valid_history.append(frac_correct_valid)

        if not np.isfinite(f):
            raise Exception('nan/inf error')

        weights = weights - parameters['learning_rate'] * df

        print ('ITERATION %4i   LOGL:%4.2f   TRAIN FRAC:%2.2f   VALID FRAC:%2.2f'
                    % (t, f, frac_correct_train*100, frac_correct_valid*100))

    return weights, train_history, valid_history

def plot_lr(train_history_noreg,valid_history_noreg,train_history_reg,valid_history_reg):
    pylab.ioff()
    pylab.figure()
    pylab.plot(train_history_noreg,'b')
    pylab.plot(valid_history_noreg,'r')
    pylab.plot(train_history_reg,'--b')
    pylab.plot(valid_history_reg,'--r')
    ax = pylab.gca()
    ax.legend(['Train accuracy noreg','Test accuracy noreg', 'Train accuracy reg', 'Valid accuracy reg'],'lower right')

if __name__ == '__main__':
    # Load the data.
    D = loadmat('data/a1spam.mat')
    D['labels_train'] = D['labels_train'].ravel()
    D['labels_valid']  = D['labels_valid'].ravel()
    D['feature_names'] = D['feature_names'].ravel()

	# Pad 1 to each train/validation data for the bias term
    D['data_train']=np.lib.pad(D['data_train'],((0,0),(0,1)),'constant',constant_values=1)
    D['data_valid']=np.lib.pad(D['data_valid'],((0,0),(0,1)),'constant',constant_values=1)
	
    # Check gradients, make sure that this is a number close to 0.
    check_lr_grad()

    # Train LR first without weight regularization.
    print 'Training logistic regression with no regularization...'
    np.random.seed(1) #seed the random number generator so results are the same each run.

    #YOU WILL NEED TO PUT IN BETTER NUMBERS BELOW
    parameters = {
            'learning_rate'         : 0.1,
            'weight_regularization' : 0,
            'num_iterations'        : 30}

    weights = 0.01*np.random.randn(D['data_train'].shape[1])
    weights, train_history_noreg, valid_history_noreg = train_lr(
            D['data_train'],
            D['labels_train'],
            D['data_valid'],
            D['labels_valid'],
            weights,
            parameters)

    print 'Features for the ten largest weights:'
    print [str(s[0]) for s in D['feature_names'][weights[:-1].argsort()[::-1][:10]]]

    print 'Features for the ten smallest weights:'
    print [str(s[0]) for s in D['feature_names'][weights[:-1].argsort()[:10]]]

    # Now add some regularization.
    print 'Training logistic regression with regularization...'
    np.random.seed(1)
    parameters['weight_regularization'] = 1
    weights = 0.01*np.random.randn(D['data_train'].shape[1])
    weights, train_history_reg, valid_history_reg = train_lr(
            D['data_train'],
            D['labels_train'],
            D['data_valid'],
            D['labels_valid'],
            weights,
            parameters)

    print 'Features for the ten largest weights:'
    print [str(s[0]) for s in D['feature_names'][weights[:-1].argsort()[::-1][:10]]]

    print 'Features for the ten smallest weights:'
    print [str(s[0]) for s in D['feature_names'][weights[:-1].argsort()[:10]]]

    # Plot the training/validation accuracy of each run.
    plot_lr(train_history_noreg,valid_history_noreg,train_history_reg,valid_history_reg)
    pylab.show()
