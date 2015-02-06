function LR()
    addpath('my_code');
    % Load the data.
    D = load('data/a1spam.mat');
    % Check gradients, make sure that this is a number close to 0.
    check_lr_grad();

    % Train LR first without weight regularization.
    % seed the random number generator so results are the same each run.

    fprintf(1, 'Training logistic regression with no regularization...\n');
    
    % For MATLAB R2013a and higher, replace RandStream.getDefaultStream
    % with RandStream.getGlobalStream
    defaultStream = RandStream.getDefaultStream;
    reset(defaultStream, 0);
    
    % YOU WILL NEED TO PUT IN BETTER NUMBERS BELOW
    parameters = struct('learning_rate', 0, ...
                        'weight_regularization', 0, ...
                        'num_iterations', 2000);

    % Padding 1 to each training/validation feature for the bias term
    D.data_train = [D.data_train ones(size(D.data_train,1),1)];
    D.data_valid = [D.data_valid ones(size(D.data_valid,1),1)];
    
    [num_pts, dim] = size(D.data_train);
    weights = 0.01*randn(dim,1);
    [weights, train_history_noreg, valid_history_noreg] = train_lr( ...
            D.data_train, ...
            D.labels_train, ...
            D.data_valid, ...
            D.labels_valid, ...
            weights, ...
            parameters);

    % Ignore the bias term
    weights = weights(1:end-1);
    fprintf(1, 'Features for the ten largest weights:\n');
    [wts_sorted, I] = sort(-weights);
    for i = 1:10
      fprintf(1, '%d. %s\n',i, D.feature_names{I(i)});
    end
    fprintf(1, '\n');

    fprintf(1, 'Features for the ten smallest weights:\n');
    [wts_sorted, I] = sort(weights);
    for i = 1:10
      fprintf(1, '%d. %s\n',i, D.feature_names{I(i)});
    end
    fprintf(1, '\n');

    % Now add some regularization.
    fprintf(1, 'Training logistic regression with regularization...\n');
    reset(defaultStream, 0);
    parameters.weight_regularization = 1;
    weights = 0.01*randn(dim,1);
    [weights, train_history_reg, valid_history_reg] = train_lr(...
            D.data_train,...
            D.labels_train,...
            D.data_valid,...
            D.labels_valid,...
            weights,...
            parameters);

    % Ignore the bias term
    weights = weights(1:end-1);
    fprintf(1, 'Features for the ten largest weights:\n');
    [wts_sorted, I] = sort(-weights);;
    for i = 1:10
      fprintf(1, '%d. %s\n',i, D.feature_names{I(i)});
    end
    fprintf(1, '\n');

    fprintf(1, 'Features for the ten smallest weights:\n');
    [wts_sorted, I] = sort(weights);;
    for i = 1:10
      fprintf(1, '%d. %s\n',i, D.feature_names{I(i)});
    end
    fprintf(1, '\n');

    % Plot the training/validation accuracy of each run.
    plot_lr(train_history_noreg,valid_history_noreg, ...
            train_history_reg,valid_history_reg);
    drawnow();

function check_lr_grad()
  nexamples   = 20;
  ndimensions = 10;
  
  % Here we randomly generate training data and parameters 
  parameters = struct('learning_rate',1, ...
                      'weight_regularization', 0, ...
                      'num_iterations', 2000);

  diff = checkgrad(@logistic_err, randn(ndimensions,1), 1e-3, ...
                   randn(nexamples,ndimensions), round(rand(nexamples,1)), ...
                   parameters);
  fprintf(1, 'Checkgrad gives: %f\n', diff)

function [weights, train_history, valid_history] = train_lr(X_train,y_train, ...
                                             X_valid,y_valid,weights,parameters)
  train_history = [];
  valid_history = [];

  [f, df, frac_correct_train] = logistic_err(...
                    weights,...
                    X_train,...
                    y_train,...
                    parameters);
  train_history = [train_history frac_correct_train];

  [temp,temp2,frac_correct_valid] = logistic_err(...
                    weights,...
                    X_valid,...
                    y_valid,...
                    parameters);
  valid_history = [valid_history frac_correct_valid];

  for t = 1:parameters.num_iterations
    [f, df, frac_correct_train] = logistic_err(...
                    weights,...
                    X_train,...
                    y_train,...
                    parameters);
    [temp,temp2,frac_correct_valid] = logistic_err(...
                    weights,...
                    X_valid,...
                    y_valid,...
                    parameters);

    train_history = [train_history frac_correct_train];
    valid_history = [valid_history frac_correct_valid];

    if isinf(f)
        error('nan/inf error')
    end

    weights = weights - parameters.learning_rate * df;

    fprintf (1, 'ITERATION %4i   LOGL:%4.2f   TRAIN FRAC:%2.2f   VALID FRAC:%2.2f\n', ...
              t, f, frac_correct_train*100, frac_correct_valid*100);
  end

function plot_lr(train_history_noreg,valid_history_noreg,train_history_reg,...
                 valid_history_reg)
    figure()
    hold on ;
    plot(train_history_noreg,'b','LineWidth',1);
    plot(valid_history_noreg,'r','LineWidth',1);
    plot(train_history_reg,'--b','LineWidth',2);
    plot(valid_history_reg,'--r','LineWidth',2);
    legend({'Train accuracy noreg','Test accuracy noreg', ...
            'Train accuracy reg', 'Valid accuracy reg'}, ...
            'Location', 'SouthEast');


