function NB()
  D = load('data/a1spam.mat');

  addpath('my_code');
  
  [num_pts, dim] = size(D.data_train);
  X = [D.data_train; zeros(2,dim); ones(2,dim)];
  y = [D.labels_train; 0; 1; 0; 1];

  [p_y, p_x_given_y] = train_nb(X, y);
  yhat_train = nb_decision(D.data_train, p_y, p_x_given_y);
  yhat_valid = nb_decision(D.data_valid, p_y, p_x_given_y);

  train_acc = mean(yhat_train == D.labels_train);
  valid_acc = mean(yhat_valid == D.labels_valid);

  fprintf(1, 'TRAIN ACC:%4.2f VALID ACC:%4.2f',train_acc,valid_acc);
  p_y
 
function [p_y, p_x_given_y] = train_nb(X,y)
  [num_pts, dim] = size(X);
  n1 = sum(y);
  n0 = num_pts - n1;

  p_x_given_y = zeros(2,dim);
  p_y = [n0/num_pts n1/num_pts];


  p_x_given_y(1,:) = sum(X .* repmat(y==0,1,dim), 1) / n0;
  p_x_given_y(2,:) = sum(X .* repmat(y==1,1,dim), 1) / n1;
