% SCN - Classification
clear;
clc;
close all; 
format long;
load('Iris_Data.mat');
 
%% show some samples
% X: each row vector represents one sample input.
% T: each row vector represents one sample label.
% same to the X2 and T2.
n1 = randperm(120, 5); % Randomly select 5 samples from training set.
disp('Training Input:')
disp(X(n1,:))
disp('Training Label:')
disp(T(n1,:))

n2 = randperm(30, 2);  % Randomly select 2 samples from test set.
disp('Test Input:')
disp(X2(n2,:))
disp('Test Label:')
disp(T2(n2,:))

%% Parameter Setting
L_max = 40;                    % maximum hidden node number
tol = 0.01;                    % training tolerance
T_max = 100;                    % maximun candidate nodes number
Lambdas = [0.5, 1, 5, 10, 30, 50,100];% scope sequence
r =  [ 0.9, 0.99, 0.999, ...
    0.9999, 0.99999, 0.999999]; % 1-r contraction sequence
nB = 1;       % batch size

%% Model Initialization
M = SCN(L_max, T_max, tol, Lambdas, r , nB);

%% Model Training
% M is the trained model
% per contains the training error and accuracy with respect to the increasing L
[M, per] = M.Classification(X, T);
disp(M);

%% Training error and accuracy demo
figure;
yyaxis left;
plot(per.Rate, 'b.-'); hold on;
ylabel('Accuracy');
yyaxis right;
plot(per.Error, 'r.-'); 
xlabel('L');
ylabel('RMSE');
xlabel('L');
legend( 'Training ACC', 'Training RMSE');

%% Model output vs target on training dataset
O1 = M.GetLabel(X);
disp(['Training Acc: ' num2str(M.GetAccuracy(X, T))]);
figure;
plotconfusion(T',O1','Training  ');

%% Model output vs target on test dataset
O2 = M.GetLabel(X2);
disp(['Test Acc: ', num2str(M.GetAccuracy(X2, T2))]);
figure;
plotconfusion(T2',O2','Test  ');

% The End










