lear; 
clc; rng(10);
f = genpath(pwd); addpath(f);
warning('off');
addpath('./utils');
echo anfis off;
data_name = 'ad5_10_5000_s';

% Preprocess dataset and select parameters for the simulation
kfolds = 5;                % Number of folds
n_rules = 10;
n_agent = 5;

%parameter for fuzzy cmeans
beta = 1.1;

labeled_num = 50;

% parameter for beta distribution
alpha = 0.75;
% number of unlabeled data chosen for mix-up
n_mixup = 10000;


% parameter for admm fnn
mu = 0.1;
% parameter for graph term
eta=0.0001;
% parameter for mix-up term
gamma = 1;


semi_fnn_fg_func_c(data_name, kfolds, n_rules, n_mixup, labeled_num, mu, gamma, eta, alpha, beta)


% parameters for Laplacian funciton
rho_s=0.1;
rho_p=0.1;
% semi_fnn_fg_func_d(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
%     gamma,eta, alpha)

% [result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, time_um]=...
%         semi_fnn_fg_func_d_um(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
%     gamma,alpha);
% fprintf("Train acc using mix-up: %.4f/%.4f  %.4f\n", result_train_mean_um, result_train_std_um, time_um);
% fprintf("Test acc using mix-up: %.4f/%.4f  %.4f\n", result_test_mean_um, result_test_std_um, time_um);

