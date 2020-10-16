clear; 
clc; rng(10);
f = genpath(pwd); addpath(f);
warning('off');
addpath('./utils');
echo anfis off;
data_name = 'kc_house';
load_dir = sprintf('./data_norm/%s.mat', data_name);

% Preprocess dataset and select parameters for the simulation
kfolds = 5;                % Number of folds
n_rules = 5;
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
gamma = 0.1;
% parameters for Laplacian funciton
rho_s=0.1;
rho_p=0.1;
semi_fnn_fg_converge_analysis(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
    gamma,eta, alpha)

