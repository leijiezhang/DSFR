clear; close all; clc;

%% ��������
load abalone.mat
% load sinc_data
[n,m] = size(X);
p = randperm(n);
x = X(p,:);
y = Y(p,:);


%% �������
l = 100;                                                            % ���ڵ����
K = 10000;                                                        % ��������

lambda = 1e-3;
eta = 1e-5;
gamma = 1e-4;

a = rand(m,l);
b = rand(m,l);
c = x(p(1:l),:)';

delta = 1.0e-10;                      % delta: a small decimal for judgment

lab_per = 0.012;                          % percentage of the labeled data
n_lab = floor(n*lab_per);       % number of labeled data

DOS = {'Node 1','Node 2','Node 3','Node 4','Central'};     % �ֲ�ʽ�Ż��㷨
DOS_num = numel(DOS);

NRMSE = zeros(DOS_num,K);

color_tab = [0, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
    0.4660, 0.6740, 0.1880;
    0.3010, 0.7450, 0.9330;
    0.6350, 0.0780, 0.1840];

%% ����ʽ
H = zeros(n,l);
for i = 1:l
    H(:,i) = RBFN(x,a(:,i),b(:,i),c(:,i));    % SLFNN
end

H_lab = H(1:n_lab,:);
W_Lap = squareform(pdist(x));
W_Lap = exp(-W_Lap.^2/2);
D_Lap = diag(sum(W_Lap,2));
Lap = D_Lap - W_Lap;
D_Lap(D_Lap>0) = 1./sqrt(D_Lap(D_Lap>0));
Lap = D_Lap^(-1/2)*Lap*D_Lap^(-1/2);
Lap = D_Lap*Lap*D_Lap;
W_C_lab = (H_lab'*H_lab + lambda*eye(l) + eta*H'*Lap*H )^(-1)*H_lab'*y(1:n_lab);
Y_C_lab = H*W_C_lab;

W_C = (H'*H + lambda*eye(l))^(-1)*H'*y;
Y_C = H*W_C;

NRMSE_C = calculate_nrmse(y,Y_C_lab);
% NRMSE_C = log(NRMSE_C);

A = [0 1 1 1;
    1 0 0 0;
    1 0 0 1;
    1 0 1 0];
d = max(sum(A));
L = diag(sum(A)) - A;
V = size(A,1);

%%  ��ʼ��
ni = floor(n/V);
W0 = zeros(l,V);
Hinv = cell(V,1);
Hi = cell(V,1);
Yi = cell(V,1);
ni_lab = floor(ni*lab_per);

for i = 1:V
    idx_i = (i-1)*ni+1:i*ni;
    xi = x(idx_i,:);
    yi = y(idx_i);
    
    Hi{i} = H(idx_i,:);
    Yi{i} = yi;
    
    Hi_lab = Hi{i}(1:ni_lab,:);
    W_l = squareform(pdist(xi));
    W_l = exp(-W_l.^2/2);
    D_l = diag(sum(W_l,2));
    Lap = D_l - W_l;
    Hinv{i} = (Hi_lab'*Hi_lab + lambda*eye(l) + eta*Hi{i}'*Lap*Hi{i} )^(-1);
    W0(:,i) = Hinv{i}*(Hi_lab'*yi(1:ni_lab));
end

%% D-Lap

% ��ʼ��
W0_dac = W0;
W_dac = W0;

W_admm = W0;
t0 = W0 - W0;
z0 = t0(:,1);
t = t0;
z = z0;

W0_zgs = W0;
W_zgs = W0;

W0_dlms = W0 - W0;
W_dlms = W0_dlms;

%% ����
for k = 1:K
    
    W_D_Lap = W_zgs(:,1);
    Y_D_Lap = H*W_D_Lap;
    % NRMSE(1,k) = log(calculate_nrmse(Y_D_Lap,y));
    NRMSE(1,k) = calculate_nrmse(y,Y_D_Lap);
    
    W_D_Lap = W_zgs(:,2);
    Y_D_Lap = H*W_D_Lap;
    % NRMSE(2,k) = log(calculate_nrmse(Y_D_Lap,y));
    NRMSE(2,k) = calculate_nrmse(y,Y_D_Lap);
    
    W_D_Lap = W_zgs(:,3);
    Y_D_Lap = H*W_D_Lap;
    % NRMSE(3,k) = log(calculate_nrmse(Y_D_Lap,y));
    NRMSE(3,k) = calculate_nrmse(y,Y_D_Lap);
    
    W_D_Lap = W_zgs(:,4);
    Y_D_Lap = H*W_D_Lap;
   %  NRMSE(4,k) = log(calculate_nrmse(Y_D_Lap,y));
    NRMSE(4,k) = calculate_nrmse(y,Y_D_Lap);
    
    %% Central
    NRMSE(5,k) = NRMSE_C;
    
    for i = 1:V
        W_zgs(:,i) = W0_zgs(:,i) - gamma*Hinv{i}*W0_zgs*L(:,i);
    end
    W0_zgs = W_zgs;
end

% save('.\Result\Concrete\Log_NRMSE.mat','NRMSE');

line_width = 3;
font_size = 20;

plot(1:K,NRMSE,'linewidth',3);
% hold on
% for i = 1:DOS_num-1
%     plot(1:K,NRMSE(i,:),'color',color_tab(i,:),'linewidth',line_width);
% end
set(gca,'FontSize',font_size);
xlabel('Iteration number','fontsize',font_size);
ylabel('NRMSE','fontsize',font_size);
legend(DOS);
% set(h,'fontsize',font_size)
grid on
box on
hold off
savefig(gcf,'.\Fig\Concrete\Log_NRMSE.fig');


