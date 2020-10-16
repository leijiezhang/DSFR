function [result_train_mean_ug, result_train_std_ug, result_test_mean_ug, result_test_std_ug, time_ug]=...
        semi_fnn_fg_func_c_ug(data_name, kfolds, n_rules, labeled_num, mu, eta, beta, fid)
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);
%     alpha = 0.75;
    % load('concrete.mat')

    % Preprocess dataset and select parameters for the sietalation

    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

    result_test_ug_arr = zeros(1,kfolds); 
    result_train_ug_arr = zeros(1,kfolds);

    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data(i);
        train_data.n_smpl_l = labeled_num;
        shuffle_idx = randperm(train_data.n_smpl);
        train_data.fea_l = train_data.fea(shuffle_idx(1:train_data.n_smpl_l),:);
        train_data.gnd_l = train_data.gnd(shuffle_idx(1:train_data.n_smpl_l));
        train_data.fea_u = train_data.fea(shuffle_idx(train_data.n_smpl_l+1:end),:);
        train_data.gnd_u = train_data.gnd(shuffle_idx(train_data.n_smpl_l+1:end));
        train_data.n_smpl_u = size(train_data.fea_u,1);

        % ==============all data involved using graph============% labeled data and unlabeled data
        tic;
        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(train_data.fea, n_rules,beta);
        H_train_g = ComputeH(train_data.fea,rule_g);
        rule_lg = rule_g;
        rule_lg = rule_lg.update_fuzzyc(train_data.fea_l, rule_g.center, beta);
        H_train_lg = ComputeH(train_data.fea_l,rule_lg);
        dist_tmp = EuDist2(train_data.fea);
        t_tmp = mean(mean(dist_tmp));
        W_tmp = exp(-dist_tmp/(2*t_tmp^2));
        D_tmp = diag(sum(W_tmp,2));
        L_tmp = D_tmp - W_tmp;
        D_half = D_tmp^(-1/2);
        L_hat = D_half*L_tmp*D_half;
        [~, p] = size(H_train_g);
        Hinv = inv(eye(p)*mu  + H_train_lg' * H_train_lg + eta*H_train_g'*L_hat*H_train_g);
%       
        HY = H_train_lg'*train_data.gnd_l;
        beta_ug = Hinv*HY;
        time_ug = toc;
        y_hat_train_ug = H_train_lg * beta_ug;
        if task=='C'
            rslt_train_ug = calculate_acc(train_data.gnd_l, y_hat_train_ug);
            if nargin == 8
                fprintf("==>Train acc using graph: %f\n", rslt_train_u);
                fprintf(fid, "==>Train acc using graph: %f\n", rslt_train_u);
            end
            
        else
            rslt_train_ug = calculate_nrmse(train_data.gnd_l, y_hat_train_ug);
            if nargin == 8
                fprintf("==>Train NRMSE using graph: %f\n", rslt_train_ug);
                fprintf(fid, "==>Train NRMSE using graph: %f\n", rslt_train_ug);
            end
            
        end
        rule_lg = rule_lg.update_fuzzyc(test_data.fea, rule_lg.center, beta);
        H_test_lg = ComputeH(test_data.fea,rule_lg);
        y_hat_test_ug = H_test_lg * beta_ug;
        if task=='C'
            rslt_test_ug = calculate_acc(test_data.gnd, y_hat_test_ug);
            if nargin == 8
                fprintf("==>Test acc using graph: %f\n", rslt_test_ug);
                fprintf(fid, "==>Test acc using graph: %f\n", rslt_test_ug);
            end
            
        else
            rslt_test_ug = calculate_nrmse(test_data.gnd, y_hat_test_ug);
            if nargin == 8
                fprintf("==>Test NRMSE using graph: %f\n", rslt_test_ug);
                fprintf(fid, "==>Test NRMSE using graph: %f\n", rslt_test_ug);
            end
            
        end
        result_train_ug_arr(i) = rslt_train_ug;
        result_test_ug_arr(i) = rslt_test_ug;
    
        if nargin == 8
            fprintf("==> %d-th fold finished!\n", i);
            fprintf(fid, "==> %d-th fold finished!\n", i);
        end
        
    end

    result_test_mean_ug = mean(result_test_ug_arr); 
    result_test_std_ug = std(result_test_ug_arr);
    
    result_train_mean_ug = mean(result_train_ug_arr); 
    result_train_std_ug = std(result_train_ug_arr);
    
end