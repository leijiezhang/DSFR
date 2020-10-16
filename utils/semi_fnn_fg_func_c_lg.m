function [result_train_mean_lg, result_train_std_lg, result_test_mean_lg, result_test_std_lg, time_lg]=...
        semi_fnn_fg_func_c_lg(data_name, kfolds, n_rules, labeled_num, mu, beta, fid)
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);

    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

    result_test_lg_arr = zeros(1,kfolds); 
    result_train_lg_arr = zeros(1,kfolds);

    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data(i);
        train_data.n_smpl_l = labeled_num;
        shuffle_idx = randperm(train_data.n_smpl);
        train_data.fea_l = train_data.fea(shuffle_idx(1:train_data.n_smpl_l),:);
        train_data.gnd_l = train_data.gnd(shuffle_idx(1:train_data.n_smpl_l));
        train_data.fea_u = train_data.fea(shuffle_idx(train_data.n_smpl_l+1:end),:);
        train_data.gnd_u = train_data.gnd(shuffle_idx(train_data.n_smpl_l+1:end));
        train_data.n_smpl_u = size(train_data.fea_u,1);
                
        % ============== semi-fnn all data involved in fcmeans============
        tic;
        rule_lg = Rules(n_rules);
        rule_lg = rule_lg.init_fuzzyc(train_data.fea, n_rules,beta);
        H_train_lg = ComputeH(train_data.fea_l,rule_lg);
        [~, beta_lg] = FNN_solve(H_train_lg, train_data.gnd_l, mu);
        time_lg = toc;
        for j = 1:n_rules 
            rule_lg.consq(j,:) = beta_lg((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_lg = H_train_lg * beta_lg;
        if task=='C'
            rslt_train_lg = calculate_acc(train_data.gnd_l, y_hat_train_lg);
            if nargin == 7
                fprintf("==>Train acc with all training data only involved kmeans part: %f\n", rslt_train_lg);
                fprintf(fid, "==>Train acc with all training data only involved kmeans part: %f\n", rslt_train_lg);
            end
            
        else
            rslt_train_lg = calculate_nrmse(train_data.gnd_l, y_hat_train_lg);
            if nargin == 7
                fprintf("==>Train NRMSE with all training data only involved kmeans part: %f\n", rslt_train_lg);
                fprintf(fid, "==>Train NRMSE with all training data only involved kmeans part: %f\n", rslt_train_lg);
            end
            
        end

        rule_lg = rule_lg.update_fuzzyc(test_data.fea, rule_lg.center, beta);
        H_test_lg = ComputeH(test_data.fea,rule_lg);
        y_hat_test_lg = H_test_lg * beta_lg;
        if task=='C'
            rslt_test_lg = calculate_acc(test_data.gnd, y_hat_test_lg);
            if nargin == 7
                fprintf("==>Test acc with all training data only involved kmeans part: %f\n", rslt_test_lg);
                fprintf(fid, "==>Test acc with all training data only involved kmeans part: %f\n", rslt_test_lg);
            end
            
        else
            rslt_test_lg = calculate_nrmse(test_data.gnd, y_hat_test_lg);
            if nargin == 7
                fprintf("==>Test NRMSE with all training data only involved kmeans part: %f\n", rslt_test_lg);
                fprintf(fid, "==>Test NRMSE with all training data only involved kmeans part: %f\n", rslt_test_lg);
            end
            
        end
        result_train_lg_arr(i) = rslt_train_lg;
        result_test_lg_arr(i) = rslt_test_lg;
         
        if nargin == 7
            fprintf("==> %d-th fold finished!\n", i);
            fprintf(fid, "==> %d-th fold finished!\n", i);
        end
        
    end

    result_test_mean_lg = mean(result_test_lg_arr); 
    result_test_std_lg = std(result_test_lg_arr);
    
    result_train_mean_lg = mean(result_train_lg_arr); 
    result_train_std_lg = std(result_train_lg_arr);
    
end