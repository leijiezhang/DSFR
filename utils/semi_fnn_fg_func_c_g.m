function [result_train_mean_g, result_train_std_g, result_test_mean_g, result_test_std_g, time_g]=...
        semi_fnn_fg_func_c_g(data_name, kfolds, n_rules, labeled_num, mu,beta,fid)
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);
    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

    result_test_g_arr = zeros(1,kfolds); 
    result_train_g_arr = zeros(1,kfolds);
    
    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data(i);
        train_data.n_smpl_l = labeled_num;
        shuffle_idx = randperm(train_data.n_smpl);
        train_data.fea_l = train_data.fea(shuffle_idx(1:train_data.n_smpl_l),:);
        train_data.gnd_l = train_data.gnd(shuffle_idx(1:train_data.n_smpl_l));
        train_data.fea_u = train_data.fea(shuffle_idx(train_data.n_smpl_l+1:end),:);
        train_data.gnd_u = train_data.gnd(shuffle_idx(train_data.n_smpl_l+1:end));
        train_data.n_smpl_u = size(train_data.fea_u,1);

        % =================treat all data as labeled data ===================  
        tic;
        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(train_data.fea, n_rules,beta);
        H_train_g = ComputeH(train_data.fea,rule_g);
        [~, beta_g] = FNN_solve(H_train_g, train_data.gnd, mu);
        time_g = toc;
        for j = 1:n_rules 
            rule_g.consq(j,:) = beta_g((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 

        y_hat_train_g = H_train_g * beta_g;
        if task=='C'
            rslt_train_g = calculate_acc(train_data.gnd, y_hat_train_g);
            if nargin == 7
                fprintf(fid, "==>Train acc with all training data involved: %f\n", rslt_train_g);
                fprintf("==>Train acc with all training data involved: %f\n", rslt_train_g);
            end
            
        else
            rslt_train_g = calculate_nrmse(train_data.gnd, y_hat_train_g);
            if nargin == 7
                fprintf(fid, "==>Train NRMSE with all training data involved: %f\n", rslt_train_g);
                fprintf("==>Train NRMSE with all training data involved: %f\n", rslt_train_g);
            end
            
        end

        rule_g = rule_g.update_fuzzyc(test_data.fea, rule_g.center, beta);
        H_test_g = ComputeH(test_data.fea,rule_g);
        y_hat_test_g = H_test_g * beta_g;
        if task=='C'
            rslt_test_g = calculate_acc(test_data.gnd, y_hat_test_g);
            if nargin == 7
                fprintf(fid, "==>Test acc with all training data involved: %f\n", rslt_test_g);
                fprintf("==>Test acc with all training data involved: %f\n", rslt_test_g);
            end
            
        else
            rslt_test_g = calculate_nrmse(test_data.gnd, y_hat_test_g);
            if nargin == 7
                fprintf(fid, "==>Test NRMSE with all training data involved: %f\n", rslt_test_g);
                fprintf("==>Test NRMSE with all training data involved: %f\n", rslt_test_g);
            end
            
        end

        result_train_g_arr(i) = rslt_train_g;
        result_test_g_arr(i) = rslt_test_g;
        
        if nargin == 7
            fprintf(fid, "==> %d-th fold finished!\n", i);
            fprintf("==> %d-th fold finished!\n", i);
        end
        
    end

    result_test_mean_g = mean(result_test_g_arr); 
    result_test_std_g = std(result_test_g_arr);
    
    result_train_mean_g = mean(result_train_g_arr); 
    result_train_std_g = std(result_train_g_arr);
    
end