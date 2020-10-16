function [result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, time_um]=...
        semi_fnn_fg_func_c_um(data_name, kfolds, n_rules, n_mixup, labeled_num, mu, gamma, alpha, beta, fid)
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);

    % Preprocess dataset and select parameters for the sietalation

    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

    result_test_um_arr = zeros(1,kfolds); 
    result_train_um_arr = zeros(1,kfolds);

    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data(i);
        train_data.n_smpl_l = labeled_num;
        shuffle_idx = randperm(train_data.n_smpl);
        train_data.fea_l = train_data.fea(shuffle_idx(1:train_data.n_smpl_l),:);
        train_data.gnd_l = train_data.gnd(shuffle_idx(1:train_data.n_smpl_l));
        train_data.fea_u = train_data.fea(shuffle_idx(train_data.n_smpl_l+1:end),:);
        train_data.gnd_u = train_data.gnd(shuffle_idx(train_data.n_smpl_l+1:end));
        train_data.n_smpl_u = size(train_data.fea_u,1); 
    
        % ==============semi-fnn using mix-up============% labeled data and unlabeled data
        tic;
        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(train_data.fea, n_rules,beta);
%         H_train_g = ComputeH(train_data.fea,rule_g);
        rule_lg = rule_g;
        rule_lg = rule_lg.update_fuzzyc(train_data.fea_l, rule_lg.center, beta);
        H_train_lg = ComputeH(train_data.fea_l,rule_lg);
        mix_index1 = mod(randperm(n_mixup),train_data.n_smpl_u)+1;
        mix_data1 = train_data.fea_u(mix_index1, :);
        mix_index2 = mod(randperm(n_mixup),train_data.n_smpl_u)+1;
        mix_data2 = train_data.fea_u(mix_index2, :);
        rule_um = rule_g;
        
        rho_1 = betarnd(alpha, alpha,n_mixup,1);
        rule_um = rule_um.update_fuzzyc(rho_1.*mix_data1+(1-rho_1).*mix_data2, rule_um.center, beta);      
        H_train_mix1 = ComputeH(rho_1.*mix_data1+(1-rho_1).*mix_data2,rule_um);
        rule_um = rule_um.update_fuzzyc(mix_data1, rule_um.center, beta);   
        H_train_mix2 = ComputeH(mix_data1,rule_um);
        rule_um = rule_um.update_fuzzyc(mix_data2, rule_um.center, beta);   
        H_train_mix3 = ComputeH(mix_data2,rule_um);
        B = H_train_mix1 - rho_1.*H_train_mix2 - (1-rho_1).*H_train_mix3;
        [~, p] = size(H_train_lg);
            
        beta_um = (gamma*B'*B + H_train_lg'*H_train_lg + mu*eye(p)) \ (H_train_lg'*train_data.gnd_l);
        time_um = toc;
        for j = 1:n_rules 
            rule_um.consq(j,:) = beta_um((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_um = H_train_lg * beta_um;
        if task=='C'
            rslt_train_um = calculate_acc(train_data.gnd_l, y_hat_train_um);
            if nargin == 10
                fprintf("==>Train acc using mix-up: %f\n", rslt_train_um);
                fprintf(fid, "==>Train acc using mix-up: %f\n", rslt_train_um);
            end
            
        else
            rslt_train_um = calculate_nrmse(train_data.gnd_l, y_hat_train_um);
            if nargin == 10
                fprintf("==>Train NRMSE using mix-up: %f\n", rslt_train_um);
                fprintf(fid, "==>Train NRMSE using mix-up: %f\n", rslt_train_um);
            end
            
        end
        rule_lg = rule_lg.update_fuzzyc(test_data.fea, rule_lg.center, beta);
        H_test_lg = ComputeH(test_data.fea,rule_lg);
        y_hat_test_um = H_test_lg * beta_um;
        if task=='C'
            rslt_test_um = calculate_acc(test_data.gnd, y_hat_test_um);
            if nargin == 10
                fprintf("==>Test acc using mix-up: %f\n", rslt_test_um);
                fprintf(fid, "==>Test acc using mix-up: %f\n", rslt_test_um);
            end
            
        else
            rslt_test_um = calculate_nrmse(test_data.gnd, y_hat_test_um);
            if nargin == 10
                fprintf("==>Test NRMSE using mix-up: %f\n", rslt_test_um);
                fprintf(fid, "==>Test NRMSE using mix-up: %f\n", rslt_test_um);
            end
            
        end
        result_train_um_arr(i) = rslt_train_um;
        result_test_um_arr(i) = rslt_test_um;
        
        if nargin == 10
            fprintf("==> %d-th fold finished!\n", i);
            fprintf(fid, "==> %d-th fold finished!\n", i);
        end
        
    end

    
    result_test_mean_um = mean(result_test_um_arr); 
    result_test_std_um = std(result_test_um_arr);
    
    result_train_mean_um = mean(result_train_um_arr); 
    result_train_std_um = std(result_train_um_arr);
    
end