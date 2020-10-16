function [result_train_mean_umg, result_train_std_umg, result_test_mean_umg, result_test_std_umg, time_umg]=...
        semi_fnn_fg_func_c_umg(data_name, kfolds, n_rules, n_mixup, labeled_num, mu, gamma,eta, alpha, beta, fid)
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);

    % Preprocess dataset and select parameters for the sietalation

    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

    result_test_umg_arr = zeros(1,kfolds); 
    result_train_umg_arr = zeros(1,kfolds);

    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data(i);
        train_data.n_smpl_l = labeled_num;
        shuffle_idx = randperm(train_data.n_smpl);
        train_data.fea_l = train_data.fea(shuffle_idx(1:train_data.n_smpl_l),:);
        train_data.gnd_l = train_data.gnd(shuffle_idx(1:train_data.n_smpl_l));
        train_data.fea_u = train_data.fea(shuffle_idx(train_data.n_smpl_l+1:end),:);
        train_data.gnd_u = train_data.gnd(shuffle_idx(train_data.n_smpl_l+1:end));
        train_data.n_smpl_u = size(train_data.fea_u,1); 
    
        % ==============semi-fnn using mix-up and graph============% 
        tic;
        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(train_data.fea, n_rules,beta);
        H_train_g = ComputeH(train_data.fea,rule_g);
        rule_lg = rule_g;
        rule_lg = rule_lg.update_fuzzyc(train_data.fea_l, rule_lg.center, beta);
        H_train_lg = ComputeH(train_data.fea_l,rule_lg);
        
        %graph part
        dist_tmp = EuDist2(train_data.fea);
        t_tmp = mean(mean(dist_tmp));
        W_tmp = exp(-dist_tmp/(2*t_tmp^2));
        D_tmp = diag(sum(W_tmp,2));
        L_tmp = D_tmp - W_tmp;
        D_half = D_tmp^(-1/2);
        L_hat = D_half*L_tmp*D_half;
        
        mix_index1 = mod(randperm(n_mixup),train_data.n_smpl_u)+1;
        mix_data1 = train_data.fea_u(mix_index1, :);
        mix_index2 = mod(randperm(n_mixup),train_data.n_smpl_u)+1;
        mix_data2 = train_data.fea_u(mix_index2, :);
        rule_umg = rule_g;
        
        rho_1 = betarnd(alpha, alpha,n_mixup,1);
        rule_umg = rule_umg.update_fuzzyc(rho_1.*mix_data1+(1-rho_1).*mix_data2, rule_umg.center, beta);      
        H_train_mix1 = ComputeH(rho_1.*mix_data1+(1-rho_1).*mix_data2,rule_umg);
        rule_umg = rule_umg.update_fuzzyc(mix_data1, rule_umg.center, beta);   
        H_train_mix2 = ComputeH(mix_data1,rule_umg);
        rule_umg = rule_umg.update_fuzzyc(mix_data2, rule_umg.center, beta);   
        H_train_mix3 = ComputeH(mix_data2,rule_umg);
        B = H_train_mix1 - rho_1.*H_train_mix2 - (1-rho_1).*H_train_mix3;
        [~, p] = size(H_train_lg);
            
        beta_umg = (eta*H_train_g'*L_hat*H_train_g + gamma*B'*B + H_train_lg'*H_train_lg + mu*eye(p)) \ (H_train_lg'*train_data.gnd_l);
        time_umg = toc;
        for j = 1:n_rules 
            rule_umg.consq(j,:) = beta_umg((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_umg = H_train_lg * beta_umg;
        if task=='C'
            rslt_train_umg = calculate_acc(train_data.gnd_l, y_hat_train_umg);
            if nargin == 11
                fprintf("==>Train acc using mix-up: %f\n", rslt_train_umg);
                fprintf(fid, "==>Train acc using mix-up: %f\n", rslt_train_umg);
            end
            
        else
            rslt_train_umg = calculate_nrmse(train_data.gnd_l, y_hat_train_umg);
            if nargin == 11
                fprintf("==>Train NRMSE using mix-up: %f\n", rslt_train_umg);
                fprintf(fid, "==>Train NRMSE using mix-up: %f\n", rslt_train_umg);
            end
            
        end
        rule_lg = rule_lg.update_fuzzyc(test_data.fea, rule_lg.center, beta);
        H_test_lg = ComputeH(test_data.fea,rule_lg);
        y_hat_test_umg = H_test_lg * beta_umg;
        if task=='C'
            rslt_test_umg = calculate_acc(test_data.gnd, y_hat_test_umg);
            if nargin == 11
                fprintf("==>Test acc using mix-up: %f\n", rslt_test_umg);
                fprintf(fid, "==>Test acc using mix-up: %f\n", rslt_test_umg);
            end
            
        else
            rslt_test_umg = calculate_nrmse(test_data.gnd, y_hat_test_umg);
            if nargin == 11
                fprintf("==>Test NRMSE using mix-up: %f\n", rslt_test_umg);
                fprintf(fid, "==>Test NRMSE using mix-up: %f\n", rslt_test_umg);
            end
            
        end
        result_train_umg_arr(i) = rslt_train_umg;
        result_test_umg_arr(i) = rslt_test_umg;
        
        if nargin == 11
            fprintf("==> %d-th fold finished!\n", i);
            fprintf(fid, "==> %d-th fold finished!\n", i);
        end
        
    end

    
    result_test_mean_umg = mean(result_test_umg_arr); 
    result_test_std_umg = std(result_test_umg_arr);
    
    result_train_mean_umg = mean(result_train_umg_arr); 
    result_train_std_umg = std(result_train_umg_arr);
    
end