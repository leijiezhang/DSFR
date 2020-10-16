function semi_fnn_ff_func_c(data_name, kfolds, n_rules, n_mixup, labeled_num, rho, gamma, eta, alpha, beta)
    log_path = './log/';
    if exist(log_path,'dir')==0
       mkdir(log_path);
    end
    logfilename = sprintf('%s%s_c_ff_semifnn_r%d_l%.1f_la%.4f_eta%.4f_gamma%.4f_alpha%.4f_beta%.4f.txt',...
        log_path, data_name, n_rules, labeled_num, rho, eta, gamma, alpha, beta);
    time_local = datestr(now,0);
    fid = fopen(logfilename,'at');
    fprintf(fid, "==================================dataset: %s================================ \n", data_name);
    fprintf(fid, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf(fid, "============================================================================= \n");
    fprintf(fid, "==========rule number: %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, kfolds, labeled_num);
    fprintf(fid, "========rho: %f, eta: %f, gamma: %f, alpha: %f: , beta: %f:=========== \n",...
         rho, eta, gamma, alpha, beta);
    fprintf(fid, "============================================================================= \n");
    
    fprintf("==================================dataset: %s================================ \n", data_name);
    fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf("============================================================================= \n");
    fprintf("==========rule number: %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, kfolds, labeled_num);
    fprintf("========rho: %f, eta: %f, gamma: %f, alpha: %f: , beta: %f:=========== \n",...
         rho, eta, gamma, alpha, beta);
    fprintf("============================================================================= \n");
    fprintf(fid, "===================time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
    fprintf("===================time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
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

    result_test_g_arr = zeros(1,kfolds); 
    result_train_g_arr = zeros(1,kfolds);
    result_test_l_arr = zeros(1,kfolds); 
    result_train_l_arr = zeros(1,kfolds);
    result_test_lg_arr = zeros(1,kfolds); 
    result_train_lg_arr = zeros(1,kfolds);
    result_test_ug_arr = zeros(1,kfolds); 
    result_train_ug_arr = zeros(1,kfolds);
    result_test_um_arr = zeros(1,kfolds); 
    result_train_um_arr = zeros(1,kfolds);
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

        % =================treat all data as labeled data ===================        
%         [eta_optimal_g, ~] = Kmeans_ADMM(train_data.fea_d, n_rules, n_agent);

        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(train_data.fea, n_rules);
        P_train_g = ComputeP(train_data.fea,rule_g);
        [~, beta_g] = FNN_solve(P_train_g, train_data.gnd, rho);

        for j = 1:n_rules 
            rule_g.consq(j,:) = beta_g((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 

        y_hat_train_g = P_train_g * beta_g;
        if task=='C'
            rslt_train_g = calculate_acc(train_data.gnd, y_hat_train_g);
            fprintf(fid, "==>Train acc with all training data involved: %f\n", rslt_train_g);
            fprintf("==>Train acc with all training data involved: %f\n", rslt_train_g);
        else
            rslt_train_g = calculate_nrmse(train_data.gnd, y_hat_train_g);
            fprintf(fid, "==>Train NRMSE with all training data involved: %f\n", rslt_train_g);
            fprintf("==>Train NRMSE with all training data involved: %f\n", rslt_train_g);
        end

        rule_g = rule_g.update_u(test_data.fea, rule_g.center, beta);
        P_test_g = ComputeP(test_data.fea,rule_g);
        y_hat_test_g = P_test_g * beta_g;
        if task=='C'
            rslt_test_g = calculate_acc(test_data.gnd, y_hat_test_g);
            fprintf(fid, "==>Test acc with all training data involved: %f\n", rslt_test_g);
            fprintf("==>Test acc with all training data involved: %f\n", rslt_test_g);
        else
            rslt_test_g = calculate_nrmse(test_data.gnd, y_hat_test_g);
            fprintf(fid, "==>Test NRMSE with all training data involved: %f\n", rslt_test_g);
            fprintf("==>Test NRMSE with all training data involved: %f\n", rslt_test_g);
        end

        result_train_g_arr(i) = rslt_train_g;
        result_test_g_arr(i) = rslt_test_g;
        
        % ==============only labeled data involved============
    
        rule_l = Rules(n_rules);
        rule_l = rule_l.init_fuzzyc(train_data.fea_l, n_rules);
        P_train_l = ComputeP(train_data.fea_l,rule_l);
        [~, beta_l] = FNN_solve(P_train_l, train_data.gnd_l, rho);

        for j = 1:n_rules 
            rule_l.consq(j,:) = beta_l((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_l = P_train_l * beta_l;
        if task=='C'
            rslt_train_l = calculate_acc(train_data.gnd_l, y_hat_train_l);
            fprintf(fid, "==>Train acc with only labeled data involved: %f\n", rslt_train_l);
            fprintf("==>Train acc with only labeled data involved: %f\n", rslt_train_l);
        else
            rslt_train_l = calculate_nrmse(train_data.gnd_l, y_hat_train_l);
            fprintf(fid, "==>Train NRMSE with only labeled data involved: %f\n", rslt_train_l);
            fprintf("==>Train NRMSE with only labeled data involved: %f\n", rslt_train_l);
        end

        rule_l = rule_l.update_u(test_data.fea, rule_l.center, beta);
        P_test_l = ComputeP(test_data.fea,rule_l);
        y_hat_test_l = P_test_l * beta_l;
        if task=='C'
            rslt_test_l = calculate_acc(test_data.gnd, y_hat_test_l);
            fprintf(fid, "==>Test acc with only labeled data involved: %f \n", rslt_test_l);
            fprintf("==>Test acc with only labeled data involved: %f \n", rslt_test_l);
        else
            rslt_test_l = calculate_nrmse(test_data.gnd, y_hat_test_l);
            fprintf(fid, "==>Test NRMSE with only labeled data involved: %f \n", rslt_test_l);
            fprintf("==>Test NRMSE with only labeled data involved: %f \n", rslt_test_l);
        end
        result_train_l_arr(i) = rslt_train_l;
        result_test_l_arr(i) = rslt_test_l;
        
        % ============== semi-fnn all data involved in kmeans============
        rule_lg = rule_g;
        rule_lg = rule_lg.update_u(train_data.fea_l, rule_lg.center, beta);
        P_train_lg = ComputeP(train_data.fea_l,rule_lg);
        [~, beta_lg] = FNN_solve(P_train_lg, train_data.gnd_l, rho);

        for j = 1:n_rules 
            rule_lg.consq(j,:) = beta_lg((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_lg = P_train_lg * beta_lg;
        if task=='C'
            rslt_train_lg = calculate_acc(train_data.gnd_l, y_hat_train_lg);
            fprintf(fid, "==>Train acc with all training data only involved kmeans part: %f\n", rslt_train_lg);
            fprintf("==>Train acc with all training data only involved kmeans part: %f\n", rslt_train_lg);
        else
            rslt_train_lg = calculate_nrmse(train_data.gnd_l, y_hat_train_lg);
            fprintf(fid, "==>Train NRMSE with all training data only involved kmeans part: %f\n", rslt_train_lg);
            fprintf("==>Train NRMSE with all training data only involved kmeans part: %f\n", rslt_train_lg);
        end

        rule_lg = rule_lg.update_u(test_data.fea, rule_lg.center, beta);
        P_test_lg = ComputeP(test_data.fea,rule_lg);
        y_hat_test_lg = P_test_lg * beta_lg;
        if task=='C'
            rslt_test_lg = calculate_acc(test_data.gnd, y_hat_test_lg);
            fprintf(fid, "==>Test acc with all training data only involved kmeans part: %f\n", rslt_test_lg);
            fprintf("==>Test acc with all training data only involved kmeans part: %f\n", rslt_test_lg);
        else
            rslt_test_lg = calculate_nrmse(test_data.gnd, y_hat_test_lg);
            fprintf(fid, "==>Test NRMSE with all training data only involved kmeans part: %f\n", rslt_test_lg);
            fprintf("==>Test NRMSE with all training data only involved kmeans part: %f\n", rslt_test_lg);
        end
        result_train_lg_arr(i) = rslt_train_lg;
        result_test_lg_arr(i) = rslt_test_lg;
         
        % ==============all data involved using graph============% labeled data and unlabeled data
        
        dist_tmp = EuDist2(train_data.fea);
        t_tmp = mean(mean(dist_tmp));
        W_tmp = exp(-dist_tmp/(2*t_tmp^2));
        D_tmp = diag(sum(W_tmp,2));
        L_tmp = D_tmp - W_tmp;
        D_half = D_tmp^(-1/2);
        L_hat = D_half*L_tmp*D_half;
        [~, p] = size(P_train_l);
        Hinv = inv(eye(p)*rho  + P_train_lg' * P_train_lg + eta*P_train_g'*L_hat*P_train_g);
%       
        HY = P_train_lg'*train_data.gnd_l;
        beta_ug = Hinv*HY;
        
        y_hat_train_ug = P_train_lg * beta_ug;
        if task=='C'
            rslt_train_ug = calculate_acc(train_data.gnd_l, y_hat_train_ug);
            fprintf(fid, "==>Train acc using graph: %f\n", rslt_train_u);
            fprintf("==>Train acc using graph: %f\n", rslt_train_u);
        else
            rslt_train_ug = calculate_nrmse(train_data.gnd_l, y_hat_train_ug);
            fprintf(fid, "==>Train NRMSE using graph: %f\n", rslt_train_ug);
            fprintf("==>Train NRMSE using graph: %f\n", rslt_train_ug);
        end
        y_hat_test_ug = P_test_lg * beta_ug;
        if task=='C'
            rslt_test_ug = calculate_acc(test_data.gnd, y_hat_test_ug);
            fprintf(fid, "==>Test acc using graph: %f\n", rslt_test_ug);
            fprintf("==>Test acc using graph: %f\n", rslt_test_ug);
        else
            rslt_test_ug = calculate_nrmse(test_data.gnd, y_hat_test_ug);
            fprintf(fid, "==>Test NRMSE using graph: %f\n", rslt_test_ug);
            fprintf("==>Test NRMSE using graph: %f\n", rslt_test_ug);
        end
        result_train_ug_arr(i) = rslt_train_ug;
        result_test_ug_arr(i) = rslt_test_ug;
    
    
        % ==============semi-fnn using mix-up============% labeled data and unlabeled data
        rule_um = rule_g;
        
        mix_index1 = mod(randperm(n_mixup),train_data.n_smpl_u)+1;
        mix_data1 = train_data.fea_u(mix_index1, :);
        mix_index2 = mod(randperm(n_mixup),train_data.n_smpl_u)+1;
        mix_data2 = train_data.fea_u(mix_index2, :);
        
        
        rho_1 = betarnd(alpha, alpha,n_mixup,1);
        rule_um = rule_um.update_u(rho_1.*mix_data1+(1-rho_1).*mix_data2, rule_um.center, beta);      
        P_train_mix1 = ComputeP(rho_1.*mix_data1+(1-rho_1).*mix_data2,rule_um);
        rule_um = rule_um.update_u(mix_data1, rule_um.center, beta);   
        P_train_mix2 = ComputeP(mix_data1,rule_um);
        rule_um = rule_um.update_u(mix_data2, rule_um.center, beta);   
        P_train_mix3 = ComputeP(mix_data2,rule_um);
        B = P_train_mix1 - rho_1.*P_train_mix2 - (1-rho_1).*P_train_mix3;
%         [~, p] = size(P_train);
            
        beta_um = (gamma*B'*B + P_train_lg'*P_train_lg + rho*eye(p)) \ (P_train_lg'*train_data.gnd_l);
       
        for j = 1:n_rules 
            rule_u.consq(j,:) = beta_um((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_um = P_train_lg * beta_um;
        if task=='C'
            rslt_train_um = calculate_acc(train_data.gnd_l, y_hat_train_um);
            fprintf(fid, "==>Train acc using mix-up: %f\n", rslt_train_um);
            fprintf("==>Train acc using mix-up: %f\n", rslt_train_um);
        else
            rslt_train_um = calculate_nrmse(train_data.gnd_l, y_hat_train_um);
            fprintf(fid, "==>Train NRMSE using mix-up: %f\n", rslt_train_um);
            fprintf("==>Train NRMSE using mix-up: %f\n", rslt_train_um);
        end
        y_hat_test_um = P_test_g * beta_um;
        if task=='C'
            rslt_test_um = calculate_acc(test_data.gnd, y_hat_test_um);
            fprintf(fid, "==>Test acc using mix-up: %f\n", rslt_test_um);
            fprintf("==>Test acc using mix-up: %f\n", rslt_test_um);
        else
            rslt_test_um = calculate_nrmse(test_data.gnd, y_hat_test_um);
            fprintf(fid, "==>Test NRMSE using mix-up: %f\n", rslt_test_um);
            fprintf("==>Test NRMSE using mix-up: %f\n", rslt_test_um);
        end
        result_train_um_arr(i) = rslt_train_um;
        result_test_um_arr(i) = rslt_test_um;
        % ==============semi-fnn using mix-up and graph============% 
%         beta_umg = (eta*P_train_g'*L_hat*P_train_g + gamma*B'*B + P_train_l'*P_train_l + rho*eye(p)) \ (P_train_l'*train_data.gnd_l);
        
        beta_umg = (eta*P_train_g'*L_hat*P_train_g + gamma*B'*B + P_train_lg'*P_train_lg + rho*eye(p)) \ (P_train_lg'*train_data.gnd_l);
        for j = 1:n_rules 
            rule_u.consq(j,:) = beta_um((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_umg = P_train_lg * beta_umg;
        if task=='C'
            rslt_train_umg = calculate_acc(train_data.gnd_l, y_hat_train_umg);
            fprintf(fid, "==>Train acc using mix-up and graph: %f\n", rslt_train_umg);
            fprintf("==>Train acc using mix-up and graph: %f\n", rslt_train_umg);
        else
            rslt_train_umg = calculate_nrmse(train_data.gnd_l, y_hat_train_umg);
            fprintf(fid, "==>Train NRMSE using mix-up and graph: %f\n", rslt_train_umg);
            fprintf("==>Train NRMSE using mix-up and graph: %f\n", rslt_train_umg);
        end
        y_hat_test_umg = P_test_g * beta_umg;
        if task=='C'
            rslt_test_umg = calculate_acc(test_data.gnd, y_hat_test_umg);
            fprintf(fid, "==>Test acc using mix-up and graph: %f\n", rslt_test_umg);
            fprintf("==>Test acc using mix-up and graph: %f\n", rslt_test_umg);
        else
            rslt_test_umg = calculate_nrmse(test_data.gnd, y_hat_test_umg);
            fprintf(fid, "==>Test NRMSE using mix-up and graph: %f\n", rslt_test_umg);
            fprintf("==>Test NRMSE using mix-up and graph: %f\n", rslt_test_umg);
        end
        result_train_umg_arr(i) = rslt_train_umg;
        result_test_umg_arr(i) = rslt_test_umg;
        fprintf(fid, "==> %d-th fold finished!\n", i);
        fprintf("==> %d-th fold finished!\n", i);
    end

    result_test_mean_g = mean(result_test_g_arr); 
    result_test_std_g = std(result_test_g_arr);
    result_test_mean_l = mean(result_test_l_arr); 
    result_test_std_l = std(result_test_l_arr);
    result_test_mean_lg = mean(result_test_lg_arr); 
    result_test_std_lg = std(result_test_lg_arr);
    result_test_mean_um = mean(result_test_um_arr); 
    result_test_std_um = std(result_test_um_arr);
    result_test_mean_ug = mean(result_test_ug_arr); 
    result_test_std_ug = std(result_test_ug_arr);
    result_test_mean_umg = mean(result_test_umg_arr); 
    result_test_std_umg = std(result_test_umg_arr);

    result_train_mean_g = mean(result_train_g_arr); 
    result_train_std_g = std(result_train_g_arr);
    result_train_mean_l = mean(result_train_l_arr); 
    result_train_std_l = std(result_train_l_arr);
    result_train_mean_lg = mean(result_train_lg_arr); 
    result_train_std_lg = std(result_train_lg_arr);
    result_train_mean_um = mean(result_train_um_arr); 
    result_train_std_um = std(result_train_um_arr);
    result_train_mean_ug = mean(result_train_ug_arr); 
    result_train_std_ug = std(result_train_ug_arr);
    result_train_mean_umg = mean(result_train_umg_arr); 
    result_train_std_umg = std(result_train_umg_arr);

    if exist('./results','dir')==0
       mkdir('./results');
    end
    save_dir = sprintf('./results/%s_c_ff_semifnn_r%d_l%.1f_la%.4f_eta%.4f_gamma%.4f_alpha%.4f_beta%.4f.mat',...
        data_name, n_rules, labeled_num, rho, eta, gamma, alpha, beta);
    save(save_dir, 'result_test_mean_g', 'result_test_std_g', 'result_train_mean_g', 'result_train_std_g',...
        'result_test_mean_l', 'result_test_std_l', 'result_train_mean_l', 'result_train_std_l',...
        'result_test_mean_lg', 'result_test_std_lg', 'result_train_mean_lg', 'result_train_std_lg',...
        'result_test_mean_um', 'result_test_std_um', 'result_train_mean_um', 'result_train_std_um',...
        'result_test_mean_ug', 'result_test_std_ug', 'result_train_mean_ug', 'result_train_std_ug',...
        'result_test_mean_umg', 'result_test_std_umg', 'result_train_mean_umg', 'result_train_std_umg');

    
    fprintf(fid, "Train acc with all training data involved: %.4f/%.4f\n", result_train_mean_g, result_train_std_g);
    fprintf(fid, "Test acc with all training data involved: %.4f/%.4f\n", result_test_mean_g, result_test_std_g);
    fprintf(fid, "Train acc with only labeled data involved: %.4f/%.4f\n", result_train_mean_l, result_train_std_l);
    fprintf(fid, "Test acc with only labeled data involved: %.4f/%.4f\n", result_test_mean_l, result_test_std_l);
    fprintf(fid, "Train acc with all training data only involved kmeans part: %.4f/%.4f\n", result_train_mean_lg, result_train_std_lg);
    fprintf(fid, "Test acc with all training data only involved kmeans part: %.4f/%.4f\n", result_test_mean_lg, result_test_std_lg);
    fprintf(fid, "Train acc using graph: %.4f/%.4f\n", result_train_mean_ug, result_train_std_ug);
    fprintf(fid, "Test acc using graph: %.4f/%.4f\n", result_test_mean_ug, result_test_std_ug);
    fprintf(fid, "Train acc using mix-up: %.4f/%.4f\n", result_train_mean_um, result_train_std_um);
    fprintf(fid, "Test acc using mix-up: %.4f/%.4f\n", result_test_mean_um, result_test_std_um);
    fprintf(fid, "Train acc using mix-up and graph: %.4f/%.4f\n", result_train_mean_umg, result_train_std_umg);
    fprintf(fid, "Test acc using mix-up and graph: %.4f/%.4f\n", result_test_mean_umg, result_test_std_umg);
    
    fprintf("Train acc with all training data involved: %.4f/%.4f\n", result_train_mean_g, result_train_std_g);
    fprintf("Test acc with all training data involved: %.4f/%.4f\n", result_test_mean_g, result_test_std_g);
    fprintf("Train acc with only labeled data involved: %.4f/%.4f\n", result_train_mean_l, result_train_std_l);
    fprintf("Test acc with only labeled data involved: %.4f/%.4f\n", result_test_mean_l, result_test_std_l);
    fprintf("Train acc with all training data only involved kmeans part: %.4f/%.4f\n", result_train_mean_lg, result_train_std_lg);
    fprintf("Test acc with all training data only involved kmeans part: %.4f/%.4f\n", result_test_mean_lg, result_test_std_lg);
    fprintf("Train acc using graph: %.4f/%.4f\n", result_train_mean_ug, result_train_std_ug);
    fprintf("Test acc using graph: %.4f/%.4f\n", result_test_mean_ug, result_test_std_ug);
    fprintf("Train acc using mix-up: %.4f/%.4f\n", result_train_mean_um, result_train_std_um);
    fprintf("Test acc using mix-up: %.4f/%.4f\n", result_test_mean_um, result_test_std_um);
    fprintf("Train acc using mix-up and graph: %.4f/%.4f\n", result_train_mean_umg, result_train_std_umg);
    fprintf("Test acc using mix-up and graph: %.4f/%.4f\n", result_test_mean_umg, result_test_std_umg);
    
    fclose(fid);
end