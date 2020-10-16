function semi_cluter_cmp_func_c(data_name, kfolds, n_rules, n_mixup, labeled_rate, rho)
    log_path = './log/';
    if exist(log_path,'dir')==0
       mkdir(log_path);
    end
    logfilename = sprintf('%s%s_cmp_semifnn_r%d_l%.1f_rho%.4f.txt',...
        log_path, data_name, n_rules, labeled_rate, rho);
    time_local = datestr(now,0);
    fid = fopen(logfilename,'at');
    fprintf(fid, "==================================dataset: %s================================ \n", data_name);
    fprintf(fid, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf(fid, "============================================================================= \n");
    fprintf(fid, "==========rule number: %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, kfolds, labeled_rate);
    fprintf(fid, "========rho: %f: =========== \n",...
         rho);
    fprintf(fid, "============================================================================= \n");
    
    fprintf("==================================dataset: %s================================ \n", data_name);
    fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf("============================================================================= \n");
    fprintf("==========rule number: %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, kfolds, labeled_rate);
    fprintf("========rho: %f: =========== \n",...
         rho);
    fprintf("============================================================================= \n");
    fprintf(fid, "===================Centralized method: time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
    fprintf("===================Centralized method: time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
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

    % kmeans + Guassian active function
    result_test_g_kg_arr = zeros(1,kfolds); 
    result_train_g_kg_arr = zeros(1,kfolds);
    result_test_l_kg_arr = zeros(1,kfolds); 
    result_train_l_kg_arr = zeros(1,kfolds);
    
    % FuzzyC means + Guassian active function
    result_test_g_fg_arr = zeros(1,kfolds); 
    result_train_g_fg_arr = zeros(1,kfolds);
    result_test_l_fg_arr = zeros(1,kfolds); 
    result_train_l_fg_arr = zeros(1,kfolds);
    
    % FuzzyC means + Fuzzy partition matrix
    result_test_g_ff_arr = zeros(1,kfolds); 
    result_train_g_ff_arr = zeros(1,kfolds);
    result_test_l_ff_arr = zeros(1,kfolds); 
    result_train_l_ff_arr = zeros(1,kfolds);
    

    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data(i);
        train_data.n_smpl_l = floor(labeled_rate*train_data.n_smpl);
        shuffle_idx = randperm(train_data.n_smpl);
        train_data.fea_l = train_data.fea(shuffle_idx(1:train_data.n_smpl_l),:);
        train_data.gnd_l = train_data.gnd(shuffle_idx(1:train_data.n_smpl_l));
        train_data.fea_u = train_data.fea(shuffle_idx(train_data.n_smpl_l+1:end),:);
        train_data.gnd_u = train_data.gnd(shuffle_idx(train_data.n_smpl_l+1:end));
        train_data.n_smpl_u = size(train_data.fea_u,1);

        % =================treat all data as labeled data(kg) ===================     
        rule_g_kg =  Rules(n_rules);
        rule_g_kg = rule_g_kg.init_kmeans(train_data.fea, n_rules);
        H_train_g_kg = ComputeH(train_data.fea,rule_g_kg);
        [~, beta_g_kg] = FNN_solve(H_train_g_kg, train_data.gnd, rho);
        
        y_hat_train_g_kg = H_train_g_kg * beta_g_kg;
        if task=='C'
            rslt_train_g_kg = calculate_acc(train_data.gnd, y_hat_train_g_kg);
            fprintf(fid, "==>Train acc with all training data involved(kg): %f\n", rslt_train_g_kg);
            fprintf("==>Train acc with all training data involved(kg): %f\n", rslt_train_g_kg);
        else
            rslt_train_g_kg = calculate_nrmse(train_data.gnd, y_hat_train_g_kg);
            fprintf(fid, "==>Train NRMSE with all training data involved(kg): %f\n", rslt_train_g_kg);
            fprintf("==>Train NRMSE with all training data involved(kg): %f\n", rslt_train_g_kg);
        end

        H_test_g_kg = ComputeH(test_data.fea,rule_g_kg);
        y_hat_test_g_kg = H_test_g_kg * beta_g_kg;
        if task=='C'
            rslt_test_g_kg = calculate_acc(test_data.gnd, y_hat_test_g_kg);
            fprintf(fid, "==>Test acc with all training data involved(kg): %f\n", rslt_test_g_kg);
            fprintf("==>Test acc with all training data involved(kg): %f\n", rslt_test_g_kg);
        else
            rslt_test_g_kg = calculate_nrmse(test_data.gnd, y_hat_test_g_kg);
            fprintf(fid, "==>Test NRMSE with all training data involved(kg): %f\n", rslt_test_g_kg);
            fprintf("==>Test NRMSE with all training data involved(kg): %f\n", rslt_test_g_kg);
        end

        result_train_g_kg_arr(i) = rslt_train_g_kg;
        result_test_g_kg_arr(i) = rslt_test_g_kg;
        
        % =================treat all data as labeled data(fg) ===================     
       rule_g_fg =  Rules(n_rules);
        rule_g_fg = rule_g_fg.init_fuzzyc(train_data.fea, n_rules);
        H_train_g_fg = ComputeH(train_data.fea,rule_g_fg);
        [~, beta_g_fg] = FNN_solve(H_train_g_fg, train_data.gnd, rho);

        y_hat_train_g_fg = H_train_g_fg * beta_g_fg;
        if task=='C'
            rslt_train_g_fg = calculate_acc(train_data.gnd, y_hat_train_g_fg);
            fprintf(fid, "==>Train acc with all training data involved(fg): %f\n", rslt_train_g_fg);
            fprintf("==>Train acc with all training data involved(fg): %f\n", rslt_train_g_fg);
        else
            rslt_train_g_fg = calculate_nrmse(train_data.gnd, y_hat_train_g_fg);
            fprintf(fid, "==>Train NRMSE with all training data involved(fg): %f\n", rslt_train_g_fg);
            fprintf("==>Train NRMSE with all training data involved(fg): %f\n", rslt_train_g_fg);
        end

        H_test_g_fg = ComputeH(test_data.fea,rule_g_fg);
        y_hat_test_g_fg = H_test_g_fg * beta_g_fg;
        if task=='C'
            rslt_test_g_fg = calculate_acc(test_data.gnd, y_hat_test_g_fg);
            fprintf(fid, "==>Test acc with all training data involved(fg): %f\n", rslt_test_g_fg);
            fprintf("==>Test acc with all training data involved(fg): %f\n", rslt_test_g_fg);
        else
            rslt_test_g_fg = calculate_nrmse(test_data.gnd, y_hat_test_g_fg);
            fprintf(fid, "==>Test NRMSE with all training data involved(fg): %f\n", rslt_test_g_fg);
            fprintf("==>Test NRMSE with all training data involved(fg): %f\n", rslt_test_g_fg);
        end

        result_train_g_fg_arr(i) = rslt_train_g_fg;
        result_test_g_fg_arr(i) = rslt_test_g_fg;
        
        
        % =================treat all data as labeled data(ff) ===================     
       rule_g_ff =  Rules(n_rules);
        rule_g_ff = rule_g_ff.init_fuzzyc(train_data.fea, n_rules);
        P_train_g_ff = ComputeP(train_data.fea,rule_g_ff);
        [~, beta_g_ff] = FNN_solve(P_train_g_ff, train_data.gnd, rho);

        y_hat_train_g_ff = P_train_g_ff * beta_g_ff;
        if task=='C'
            rslt_train_g_ff = calculate_acc(train_data.gnd, y_hat_train_g_ff);
            fprintf(fid, "==>Train acc with all training data involved(ff): %f\n", rslt_train_g_ff);
            fprintf("==>Train acc with all training data involved(ff): %f\n", rslt_train_g_ff);
        else
            rslt_train_g_ff = calculate_nrmse(train_data.gnd, y_hat_train_g_ff);
            fprintf(fid, "==>Train NRMSE with all training data involved(ff): %f\n", rslt_train_g_ff);
            fprintf("==>Train NRMSE with all training data involved(ff): %f\n", rslt_train_g_ff);
        end

        P_test_g_ff = ComputeP(test_data.fea,rule_g_ff);
        y_hat_test_g_ff = P_test_g_ff * beta_g_ff;
        if task=='C'
            rslt_test_g_ff = calculate_acc(test_data.gnd, y_hat_test_g_ff);
            fprintf(fid, "==>Test acc with all training data involved(ff): %f\n", rslt_test_g_ff);
            fprintf("==>Test acc with all training data involved(ff): %f\n", rslt_test_g_ff);
        else
            rslt_test_g_ff = calculate_nrmse(test_data.gnd, y_hat_test_g_ff);
            fprintf(fid, "==>Test NRMSE with all training data involved(ff): %f\n", rslt_test_g_ff);
            fprintf("==>Test NRMSE with all training data involved(ff): %f\n", rslt_test_g_ff);
        end

        result_train_g_ff_arr(i) = rslt_train_g_ff;
        result_test_g_ff_arr(i) = rslt_test_g_ff;
        
        
        % ==============only labeled data involved(kg)============
    
        rule_l_kg = Rules(n_rules);
        rule_l_kg = rule_l_kg.init_kmeans(train_data.fea_l, n_rules);
        H_train_l_kg = ComputeH(train_data.fea_l,rule_l_kg);
        [~, beta_l_kg] = FNN_solve(H_train_l_kg, train_data.gnd_l, rho);

        for j = 1:n_rules 
            rule_l_kg.consq(j,:) = beta_l_kg((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_l_kg = H_train_l_kg * beta_l_kg;
        if task=='C'
            rslt_train_l_kg = calculate_acc(train_data.gnd_l, y_hat_train_l_kg);
            fprintf(fid, "==>Train acc with only labeled data involved(kg): %f\n", rslt_train_l_kg);
            fprintf("==>Train acc with only labeled data involved(kg): %f\n", rslt_train_l_kg);
        else
            rslt_train_l_kg = calculate_nrmse(train_data.gnd_l, y_hat_train_l_kg);
            fprintf(fid, "==>Train NRMSE with only labeled data involved(kg): %f\n", rslt_train_l_kg);
            fprintf("==>Train NRMSE with only labeled data involved(kg): %f\n", rslt_train_l_kg);
        end

        H_test_l_kg = ComputeH(test_data.fea,rule_l_kg);
        y_hat_test_l_kg = H_test_l_kg * beta_l_kg;
        if task=='C'
            rslt_test_l_kg = calculate_acc(test_data.gnd, y_hat_test_l_kg);
            fprintf(fid, "==>Test acc with only labeled data involved(kg): %f \n", rslt_test_l_kg);
            fprintf("==>Test acc with only labeled data involved(kg): %f \n", rslt_test_l_kg);
        else
            rslt_test_l_kg = calculate_nrmse(test_data.gnd, y_hat_test_l_kg);
            fprintf(fid, "==>Test NRMSE with only labeled data involved(kg): %f \n", rslt_test_l_kg);
            fprintf("==>Test NRMSE with only labeled data involved(kg): %f \n", rslt_test_l_kg);
        end
        result_train_l_kg_arr(i) = rslt_train_l_kg;
        result_test_l_kg_arr(i) = rslt_test_l_kg;
        
        
        % ==============only labeled data involved(fg)============
    
        rule_l_fg = Rules(n_rules);
        rule_l_fg = rule_l_fg.init_fuzzyc(train_data.fea_l, n_rules);
        H_train_l_fg = ComputeH(train_data.fea_l,rule_l_fg);
        [~, beta_l_fg] = FNN_solve(H_train_l_fg, train_data.gnd_l, rho);

        for j = 1:n_rules 
            rule_l_fg.consq(j,:) = beta_l_fg((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_l_fg = H_train_l_fg * beta_l_fg;
        if task=='C'
            rslt_train_l_fg = calculate_acc(train_data.gnd_l, y_hat_train_l_fg);
            fprintf(fid, "==>Train acc with only labeled data involved(fg): %f\n", rslt_train_l_fg);
            fprintf("==>Train acc with only labeled data involved(fg): %f\n", rslt_train_l_fg);
        else
            rslt_train_l_fg = calculate_nrmse(train_data.gnd_l, y_hat_train_l_fg);
            fprintf(fid, "==>Train NRMSE with only labeled data involved(fg): %f\n", rslt_train_l_fg);
            fprintf("==>Train NRMSE with only labeled data involved(fg): %f\n", rslt_train_l_fg);
        end

        H_test_l_fg = ComputeH(test_data.fea,rule_l_fg);
        y_hat_test_l_fg = H_test_l_fg * beta_l_fg;
        if task=='C'
            rslt_test_l_fg = calculate_acc(test_data.gnd, y_hat_test_l_fg);
            fprintf(fid, "==>Test acc with only labeled data involved(fg): %f \n", rslt_test_l_fg);
            fprintf("==>Test acc with only labeled data involved(fg): %f \n", rslt_test_l_fg);
        else
            rslt_test_l_fg = calculate_nrmse(test_data.gnd, y_hat_test_l_fg);
            fprintf(fid, "==>Test NRMSE with only labeled data involved(fg): %f \n", rslt_test_l_fg);
            fprintf("==>Test NRMSE with only labeled data involved(fg): %f \n", rslt_test_l_fg);
        end
        result_train_l_fg_arr(i) = rslt_train_l_fg;
        result_test_l_fg_arr(i) = rslt_test_l_fg;
        
         
        
        % ==============only labeled data involved============
    
        rule_l_ff = Rules(n_rules);
        rule_l_ff = rule_l_ff.init_fuzzyc(train_data.fea_l, n_rules);
        P_train_l_ff = ComputeP(train_data.fea_l,rule_l_ff);
        [~, beta_l_ff] = FNN_solve(P_train_l_ff, train_data.gnd_l, rho);

        for j = 1:n_rules 
            rule_l_ff.consq(j,:) = beta_l_ff((j - 1)*(train_data.n_fea + 1) + 1: j*(train_data.n_fea + 1))';
        end 
        y_hat_train_l_ff = P_train_l_ff * beta_l_ff;
        if task=='C'
            rslt_train_l_ff = calculate_acc(train_data.gnd_l, y_hat_train_l_ff);
            fprintf(fid, "==>Train acc with only labeled data involved(ff): %f\n", rslt_train_l_ff);
            fprintf("==>Train acc with only labeled data involved(ff): %f\n", rslt_train_l_ff);
        else
            rslt_train_l_ff = calculate_nrmse(train_data.gnd_l, y_hat_train_l_ff);
            fprintf(fid, "==>Train NRMSE with only labeled data involved(ff): %f\n", rslt_train_l_ff);
            fprintf("==>Train NRMSE with only labeled data involved(ff): %f\n", rslt_train_l_ff);
        end

        P_test_l_ff = ComputeP(test_data.fea,rule_l_ff);
        y_hat_test_l_ff = P_test_l_ff * beta_l_ff;
        if task=='C'
            rslt_test_l_ff = calculate_acc(test_data.gnd, y_hat_test_l_ff);
            fprintf(fid, "==>Test acc with only labeled data involved(ff): %f \n", rslt_test_l_ff);
            fprintf("==>Test acc with only labeled data involved(ff): %f \n", rslt_test_l_ff);
        else
            rslt_test_l_ff = calculate_nrmse(test_data.gnd, y_hat_test_l_ff);
            fprintf(fid, "==>Test NRMSE with only labeled data involved(ff): %f \n", rslt_test_l_ff);
            fprintf("==>Test NRMSE with only labeled data involved(ff): %f \n", rslt_test_l_ff);
        end
        result_train_l_ff_arr(i) = rslt_train_l_ff;
        result_test_l_ff_arr(i) = rslt_test_l_ff;
        
        fprintf(fid, "==> %d-th fold finished!\n", i);
        fprintf("==> %d-th fold finished!\n", i);
    end

    result_test_mean_g_kg = mean(result_test_g_kg_arr); 
    result_test_std_g_kg = std(result_test_g_kg_arr);
    result_test_mean_l_kg = mean(result_test_l_kg_arr); 
    result_test_std_l_kg = std(result_test_l_kg_arr);
    
    result_train_mean_g_kg = mean(result_train_g_kg_arr); 
    result_train_std_g_kg = std(result_train_g_kg_arr);
    result_train_mean_l_kg = mean(result_train_l_kg_arr); 
    result_train_std_l_kg = std(result_train_l_kg_arr);
    
    result_test_mean_g_fg = mean(result_test_g_fg_arr); 
    result_test_std_g_fg = std(result_test_g_fg_arr);
    result_test_mean_l_fg = mean(result_test_l_fg_arr); 
    result_test_std_l_fg = std(result_test_l_fg_arr);
    
    result_train_mean_g_fg = mean(result_train_g_fg_arr); 
    result_train_std_g_fg = std(result_train_g_fg_arr);
    result_train_mean_l_fg = mean(result_train_l_fg_arr); 
    result_train_std_l_fg = std(result_train_l_fg_arr);
    
    result_test_mean_g_ff = mean(result_test_g_ff_arr); 
    result_test_std_g_ff = std(result_test_g_ff_arr);
    result_test_mean_l_ff = mean(result_test_l_ff_arr); 
    result_test_std_l_ff = std(result_test_l_ff_arr);
    
    result_train_mean_g_ff = mean(result_train_g_ff_arr); 
    result_train_std_g_ff = std(result_train_g_ff_arr);
    result_train_mean_l_ff = mean(result_train_l_ff_arr); 
    result_train_std_l_ff = std(result_train_l_ff_arr);
    
    if exist('./results','dir')==0
       mkdir('./results');
    end
    save_dir = sprintf('./results/%s_cmp_semifnn_r%d_l%.1f_rho%.4f.mat',...
        data_name, n_rules, labeled_rate, rho);
    save(save_dir, 'result_test_mean_g_kg', 'result_test_std_g_kg', 'result_train_mean_g_kg', 'result_train_std_g_kg',...
        'result_test_mean_l_kg', 'result_test_std_l_kg', 'result_train_mean_l_kg', 'result_train_std_l_kg',...
        'result_test_mean_g_fg', 'result_test_std_g_fg', 'result_train_mean_g_fg', 'result_train_std_g_fg',...
        'result_test_mean_l_fg', 'result_test_std_l_fg', 'result_train_mean_l_fg', 'result_train_std_l_fg',...
        'result_test_mean_g_ff', 'result_test_std_g_ff', 'result_train_mean_g_ff', 'result_train_std_g_ff',...
        'result_test_mean_l_ff', 'result_test_std_l_ff', 'result_train_mean_l_ff', 'result_train_std_l_ff');

    
    fprintf(fid, "Train acc with all training data involved(kg): %.4f/%.4f\n", result_train_mean_g_kg, result_train_std_g_kg);
    fprintf(fid, "Test acc with all training data involved(kg): %.4f/%.4f\n", result_test_mean_g_kg, result_test_std_g_kg);
    
    fprintf(fid, "Train acc with only labeled data involved(kg): %.4f/%.4f\n", result_train_mean_l_kg, result_train_std_l_kg);
    fprintf(fid, "Test acc with only labeled data involved(kg): %.4f/%.4f\n", result_test_mean_l_kg, result_test_std_l_kg);
    fprintf(fid, "Train acc with all training data involved(fg): %.4f/%.4f\n", result_train_mean_g_fg, result_train_std_g_fg);
    fprintf(fid, "Test acc with all training data involved(fg): %.4f/%.4f\n", result_test_mean_g_fg, result_test_std_g_fg);
    fprintf(fid, "Train acc with only labeled data involved(fg): %.4f/%.4f\n", result_train_mean_l_fg, result_train_std_l_fg);
    fprintf(fid, "Test acc with only labeled data involved(fg): %.4f/%.4f\n", result_test_mean_l_fg, result_test_std_l_fg);
    fprintf(fid, "Train acc with all training data involved(ff): %.4f/%.4f\n", result_train_mean_g_ff, result_train_std_g_ff);
    fprintf(fid, "Test acc with all training data involved(ff): %.4f/%.4f\n", result_test_mean_g_ff, result_test_std_g_ff);
    fprintf(fid, "Train acc with only labeled data involved(ff): %.4f/%.4f\n", result_train_mean_l_ff, result_train_std_l_ff);
    fprintf(fid, "Test acc with only labeled data involved(ff): %.4f/%.4f\n", result_test_mean_l_ff, result_test_std_l_ff);
    
  
    
    fprintf("Train acc with all training data involved(kg): %.4f/%.4f\n", result_train_mean_g_kg, result_train_std_g_kg);
    fprintf("Test acc with all training data involved(kg): %.4f/%.4f\n", result_test_mean_g_kg, result_test_std_g_kg);
    
    fprintf("Train acc with all training data involved(fg): %.4f/%.4f\n", result_train_mean_g_fg, result_train_std_g_fg);
    fprintf("Test acc with all training data involved(fg): %.4f/%.4f\n", result_test_mean_g_fg, result_test_std_g_fg);
    
    fprintf("Train acc with all training data involved(ff): %.4f/%.4f\n", result_train_mean_g_ff, result_train_std_g_ff);
    fprintf("Test acc with all training data involved(ff): %.4f/%.4f\n", result_test_mean_g_ff, result_test_std_g_ff);
    
    fprintf("Train acc with only labeled data involved(kg): %.4f/%.4f\n", result_train_mean_l_kg, result_train_std_l_kg);
    fprintf("Test acc with only labeled data involved(kg): %.4f/%.4f\n", result_test_mean_l_kg, result_test_std_l_kg);
    
    fprintf("Train acc with only labeled data involved(fg): %.4f/%.4f\n", result_train_mean_l_fg, result_train_std_l_fg);
    fprintf("Test acc with only labeled data involved(fg): %.4f/%.4f\n", result_test_mean_l_fg, result_test_std_l_fg);
    
    fprintf("Train acc with only labeled data involved(ff): %.4f/%.4f\n", result_train_mean_l_ff, result_train_std_l_ff);
    fprintf("Test acc with only labeled data involved(ff): %.4f/%.4f\n", result_test_mean_l_ff, result_test_std_l_ff);
    
    
    fclose(fid);
end