function semi_fnn_fg_func_d_V1(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
    gamma,eta, alpha)
    log_path = './log/';
    if exist(log_path,'dir')==0
       mkdir(log_path);
    end
    logfilename = sprintf('%s%s_d_fg_semifnn_r%d_l%.1f_g%.4f_mu%.4f_rho_p%.4f_rho_s%.4f_beta%.4f_eta%.4f_al%.4f.txt',...
        log_path, data_name, n_rules, labeled_num, gamma, mu, rho_p, rho_s,beta,eta,alpha);
    time_local = datestr(now,0);
    fid = fopen(logfilename,'at');
    fprintf(fid, "==================================dataset: %s================================ \n", data_name);
    fprintf(fid, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf(fid, "============================================================================= \n");
    fprintf(fid, "==========rule number: %d, agent number %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, n_agent, kfolds, labeled_num);
    fprintf(fid, "========gamma: %f, mu: %f, rho_p: %f,rho_s: %.4f, beta:%.4f, eta: %f:, alpha: %f: =========== \n",...
         gamma, mu, rho_p, rho_s,beta, eta,alpha);
    fprintf(fid, "============================================================================= \n");
    
    fprintf("==================================dataset: %s================================ \n", data_name);
    fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf("============================================================================= \n");
    fprintf("========gamma: %f, mu: %f, rho_p: %f,rho_s: %.4f, beta:%.4f, eta: %f:, alpha: %f: =========== \n",...
         gamma, mu, rho_p, rho_s,beta, eta,alpha);
  
    fprintf("============================================================================= \n");
    fprintf(fid, "===================time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
    fprintf("===================time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);
%     alpha = 0.75;
    % load('concrete.mat')

    % Preprocess dataset and select parameters for the simulation

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
        [train_data, test_data] = dataset.get_kfold_data_d(n_agent, i);
        train_data.n_smpl_d_l = labeled_num;
        train_data.n_smpl_d_l = ceil(labeled_num/n_agent);
        shuffle_idx = randperm(train_data.n_smpl_d);
        for ii=1:n_agent
            if ii~=n_agent
                train_data.fea_d_l = train_data.fea_d(:,shuffle_idx(1:train_data.n_smpl_d_l),:);
                train_data.gnd_d_l = train_data.gnd_d(:,shuffle_idx(1:train_data.n_smpl_d_l));
                train_data.fea_d_u = train_data.fea_d(:,shuffle_idx(train_data.n_smpl_d_l+1:end),:);
                train_data.gnd_d_u = train_data.gnd_d(:,shuffle_idx(train_data.n_smpl_d_l+1:end));
            else
                n_smpl_d_l_last = labeled_num - (n_agent-1)*train_data.n_smpl_d_l;
                train_data.fea_d_l = train_data.fea_d(:,shuffle_idx(1:n_smpl_d_l_last),:);
                train_data.gnd_d_l = train_data.gnd_d(:,shuffle_idx(1:n_smpl_d_l_last));
                train_data.fea_d_u = train_data.fea_d(:,shuffle_idx(n_smpl_d_l_last+1:end),:);
                train_data.gnd_d_u = train_data.gnd_d(:,shuffle_idx(n_smpl_d_l_last+1:end));
            end
        end
        train_data.n_smpl_d_u = size(train_data.fea_d_u,2);
        
        
%         for ii = 1:n_agent
%             train_data.fea_d_l = [train_data.fea_d_l; squeeze(train_data.fea_d_l(ii,:,:))];
%             train_data.gnd_d_l = [train_data.gnd_d_l; squeeze(train_data.gnd_d_l(ii,:,:))];
%             train_data.fea_d_u = [train_data.fea_d_u; squeeze(train_data.fea_d_u(ii,:,:))];
%             train_data.gnd_d_u = [train_data.gnd_d_u; squeeze(train_data.gnd_d_u(ii,:,:))];
%         end

        % =================treat all data as labeled data ===================   
        tic;
        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(squeeze(train_data.fea_d(1,:,:)), n_rules);
        [mu_optimal_g, ~] = Fuzzycmeans_ADMM(train_data.fea_d, n_rules, n_agent,rho_s,beta);
%         rule_g = rule_g.update_kmeans(train_data.fea, mu_optimal_g);
        rule_g_arr = cell(1, n_agent);
        H_train_arr_g = cell(1, n_agent);
        w_train_arr_g = zeros(n_rules*(train_data.n_fea+1), n_agent);
        
        width_tmp_g = zeros(size(rule_g.width));
        for ii = 1:n_agent
            rule_g_arr{ii}= rule_g.update_fuzzyc(squeeze(train_data.fea_d(ii,:,:)), mu_optimal_g, beta);
            width_tmp_g = width_tmp_g + train_data.n_smpl_d*(rule_g_arr{ii}.width.^2);
        end
        width_g = sqrt(width_tmp_g/(n_agent*train_data.n_smpl_d));
        
        for ii = 1:n_agent
            rule_g_arr{ii}.width = width_g;
            H_train_arr_g{ii} = ComputeH(squeeze(train_data.fea_d(ii,:,:)),rule_g_arr{ii});
            [~, w_train_arr_g(:,ii)] = FNN_solve(H_train_arr_g{ii}, train_data.gnd_d(ii,:)', mu);
        end
        [w_d_g, ~] = FNN_ADMM(train_data.gnd_d, n_agent, w_train_arr_g, H_train_arr_g, mu, rho_p);
        time_g = toc;
        for j = 1:n_rules 
            rule_g.consq(j,:) = w_d_g((j - 1)*(train_data.n_fea_d + 1) + 1: j*(train_data.n_fea_d + 1))';
        end 

        rslt_train_g = zeros(n_agent,1);
        rslt_test_g = zeros(n_agent,1);
        y_hat_train_g = cell(1, n_agent);
       
        y_hat_test_g = cell(1, n_agent);
        H_test_arr_g = cell(1, n_agent);
        
        for ii = 1:n_agent
            y_hat_train_g{ii} = H_train_arr_g{ii} * w_d_g;
            H_test_arr_g{ii} = ComputeH(test_data.fea,rule_g_arr{ii});
            y_hat_test_g{ii} = H_test_arr_g{ii} * w_d_g;
            if task=='C'
                rslt_train_g(ii) = calculate_acc(train_data.gnd_d(ii,:)', y_hat_train_g{ii});
                fprintf(fid, "==>%d-th agent Train acc with all training data involved: %f\n", ii, rslt_train_g(ii));
                fprintf("==>%d-th agent Train acc with all training data involved: %f\n", ii, rslt_train_g(ii));
                rslt_test_g(ii) = calculate_acc(test_data.gnd, y_hat_test_g{ii});
                fprintf(fid, "==>%d-th agent Test acc with all training data involved: %f\n", ii, rslt_test_g(ii));
                fprintf("==>%d-th agent Test acc with all training data involved: %f\n", ii, rslt_test_g(ii));
            else
                rslt_train_g(ii) = calculate_nrmse(train_data.gnd_d(ii,:)', y_hat_train_g{ii});
                fprintf(fid, "==>%d-th agent Train NRMSE with all training data involved: %f\n", ii, rslt_train_g(ii));
                fprintf("==>%d-th agent Train NRMSE with all training data involved: %f\n", ii, rslt_train_g(ii));
                rslt_test_g(ii) = calculate_nrmse(test_data.gnd, y_hat_test_g{ii});
                fprintf(fid, "==>%d-th agent Test NRMSE with all training data involved: %f\n", ii, rslt_test_g(ii));
                fprintf("==>%d-th agent Test NRMSE with all training data involved: %f\n", ii, rslt_test_g(ii));
            end
            
        end
        
        
        if task=='C'
            fprintf(fid, "==>%d-th fold: Train acc with all training data involved: %.4f/%.4f\n", i, mean(rslt_train_g), std(rslt_train_g));
            fprintf("==>%d-th fold: Train acc with all training data involved: %.4f/%.4f\n", i, mean(rslt_train_g), std(rslt_train_g));
            fprintf(fid, "==>%d-th fold: Test acc with all training data involved: %.4f/%.4f\n", i, mean(rslt_test_g), std(rslt_test_g));
            fprintf("==>%d-th fold: Test acc with all training data involved: %.4f/%.4f\n", i, mean(rslt_test_g), std(rslt_test_g));
        else
            fprintf(fid, "%d-th fold: ==>%d-th agent Train NRMSE with all training data involved: %.4f/%.4f\n", i, mean(rslt_train_g), std(rslt_train_g));
            fprintf("==>%d-th fold: Train NRMSE with all training data involved: %.4f/%.4f\n", i, mean(rslt_train_g), std(rslt_train_g));
            fprintf(fid, "%d-th fold: ==>%d-th agent Test NRMSE with all training data involved: %.4f/%.4f\n", i, mean(rslt_test_g), std(rslt_test_g));
            fprintf("==>%d-th fold: Test NRMSE with all training data involved: %.4f/%.4f\n", i, mean(rslt_test_g), std(rslt_test_g));
        end
        result_train_g_arr(i) = mean(rslt_train_g);
        result_test_g_arr(i) = mean(rslt_test_g);
        
        % =================only labeled data involved ===================   
        tic;
        rule_l =  Rules(n_rules);
        rule_l = rule_l.init_fuzzyc(squeeze(train_data.fea_d_l(1,:,:)), n_rules);
        [mu_optimal_l, ~] = Fuzzycmeans_ADMM(train_data.fea_d_l, n_rules, n_agent,rho_s,beta);
%         rule_g = rule_g.update_kmeans(train_data.fea, mu_optimal_g);
        rule_l_arr = cell(1, n_agent);
        H_train_arr_l = cell(1, n_agent);
        w_train_arr_l = zeros(n_rules*(train_data.n_fea+1), n_agent);
        
        width_tmp_l = zeros(size(rule_g.width));
        for ii = 1:n_agent
            rule_l_arr{ii}= rule_l.update_fuzzyc(squeeze(train_data.fea_d_l(ii,:,:)), mu_optimal_l, beta);
            width_tmp_l = width_tmp_l + train_data.n_smpl_d*(rule_l_arr{ii}.width.^2);
        end
        width_l = sqrt(width_tmp_l/(n_agent*train_data.n_smpl_d));
        
        for ii = 1:n_agent
            rule_l_arr{ii}.width = width_l;
            H_train_arr_l{ii} = ComputeH(squeeze(train_data.fea_d_l(ii,:,:)),rule_l_arr{ii});
            [~, w_train_arr_l(:,ii)] = FNN_solve(H_train_arr_l{ii}, train_data.gnd_d_l(ii,:)', mu);
        end
        [w_d_l, ~] = FNN_ADMM(train_data.gnd_d_l, n_agent, w_train_arr_l, H_train_arr_l, mu, rho_p);
        time_l = toc;
        for j = 1:n_rules 
            rule_l.consq(j,:) = w_d_l((j - 1)*(train_data.n_fea_d + 1) + 1: j*(train_data.n_fea_d + 1))';
        end 

        rslt_train_l = zeros(n_agent,1);
        rslt_test_l = zeros(n_agent,1);
        y_hat_train_l = cell(1, n_agent);
        
        y_hat_test_l = cell(1, n_agent);
        H_test_arr_l = cell(1, n_agent);
        for ii = 1:n_agent
            y_hat_train_l{ii} = H_train_arr_l{ii} * w_d_l;
            H_test_arr_l{ii} = ComputeH(test_data.fea,rule_l_arr{ii});
            y_hat_test_l{ii} = H_test_arr_l{ii} * w_d_l;
            if task=='C'
                rslt_train_l(ii) = calculate_acc(train_data.gnd_d_l(ii,:)', y_hat_train_l{ii});
                fprintf(fid, "==>%d-th agent Train acc with only labeled data involved: %f\n", ii, rslt_train_l(ii));
                fprintf("==>%d-th agent Train acc with only labeled data involved: %f\n", ii, rslt_train_l(ii));
                rslt_test_l(ii) = calculate_acc(test_data.gnd, y_hat_test_l{ii});
                fprintf(fid, "==>%d-th agent Test acc with only labeled data involved: %f\n", ii, rslt_test_l(ii));
                fprintf("==>%d-th agent Test acc with only labeled data involved: %f\n", ii, rslt_test_l(ii));
            else
                rslt_train_l(ii) = calculate_nrmse(train_data.gnd_d_l(ii,:)', y_hat_train_l{ii});
                fprintf(fid, "==>%d-th agent Train NRMSE with only labeled data involved: %f\n", ii, rslt_train_l(ii));
                fprintf("==>%d-th agent Train NRMSE with only labeled data involved: %f\n", ii, rslt_train_l(ii));
                rslt_test_l(ii) = calculate_nrmse(test_data.gnd, y_hat_test_l{ii});
                fprintf(fid, "==>%d-th agent Test NRMSE with only labeled data involved: %f\n", ii, rslt_test_l(ii));
                fprintf("==>%d-th agent Test NRMSE with only labeled data involved: %f\n", ii, rslt_test_l(ii));
            end
            
        end
        
        
        if task=='C'
            fprintf(fid, "==>%d-th fold: Train acc with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_train_l), std(rslt_train_l));
            fprintf("==>%d-th fold: Train acc with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_train_l), std(rslt_train_l));
            fprintf(fid, "==>%d-th fold: Test acc with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_test_l), std(rslt_test_l));
            fprintf("==>%d-th fold: Test acc with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_test_l), std(rslt_test_l));
        else
            fprintf(fid, "%d-th fold: ==>%d-th agent Train NRMSE with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_train_l), std(rslt_train_l));
            fprintf("==>%d-th fold: Train NRMSE with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_train_l), std(rslt_train_l));
            fprintf(fid, "%d-th fold: ==>%d-th agent Test NRMSE with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_test_l), std(rslt_test_l));
            fprintf("==>%d-th fold: Test NRMSE with only labeled data involved: %.4f/%.4f\n", i, mean(rslt_test_l), std(rslt_test_l));
        end
        result_train_l_arr(i) = mean(rslt_train_l);
        result_test_l_arr(i) = mean(rslt_test_l);
        
                
        % ============== semi-fnn all data involved in kmeans============
        tic;
        rule_lg = rule_g;
        rule_lg = rule_g.init_fuzzyc(squeeze(train_data.fea_d(1,:,:)), n_rules);
%         [mu_optimal_lg, ~] = Kmeans_ADMM(train_data.fea_d, n_rules, n_agent);
%         rule_g = rule_g.update_kmeans(train_data.fea, mu_optimal_g);
        rule_lg_arr = cell(1, n_agent);
        H_train_arr_lg = cell(1, n_agent);
        w_train_arr_lg = zeros(n_rules*(train_data.n_fea+1), n_agent);
        
        width_tmp_lg = zeros(size(rule_g.width));
        for ii = 1:n_agent
            rule_lg_arr{ii}= rule_lg.update_fuzzyc(squeeze(train_data.fea_d(ii,:,:)), mu_optimal_g, beta);
            width_tmp_lg = width_tmp_lg + train_data.n_smpl_d*(rule_lg_arr{ii}.width.^2);
        end
        width_lg = sqrt(width_tmp_lg/(n_agent*train_data.n_smpl_d));
        
        for ii = 1:n_agent
            rule_lg_arr{ii}.width = width_lg;
            H_train_arr_lg{ii} = ComputeH(squeeze(train_data.fea_d_l(ii,:,:)),rule_lg_arr{ii});
            [~, w_train_arr_lg(:,ii)] = FNN_solve(H_train_arr_lg{ii}, train_data.gnd_d_l(ii,:)', mu);
        end
        [w_d_lg, ~] = FNN_ADMM(train_data.gnd_d_l, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p);
        time_lg = toc;
        for j = 1:n_rules 
            rule_lg.consq(j,:) = w_d_lg((j - 1)*(train_data.n_fea_d + 1) + 1: j*(train_data.n_fea_d + 1))';
        end 

        rslt_train_lg = zeros(n_agent,1);
        rslt_test_lg = zeros(n_agent,1);
        y_hat_train_lg = cell(1, n_agent);
        
        y_hat_test_lg = cell(1, n_agent);
        H_test_arr_lg = cell(1, n_agent);
        for ii = 1:n_agent
            y_hat_train_lg{ii} = H_train_arr_lg{ii} * w_d_lg;
            H_test_arr_lg{ii} = ComputeH(test_data.fea,rule_lg_arr{ii});
            y_hat_test_lg{ii} = H_test_arr_lg{ii} * w_d_lg;
            if task=='C'
                rslt_train_lg(ii) = calculate_acc(train_data.gnd_d_l(ii,:)', y_hat_train_lg{ii});
                fprintf(fid, "==>%d-th agent Train acc with all training data only involved kmeans part: %f\n", ii, rslt_train_lg(ii));
                fprintf("==>%d-th agent Train acc with all training data only involved kmeans part: %f\n", ii, rslt_train_lg(ii));
                rslt_test_lg(ii) = calculate_acc(test_data.gnd, y_hat_test_lg{ii});
                fprintf(fid, "==>%d-th agent Test acc with all training data only involved kmeans part: %f\n", ii, rslt_test_lg(ii));
                fprintf("==>%d-th agent Test acc with all training data only involved kmeans part: %f\n", ii, rslt_test_lg(ii));
            else
                rslt_train_lg(ii) = calculate_nrmse(train_data.gnd_d_l(ii,:)', y_hat_train_lg{ii});
                fprintf(fid, "==>%d-th agent Train NRMSE with all training data only involved kmeans part: %f\n", ii, rslt_train_lg(ii));
                fprintf("==>%d-th agent Train NRMSE with all training data only involved kmeans part: %f\n", ii, rslt_train_lg(ii));
                rslt_test_lg(ii) = calculate_nrmse(test_data.gnd, y_hat_test_lg{ii});
                fprintf(fid, "==>%d-th agent Test NRMSE with all training data only involved kmeans part: %f\n", ii, rslt_test_lg(ii));
                fprintf("==>%d-th agent Test NRMSE with all training data only involved kmeans part: %f\n", ii, rslt_test_lg(ii));
            end
            
        end
        
        
        if task=='C'
            fprintf(fid, "==>%d-th fold: Train acc with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_train_lg), std(rslt_train_lg));
            fprintf("==>%d-th fold: Train acc with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_train_lg), std(rslt_train_lg));
            fprintf(fid, "==>%d-th fold: Test acc with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_test_lg), std(rslt_test_lg));
            fprintf("==>%d-th fold: Test acc with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_test_lg), std(rslt_test_lg));
        else
            fprintf(fid, "%d-th fold: ==>%d-th agent Train NRMSE with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_train_lg), std(rslt_train_lg));
            fprintf("==>%d-th fold: Train NRMSE with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_train_lg), std(rslt_train_lg));
            fprintf(fid, "%d-th fold: ==>%d-th agent Test NRMSE with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_test_lg), std(rslt_test_lg));
            fprintf("==>%d-th fold: Test NRMSE with all training data only involved kmeans part: %.4f/%.4f\n", i, mean(rslt_test_lg), std(rslt_test_lg));
        end
        result_train_lg_arr(i) = mean(rslt_train_lg);
        result_test_lg_arr(i) = mean(rslt_test_lg);
        
         
        % ==============all data involved using graph============% labeled data and unlabeled data
        tic;
        rule_ug = rule_g;
        [w_d_ug, ~] = graph_FNN_ADMM(train_data, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p,...
             H_train_arr_g, eta);
                
        for j = 1:n_rules 
            rule_ug.consq(j,:) = w_d_ug((j - 1)*(train_data.n_fea_d + 1) + 1: j*(train_data.n_fea_d + 1))';
        end 

        rslt_train_ug = zeros(n_agent,1);
        rslt_test_ug = zeros(n_agent,1);
        y_hat_train_ug = cell(1, n_agent);
        H_train_arr_ug = cell(1, n_agent);
        y_hat_test_ug = cell(1, n_agent);
        H_test_arr_ug = cell(1, n_agent);
        rule_ug_arr = cell(1, n_agent);
        
        width_tmp_ug= zeros(size(rule_g.width));
        for ii = 1:n_agent
            rule_ug_arr{ii}= rule_ug.update_fuzzyc(squeeze(train_data.fea_d(ii,:,:)), mu_optimal_g, beta);
            width_tmp_ug = width_tmp_ug + train_data.n_smpl_d*(rule_lg_arr{ii}.width.^2);
        end
        width_ug = sqrt(width_tmp_ug/(n_agent*train_data.n_smpl_d));
        time_ug = toc;
        for ii = 1:n_agent
            rule_ug_arr{ii}.width = width_ug;
       
            H_train_arr_ug{ii} = ComputeH(squeeze(train_data.fea_d_l(ii,:,:)),rule_ug_arr{ii});
            y_hat_train_ug{ii} = H_train_arr_ug{ii} * w_d_ug;
            H_test_arr_ug{ii} = ComputeH(test_data.fea,rule_ug_arr{ii});
            y_hat_test_ug{ii} = H_test_arr_ug{ii} * w_d_ug;
            if task=='C'
                rslt_train_ug(ii) = calculate_acc(train_data.gnd_d_l(ii,:)', y_hat_train_ug{ii});
                fprintf(fid, "==>%d-th agent Train acc using graph: %f\n", ii, rslt_train_ug(ii));
                fprintf("==>%d-th agent Train acc using graph: %f\n", ii, rslt_train_ug(ii));
                rslt_test_ug(ii) = calculate_acc(test_data.gnd, y_hat_test_ug{ii});
                fprintf(fid, "==>%d-th agent Test acc using graph: %f\n", ii, rslt_test_ug(ii));
                fprintf("==>%d-th agent Test acc using graph: %f\n", ii, rslt_test_ug(ii));
            else
                rslt_train_ug(ii) = calculate_nrmse(train_data.gnd_d_l(ii,:)', y_hat_train_ug{ii});
                fprintf(fid, "==>%d-th agent Train NRMSE using graph: %f\n", ii, rslt_train_ug(ii));
                fprintf("==>%d-th agent Train NRMSE using graph: %f\n", ii, rslt_train_ug(ii));
                rslt_test_ug(ii) = calculate_nrmse(test_data.gnd, y_hat_test_ug{ii});
                fprintf(fid, "==>%d-th agent Test NRMSE using graph: %f\n", ii, rslt_test_ug(ii));
                fprintf("==>%d-th agent Test NRMSE using graph: %f\n", ii, rslt_test_ug(ii));
            end
            
        end
        
        
        if task=='C'
            fprintf(fid, "==>%d-th fold: Train acc using graph: %.4f/%.4f\n", i, mean(rslt_train_ug), std(rslt_train_ug));
            fprintf("==>%d-th fold: Train acc using graph: %.4f/%.4f\n", i, mean(rslt_train_ug), std(rslt_train_ug));
            fprintf(fid, "==>%d-th fold: Test acc using graph: %.4f/%.4f\n", i, mean(rslt_test_ug), std(rslt_test_ug));
            fprintf("==>%d-th fold: Test acc using graph: %.4f/%.4f\n", i, mean(rslt_test_ug), std(rslt_test_ug));
        else
            fprintf(fid, "%d-th fold: ==>%d-th agent Train NRMSE using graph: %.4f/%.4f\n", i, mean(rslt_train_ug), std(rslt_train_ug));
            fprintf("==>%d-th fold: Train NRMSE using graph: %.4f/%.4f\n", i, mean(rslt_train_ug), std(rslt_train_ug));
            fprintf(fid, "%d-th fold: ==>%d-th agent Test NRMSE using graph: %.4f/%.4f\n", i, mean(rslt_test_ug), std(rslt_test_ug));
            fprintf("==>%d-th fold: Test NRMSE using graph: %.4f/%.4f\n", i, mean(rslt_test_ug), std(rslt_test_ug));
        end
        result_train_ug_arr(i) = mean(rslt_train_ug);
        result_test_ug_arr(i) = mean(rslt_test_ug);
        
    
        % ==============semi-fnn using mix-up============% labeled data and unlabeled data
        tic;
        rule_um = rule_g;
        [w_d_um, ~] = mixup_FNN_ADMM(train_data, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p,...
            rule_um, n_mixup, gamma, alpha);
                
        for j = 1:n_rules 
            rule_um.consq(j,:) = w_d_um((j - 1)*(train_data.n_fea_d + 1) + 1: j*(train_data.n_fea_d + 1))';
        end 

        rslt_train_um = zeros(n_agent,1);
        rslt_test_um = zeros(n_agent,1);
        y_hat_train_um = cell(1, n_agent);
        H_train_um = cell(1, n_agent);
        y_hat_test_um = cell(1, n_agent);
        H_test_um = cell(1, n_agent);
        rule_um_arr = cell(1, n_agent);
        
        width_tmp_um= zeros(size(rule_g.width));
        for ii = 1:n_agent
            rule_um_arr{ii}= rule_um.update_fuzzyc(squeeze(train_data.fea_d(ii,:,:)), mu_optimal_g, beta);
            width_tmp_um = width_tmp_um + train_data.n_smpl_d*(rule_lg_arr{ii}.width.^2);
        end
        width_um = sqrt(width_tmp_um/(n_agent*train_data.n_smpl_d));
        time_um = toc;
        for ii = 1:n_agent
            rule_um_arr{ii}.width = width_um;
            
            H_train_um{ii} = ComputeH(squeeze(train_data.fea_d_l(ii,:,:)),rule_um_arr{ii});
            y_hat_train_um{ii} = H_train_um{ii} * w_d_um;
            H_test_um{ii} = ComputeH(test_data.fea,rule_um_arr{ii});
            y_hat_test_um{ii} = H_test_um{ii} * w_d_um;
            if task=='C'
                rslt_train_um(ii) = calculate_acc(train_data.gnd_d_l(ii,:)', y_hat_train_um{ii});
                fprintf(fid, "==>%d-th agent Train acc using mix-up: %f\n", ii, rslt_train_um(ii));
                fprintf("==>%d-th agent Train acc using mix-up: %f\n", ii, rslt_train_um(ii));
                rslt_test_um(ii) = calculate_acc(test_data.gnd, y_hat_test_um{ii});
                fprintf(fid, "==>%d-th agent Test acc using mix-up: %f\n", ii, rslt_test_um(ii));
                fprintf("==>%d-th agent Test acc using mix-up: %f\n", ii, rslt_test_um(ii));
            else
                rslt_train_um(ii) = calculate_nrmse(train_data.gnd_d_l(ii,:)', y_hat_train_um{ii});
                fprintf(fid, "==>%d-th agent Train NRMSE using mix-up: %f\n", ii, rslt_train_um(ii));
                fprintf("==>%d-th agent Train NRMSE using mix-up: %f\n", ii, rslt_train_um(ii));
                rslt_test_um(ii) = calculate_nrmse(test_data.gnd, y_hat_test_um{ii});
                fprintf(fid, "==>%d-th agent Test NRMSE using mix-up: %f\n", ii, rslt_test_um(ii));
                fprintf("==>%d-th agent Test NRMSE using mix-up: %f\n", ii, rslt_test_um(ii));
            end
            
        end
        
        
        if task=='C'
            fprintf(fid, "==>%d-th fold: Train acc using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
            fprintf("==>%d-th fold: Train acc using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
            fprintf(fid, "==>%d-th fold: Test acc using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
            fprintf("==>%d-th fold: Test acc using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
        else
            fprintf(fid, "%d-th fold: ==>%d-th agent Train NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
            fprintf("==>%d-th fold: Train NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
            fprintf(fid, "%d-th fold: ==>%d-th agent Test NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
            fprintf("==>%d-th fold: Test NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
        end
        result_train_um_arr(i) = mean(rslt_train_um);
        result_test_um_arr(i) = mean(rslt_test_um);
        
        
        % ==============semi-fnn using mix-up and graph============% rule_umg = rule_g;
        tic;
        rule_umg = rule_g;
        [w_d_umg, ~] = graph_mixup_FNN_ADMM(train_data, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p,...
            rule_umg, n_mixup, gamma,alpha,...
            H_train_arr_g, eta);

        for j = 1:n_rules 
            rule_umg.consq(j,:) = w_d_umg((j - 1)*(train_data.n_fea_d + 1) + 1: j*(train_data.n_fea_d + 1))';
        end 

        rslt_train_umg = zeros(n_agent,1);
        rslt_test_umg = zeros(n_agent,1);
        y_hat_train_umg = cell(1, n_agent);
        H_train_arr_umg = cell(1, n_agent);
        y_hat_test_umg = cell(1, n_agent);
        H_test_arr_umg = cell(1, n_agent);
        rule_umg_arr = cell(1, n_agent);
        
        width_tmp_umg= zeros(size(rule_g.width));
        for ii = 1:n_agent
            rule_umg_arr{ii}= rule_umg.update_fuzzyc(squeeze(train_data.fea_d(ii,:,:)), mu_optimal_g, beta);
            width_tmp_umg = width_tmp_umg + train_data.n_smpl_d*(rule_lg_arr{ii}.width.^2);
        end
        width_umg = sqrt(width_tmp_umg/(n_agent*train_data.n_smpl_d));
        time_umg = toc;
        for ii = 1:n_agent
            rule_umg_arr{ii}.width = width_umg;
            H_train_arr_umg{ii} = ComputeH(squeeze(train_data.fea_d_l(ii,:,:)),rule_umg_arr{ii});
            y_hat_train_umg{ii} = H_train_arr_umg{ii} * w_d_umg;
            H_test_arr_umg{ii} = ComputeH(test_data.fea,rule_umg_arr{ii});
            y_hat_test_umg{ii} = H_test_arr_umg{ii} * w_d_umg;
            if task=='C'
                rslt_train_umg(ii) = calculate_acc(train_data.gnd_d_l(ii,:)', y_hat_train_umg{ii});
                fprintf(fid, "==>%d-th agent Train acc using mix-up and graph: %f\n", ii, rslt_train_umg(ii));
                fprintf("==>%d-th agent Train acc using mix-up and graph: %f\n", ii, rslt_train_umg(ii));
                rslt_test_umg(ii) = calculate_acc(test_data.gnd, y_hat_test_umg{ii});
                fprintf(fid, "==>%d-th agent Test acc using mix-up and graph: %f\n", ii, rslt_test_umg(ii));
                fprintf("==>%d-th agent Test acc using mix-up and graph: %f\n", ii, rslt_test_umg(ii));
            else
                rslt_train_umg(ii) = calculate_nrmse(train_data.gnd_d_l(ii,:)', y_hat_train_umg{ii});
                fprintf(fid, "==>%d-th agent Train NRMSE using mix-up and graph: %f\n", ii, rslt_train_umg(ii));
                fprintf("==>%d-th agent Train NRMSE using mix-up and graph: %f\n", ii, rslt_train_umg(ii));
                rslt_test_umg(ii) = calculate_nrmse(test_data.gnd, y_hat_test_umg{ii});
                fprintf(fid, "==>%d-th agent Test NRMSE using mix-up and graph: %f\n", ii, rslt_test_umg(ii));
                fprintf("==>%d-th agent Test NRMSE using mix-up and graph: %f\n", ii, rslt_test_umg(ii));
            end
            
        end
        
        
        if task=='C'
            fprintf(fid, "==>%d-th fold: Train acc using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_train_umg), std(rslt_train_umg));
            fprintf("==>%d-th fold: Train acc using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_train_umg), std(rslt_train_umg));
            fprintf(fid, "==>%d-th fold: Test acc using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_test_umg), std(rslt_test_umg));
            fprintf("==>%d-th fold: Test acc using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_test_umg), std(rslt_test_umg));
        else
            fprintf(fid, "%d-th fold: ==>%d-th agent Train NRMSE using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_train_umg), std(rslt_train_umg));
            fprintf("==>%d-th fold: Train NRMSE using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_train_umg), std(rslt_train_umg));
            fprintf(fid, "%d-th fold: ==>%d-th agent Test NRMSE using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_test_umg), std(rslt_test_umg));
            fprintf("==>%d-th fold: Test NRMSE using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_test_umg), std(rslt_test_umg));
        end
        result_train_umg_arr(i) = mean(rslt_train_umg);
        result_test_umg_arr(i) = mean(rslt_test_umg);
        
        
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
    save_dir = sprintf('./results/%s_d_fg_semifnn_d_r%d_l%.1f_g%.4f_mu%.4f_rho_p%.4f_rho_s%.4f_beta%.4f_eta%.4f_al%.4f.mat',...
        data_name, n_rules, labeled_num, gamma, mu, rho_p,rho_s,beta,eta,alpha);
    save(save_dir, 'result_test_mean_g', 'result_test_std_g', 'result_train_mean_g', 'result_train_std_g',...
        'result_test_mean_l', 'result_test_std_l', 'result_train_mean_l', 'result_train_std_l',...
        'result_test_mean_lg', 'result_test_std_lg', 'result_train_mean_lg', 'result_train_std_lg',...
        'result_test_mean_um', 'result_test_std_um', 'result_train_mean_um', 'result_train_std_um',...
        'result_test_mean_ug', 'result_test_std_ug', 'result_train_mean_ug', 'result_train_std_ug',...
        'result_test_mean_umg', 'result_test_std_umg', 'result_train_mean_umg', 'result_train_std_umg');

    
    fprintf(fid, "Train acc with all training data involved: %.4f/%.4f  %.4f\n", result_train_mean_g, result_train_std_g, time_g);
    fprintf(fid, "Test acc with all training data involved: %.4f/%.4f  %.4f\n", result_test_mean_g, result_test_std_g, time_g);
    fprintf(fid, "Train acc with only labeled data involved: %.4f/%.4f  %.4f\n", result_train_mean_l, result_train_std_l, time_l);
    fprintf(fid, "Test acc with only labeled data involved: %.4f/%.4f  %.4f\n", result_test_mean_l, result_test_std_l, time_l);
    fprintf(fid, "Train acc with all training data only involved kmeans part: %.4f/%.4f  %.4f\n", result_train_mean_lg, result_train_std_lg, time_lg);
    fprintf(fid, "Test acc with all training data only involved kmeans part: %.4f/%.4f  %.4f\n", result_test_mean_lg, result_test_std_lg, time_lg);
    fprintf(fid, "Train acc using graph: %.4f/%.4f  %.4f\n", result_train_mean_ug, result_train_std_ug, time_ug);
    fprintf(fid, "Test acc using graph: %.4f/%.4f  %.4f\n", result_test_mean_ug, result_test_std_ug, time_ug);
    fprintf(fid, "Train acc using mix-up: %.4f/%.4f  %.4f\n", result_train_mean_um, result_train_std_um, time_um);
    fprintf(fid, "Test acc using mix-up: %.4f/%.4f  %.4f\n", result_test_mean_um, result_test_std_um, time_um);
    fprintf(fid, "Train acc using mix-up and graph: %.4f/%.4f  %.4f\n", result_train_mean_umg, result_train_std_umg, time_umg);
    fprintf(fid, "Test acc using mix-up and graph: %.4f/%.4f  %.4f\n", result_test_mean_umg, result_test_std_umg, time_umg);
    
    fprintf("Train acc with all training data involved: %.4f/%.4f  %.4f\n", result_train_mean_g, result_train_std_g, time_g);
    fprintf("Test acc with all training data involved: %.4f/%.4f  %.4f\n", result_test_mean_g, result_test_std_g, time_g);
    fprintf("Train acc with only labeled data involved: %.4f/%.4f  %.4f\n", result_train_mean_l, result_train_std_l, time_l);
    fprintf("Test acc with only labeled data involved: %.4f/%.4f  %.4f\n", result_test_mean_l, result_test_std_l, time_l);
    fprintf("Train acc with all training data only involved kmeans part: %.4f/%.4f  %.4f\n", result_train_mean_lg, result_train_std_lg, time_lg);
    fprintf("Test acc with all training data only involved kmeans part: %.4f/%.4f  %.4f\n", result_test_mean_lg, result_test_std_lg, time_lg);
    fprintf("Train acc using graph: %.4f/%.4f  %.4f\n", result_train_mean_ug, result_train_std_ug, time_ug);
    fprintf("Test acc using graph: %.4f/%.4f  %.4f\n", result_test_mean_ug, result_test_std_ug, time_ug);
    fprintf("Train acc using mix-up: %.4f/%.4f  %.4f\n", result_train_mean_um, result_train_std_um, time_um);
    fprintf("Test acc using mix-up: %.4f/%.4f  %.4f\n", result_test_mean_um, result_test_std_um, time_um);
    fprintf("Train acc using mix-up and graph: %.4f/%.4f  %.4f\n", result_train_mean_umg, result_train_std_umg, time_umg);
    fprintf("Test acc using mix-up and graph: %.4f/%.4f  %.4f\n", result_test_mean_umg, result_test_std_umg, time_umg);
    
    fclose(fid);
end