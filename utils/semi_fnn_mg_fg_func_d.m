function [result_test_mean_umg, result_test_std_umg, result_train_mean_umg, result_train_std_umg] = ...
semi_fnn_mg_fg_func_d(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
    gamma,eta, alpha)
    
    fprintf("==================================dataset: %s================================ \n", data_name);
    fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf("============================================================================= \n");
    fprintf("========gamma: %f, mu: %f, rho_p: %f,rho_s: %.4f, beta:%.4f, eta: %f:, alpha: %f: =========== \n",...
         gamma, mu, rho_p, rho_s,beta, eta,alpha);
    time_local = datestr(now,0);
    fprintf("============================================================================= \n");
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

    result_test_umg_arr = zeros(1,kfolds); 
    result_train_umg_arr = zeros(1,kfolds);

    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data_d(n_agent, i);
        
        min_n_smple_d_l = floor(labeled_num/n_agent);
        max_n_smple_d_l = ceil(labeled_num/n_agent);
        div_n_smple_d_l = labeled_num - min_n_smple_d_l*n_agent;
        train_data.n_smpl_d_l = floor(labeled_num/n_agent);
        shuffle_idx = randperm(train_data.n_smpl_d);
        for ii=1:n_agent
            if ii<= div_n_smple_d_l
                train_data.n_smpl_d_l = max_n_smple_d_l;
            else
                train_data.n_smpl_d_l = min_n_smple_d_l;
            end
            
            train_data.fea_d_l = train_data.fea_d(:,shuffle_idx(1:train_data.n_smpl_d_l),:);
            train_data.gnd_d_l = train_data.gnd_d(:,shuffle_idx(1:train_data.n_smpl_d_l));
            train_data.fea_d_u = train_data.fea_d(:,shuffle_idx(train_data.n_smpl_d_l+1:end),:);
            train_data.gnd_d_u = train_data.gnd_d(:,shuffle_idx(train_data.n_smpl_d_l+1:end));
            
        end
        train_data.n_smpl_d_u = size(train_data.fea_d_u,2);
        
        
        % ============== semi-fnn all data involved in kmeans============
        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(squeeze(train_data.fea_d(1,:,:)), n_rules);
        [mu_optimal_g, ~] = Fuzzycmeans_ADMM(train_data.fea_d, n_rules, n_agent,rho_s,beta);
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
        rule_lg = rule_g.init_fuzzyc(squeeze(train_data.fea_d(1,:,:)), n_rules);
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
        
        
        % ==============semi-fnn using mix-up and graph============% rule_umg = rule_g;

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

        for ii = 1:n_agent
            rule_umg_arr{ii}.width = width_umg;
            H_train_arr_umg{ii} = ComputeH(squeeze(train_data.fea_d_l(ii,:,:)),rule_umg_arr{ii});
            y_hat_train_umg{ii} = H_train_arr_umg{ii} * w_d_umg;
            H_test_arr_umg{ii} = ComputeH(test_data.fea,rule_umg_arr{ii});
            y_hat_test_umg{ii} = H_test_arr_umg{ii} * w_d_umg;
            if task=='C'
                rslt_train_umg(ii) = calculate_acc(train_data.gnd_d_l(ii,:)', y_hat_train_umg{ii});
                fprintf("==>%d-th agent Train acc using mix-up and graph: %f\n", ii, rslt_train_umg(ii));
                rslt_test_umg(ii) = calculate_acc(test_data.gnd, y_hat_test_umg{ii});
                fprintf("==>%d-th agent Test acc using mix-up and graph: %f\n", ii, rslt_test_umg(ii));
            else
                rslt_train_umg(ii) = calculate_nrmse(train_data.gnd_d_l(ii,:)', y_hat_train_umg{ii});
                fprintf("==>%d-th agent Train NRMSE using mix-up and graph: %f\n", ii, rslt_train_umg(ii));
                rslt_test_umg(ii) = calculate_nrmse(test_data.gnd, y_hat_test_umg{ii});
                fprintf("==>%d-th agent Test NRMSE using mix-up and graph: %f\n", ii, rslt_test_umg(ii));
            end
            
        end
        
        
        if task=='C'
            fprintf("==>%d-th fold: Train acc using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_train_umg), std(rslt_train_umg));
            fprintf("==>%d-th fold: Test acc using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_test_umg), std(rslt_test_umg));
        else
            fprintf("==>%d-th fold: Train NRMSE using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_train_umg), std(rslt_train_umg));
            fprintf("==>%d-th fold: Test NRMSE using mix-up and graph: %.4f/%.4f\n", i, mean(rslt_test_umg), std(rslt_test_umg));
        end
        result_train_umg_arr(i) = mean(rslt_train_umg);
        result_test_umg_arr(i) = mean(rslt_test_umg);
        
        
        
    end

    
    result_test_mean_umg = mean(result_test_umg_arr); 
    result_test_std_umg = std(result_test_umg_arr);

    
    result_train_mean_umg = mean(result_train_umg_arr); 
    result_train_std_umg = std(result_train_umg_arr);

   
end