function [result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, time_um]=...
        semi_fnn_fg_func_d_um(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
    gamma,alpha, fid)
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);

    % Preprocess dataset and select parameters for the simulation

    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

    result_test_um_arr = zeros(1,kfolds); 
    result_train_um_arr = zeros(1,kfolds);
    
    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data_d(n_agent, i);
        
        min_n_smple_d_l = floor(labeled_num/n_agent);
        max_n_smple_d_l = ceil(labeled_num/n_agent);
        div_n_smple_d_l = labeled_num - min_n_smple_d_l*n_agent;
        train_data.n_smpl_d_l = floor(labeled_num/n_agent);
        shuffle_idx = randperm(train_data.n_smpl_d);
        train_data.fea_d_l = cell(n_agent,1);
        train_data.gnd_d_l = cell(n_agent,1);
        train_data.fea_d_u = cell(n_agent,1);
        gnd_d_tmp = cell(n_agent,1);
        fea_d_tmp = cell(n_agent,1);
        for ii=1:n_agent
            if ii<= div_n_smple_d_l
                n_smpl_d_l_agent = max_n_smple_d_l;
            else
                n_smpl_d_l_agent = min_n_smple_d_l;
            end
            gnd_d_tmp{ii} = train_data.gnd_d(ii,:);
            fea_d_tmp{ii} = squeeze(train_data.fea_d(ii,:,:));
            train_data.fea_d_l{ii} = fea_d_tmp{ii}(shuffle_idx(1:n_smpl_d_l_agent),:);
            train_data.gnd_d_l{ii} = gnd_d_tmp{ii}(:,shuffle_idx(1:n_smpl_d_l_agent));
            train_data.fea_d_u{ii} = fea_d_tmp{ii}(shuffle_idx(n_smpl_d_l_agent+1:end),:);
        end
        train_data.gnd_d = gnd_d_tmp;
        train_data.fea_d = fea_d_tmp;
       
        % ==============semi-fnn using mix-up============% 
        tic;
        rule_g =  Rules(n_rules);
        rule_g = rule_g.init_fuzzyc(train_data.fea_d{1}, n_rules,beta);
        [mu_optimal_g, ~] = Fuzzycmeans_ADMM(train_data.fea_d, n_rules, n_agent,rho_s,beta);
        rule_g_arr = cell(1, n_agent);
        H_train_arr_g = cell(1, n_agent);
        H_test_arr_g = cell(1, n_agent);
        w_train_arr_g = zeros(n_rules*(train_data.n_fea+1), n_agent);
        H_train_arr_lg = cell(1, n_agent);
        w_train_arr_lg = zeros(n_rules*(train_data.n_fea+1), n_agent);
        
        width_tmp_g = zeros(size(rule_g.width));
        for ii = 1:n_agent
            n_smpl_d_ii = size(train_data.fea_d{ii},1);
            rule_g_arr{ii}= rule_g.update_fuzzyc(train_data.fea_d{ii}, mu_optimal_g, beta);
            width_tmp_g = width_tmp_g + n_smpl_d_ii*(rule_g_arr{ii}.width.^2);
        end
        width_g = sqrt(width_tmp_g/(n_agent*train_data.n_smpl_d));
        
        for ii = 1:n_agent
            rule_g_arr{ii}.width = width_g;
            H_train_arr_g{ii} = ComputeH(train_data.fea_d{ii},rule_g_arr{ii});
            H_test_arr_g{ii} = ComputeH(test_data.fea,rule_g_arr{ii});
            [~, w_train_arr_g(:,ii)] = FNN_solve(H_train_arr_g{ii}, train_data.gnd_d{ii}', mu);
            H_train_arr_lg{ii} = ComputeH(train_data.fea_d_l{ii},rule_g_arr{ii});
            [~, w_train_arr_lg(:,ii)] = FNN_solve(H_train_arr_lg{ii}, train_data.gnd_d_l{ii}', mu);
        end
        [w_d_um, ~] = mixup_FNN_ADMM(train_data, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p,...
            rule_g, n_mixup, gamma, alpha);
        time_um = toc;        
%         for j = 1:n_rules 
%             rule_um.consq(j,:) = w_d_um((j - 1)*(train_data.n_fea_d + 1) + 1: j*(train_data.n_fea_d + 1))';
%         end 
        
        H_train_um = H_train_arr_lg;
        H_test_um = H_test_arr_g;
        y_hat_train_um = cell(1, n_agent);
        y_hat_test_um = cell(1, n_agent);
        rslt_train_um = zeros(n_agent,1);
        rslt_test_um = zeros(n_agent,1);
        for ii = 1:n_agent
            y_hat_train_um{ii} = H_train_um{ii} * w_d_um;
            
            y_hat_test_um{ii} = H_test_um{ii} * w_d_um;
            if task=='C'
                rslt_train_um(ii) = calculate_acc(train_data.gnd_d_l{ii}', y_hat_train_um{ii});
                rslt_test_um(ii) = calculate_acc(test_data.gnd, y_hat_test_um{ii});
                if nargin == 13
                    fprintf("==>%d-th agent Train acc using mix-up: %f\n", ii, rslt_train_um(ii));
                    fprintf("==>%d-th agent Test acc using mix-up: %f\n", ii, rslt_test_um(ii));
                    fprintf(fid, "==>%d-th agent Train acc using mix-up: %f\n", ii, rslt_train_um(ii));
                    fprintf(fid, "==>%d-th agent Test acc using mix-up: %f\n", ii, rslt_test_um(ii));
                end
                
            else
                rslt_train_um(ii) = calculate_nrmse(train_data.gnd_d_l{ii}', y_hat_train_um{ii});
                rslt_test_um(ii) = calculate_nrmse(test_data.gnd, y_hat_test_um{ii});
                if nargin == 13
                    fprintf("==>%d-th agent Train NRMSE using mix-up: %f\n", ii, rslt_train_um(ii));
                    fprintf("==>%d-th agent Test NRMSE using mix-up: %f\n", ii, rslt_test_um(ii));
                    fprintf(fid, "==>%d-th agent Train NRMSE using mix-up: %f\n", ii, rslt_train_um(ii));
                    fprintf(fid, "==>%d-th agent Test NRMSE using mix-up: %f\n", ii, rslt_test_um(ii));
                end
                
            end
            
        end
        
        
        if task=='C'
            if nargin == 13
                fprintf("==>%d-th fold: Train acc using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
                fprintf("==>%d-th fold: Test acc using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
                fprintf(fid, "==>%d-th fold: Train acc using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
                fprintf(fid, "==>%d-th fold: Test acc using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
            end
            
        else
            if nargin == 13
                fprintf("==>%d-th fold: Train NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
                fprintf("==>%d-th fold: Test NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
                fprintf(fid, "%d-th fold: ==>%d-th agent Train NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_train_um), std(rslt_train_um));
                fprintf(fid, "%d-th fold: ==>%d-th agent Test NRMSE using mix-up: %.4f/%.4f\n", i, mean(rslt_test_um), std(rslt_test_um));
            end
            
        end
        result_train_um_arr(i) = mean(rslt_train_um);
        result_test_um_arr(i) = mean(rslt_test_um);
        if nargin == 13 
            fprintf("==> %d-th fold finished!\n", i);
            fprintf(fid, "==> %d-th fold finished!\n", i);
        end
        
    end

    result_test_mean_um = mean(result_test_um_arr); 
    result_test_std_um = std(result_test_um_arr);
    
    result_train_mean_um = mean(result_train_um_arr); 
    result_train_std_um = std(result_train_um_arr);
    
end