function semi_fnn_fg_func_c_log(data_name, kfolds, n_rules, n_mixup, labeled_num, mu, gamma, eta, alpha, beta)
    log_path = './log/';
    if exist(log_path,'dir')==0
       mkdir(log_path);
    end
    logfilename = sprintf('%s%s_c_fg_semifnn_r%d_l%.1f_mu%.4f_eta%.4f_gamma%.4f.txt',...
        log_path, data_name, n_rules, labeled_num, mu, eta, gamma);
    time_local = datestr(now,0);
    fid = fopen(logfilename,'at');
    fprintf(fid, "==================================dataset: %s================================ \n", data_name);
    fprintf(fid, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf(fid, "============================================================================= \n");
    fprintf(fid, "==========rule number: %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, kfolds, labeled_num);
    fprintf(fid, "========mu: %f, eta: %f, gamma: %f, alpha: %f: , beta: %f:=========== \n",...
         mu, eta, gamma, alpha, beta);
    fprintf(fid, "============================================================================= \n");
    
    fprintf("==================================dataset: %s================================ \n", data_name);
    fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf("============================================================================= \n");
    fprintf("==========rule number: %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, kfolds, labeled_num);
    fprintf("========mu: %f, eta: %f, gamma: %f, alpha: %f: , beta: %f:=========== \n",...
         mu, eta, gamma, alpha, beta);
    fprintf("============================================================================= \n");
    fprintf(fid, "===================time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
    fprintf("===================time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
        
    % =================treat all data as labeled data (g) =================== 

    [result_train_mean_g, result_train_std_g, result_test_mean_g, result_test_std_g, time_g]=...
        semi_fnn_fg_func_c_g(data_name, kfolds, n_rules, labeled_num, mu,beta,fid);
   
    % ============== semi-fnn all data involved in fcmeans (lg)============
    
    [result_train_mean_lg, result_train_std_lg, result_test_mean_lg, result_test_std_lg, time_lg]=...
        semi_fnn_fg_func_c_lg(data_name, kfolds, n_rules, labeled_num, mu, beta, fid);

    % ==============all data involved (ug)============%
    
    [result_train_mean_ug, result_train_std_ug, result_test_mean_ug, result_test_std_ug, time_ug]=...
        semi_fnn_fg_func_c_ug(data_name, kfolds, n_rules, labeled_num, mu, eta, beta, fid);
    
    
    % ==============semi-fnn (um)============%
    
    [result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, time_um]=...
        semi_fnn_fg_func_c_um(data_name, kfolds, n_rules, n_mixup, labeled_num, mu, gamma, alpha, beta, fid);
    
    
    if exist('./results','dir')==0
       mkdir('./results');
    end
    
    save_dir = sprintf('./results/%s_c_fg_semifnn_r%d_l%.1f_mu%.4f_eta%.4f_gamma%.4f.mat',...
        data_name, n_rules, labeled_num, mu, eta, gamma);
    
    save(save_dir, 'result_test_mean_g', 'result_test_std_g', 'result_train_mean_g', 'result_train_std_g',...
        'result_test_mean_lg', 'result_test_std_lg', 'result_train_mean_lg', 'result_train_std_lg',...
        'result_test_mean_um', 'result_test_std_um', 'result_train_mean_um', 'result_train_std_um',...
        'result_test_mean_ug', 'result_test_std_ug', 'result_train_mean_ug', 'result_train_std_ug');

    
    fprintf(fid, "results: [Train Test time](g): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_g, result_train_std_g, result_test_mean_g, result_test_std_g, time_g);
    fprintf(fid, "results: [Train Test time](lg): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_lg, result_train_std_lg, result_test_mean_lg, result_test_std_lg, time_lg);
    fprintf(fid, "results: [Train Test time](ug): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_ug, result_train_std_ug, result_test_mean_ug, result_test_std_ug, time_ug);
    fprintf(fid, "results: [Train Test time](um): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, time_um);
    
    fprintf("results: [Train Test time](g): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_g, result_train_std_g, result_test_mean_g, result_test_std_g, time_g);
    fprintf("results: [Train Test time](lg): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_lg, result_train_std_lg, result_test_mean_lg, result_test_std_lg, time_lg);
    fprintf("results: [Train Test time](ug): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_ug, result_train_std_ug, result_test_mean_ug, result_test_std_ug, time_ug);
    fprintf("results: [Train Test time](um): & %.4f/%.4f  & %.4f/%.4f & %.4f\n",...
        result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, time_um);
    
    fclose(fid);
end