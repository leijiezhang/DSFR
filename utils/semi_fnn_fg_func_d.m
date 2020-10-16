function semi_fnn_fg_func_d(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
    gamma,eta, alpha)
    log_path = './log/';
    if exist(log_path,'dir')==0
       mkdir(log_path);
    end
    logfilename = sprintf('%s%s_d_fg_semifnn_r%d_l%.1f_rho_s%.4f_rho_p%.4f_mu%.4f_eta%.4f_gamma%.4f.txt',...
        log_path, data_name, n_rules, labeled_num, rho_s, rho_p, mu,eta, gamma);
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
    fprintf(fid, "===================Distributed method: time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
    fprintf("===================Distributed method: time: %s, n_mixup: %d==============================\n", time_local, n_mixup);
    
    
    % =================treat all data as labeled data (g) ===================
    
    [result_train_mean_g, result_train_std_g, result_test_mean_g, result_test_std_g, time_g]=...
    semi_fnn_fg_func_d_g(data_name, kfolds, n_rules, n_agent, labeled_num, mu,rho_p,rho_s,beta);
    
    % ============== semi-fnn all data involved in kmeans (lg)============
    

    [result_train_mean_lg, result_train_std_lg, result_test_mean_lg, result_test_std_lg, time_lg]=...
        semi_fnn_fg_func_d_lg(data_name, kfolds, n_rules, n_agent, labeled_num, mu,rho_p,rho_s,beta);
    % ==============all data involved using graph (ug)============% 
    
    [result_train_mean_ug, result_train_std_ug, result_test_mean_ug, result_test_std_ug, time_ug]=...
        semi_fnn_fg_func_d_ug(data_name, kfolds, n_rules, n_agent, labeled_num, mu,rho_p,rho_s,beta,...
    eta);
    % ==============semi-fnn using mix-up (um)============%
    
    [result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, time_um]=...
        semi_fnn_fg_func_d_um(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
    gamma,alpha);

    if exist('./results','dir')==0
       mkdir('./results');
    end
    save_dir = sprintf('./results/%s_d_fg_semifnn_d_r%d_l%.1f_rho_s%.4f_rho_p%.4f_mu%.4f_eta%.4f_gamma%.4f.mat',...
        data_name, n_rules, labeled_num, rho_s, rho_p, mu,eta, gamma);
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