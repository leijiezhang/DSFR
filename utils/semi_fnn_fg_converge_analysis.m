function semi_fnn_fg_converge_analysis(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
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

    fig_path = sprintf('./fig/%s/convergence/', data_name);
    if exist(fig_path,'dir')==0
       mkdir(fig_path);
    end
    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

   
    [train_data, ~] = dataset.get_kfold_data_d(n_agent, 1);
    
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
    figure(1);
    
%     plot(car_idx',lei_loss_c_test(1,:)','m-*',...
%         car_idx',lei_loss_c_test(2,:)','b-s',...
%         car_idx',lei_loss_c_test(3,:)','r-*',...
%         car_idx',lei_loss_c_test(4,:)','g-+','LineWidth',0.3);
    
    

    % =================treat all data as labeled data ===================   
    rule_g =  Rules(n_rules);
    rule_g = rule_g.init_fuzzyc(train_data.fea_d{1}, n_rules,beta);
    [mu_optimal_g, trainInfo_fc] = Fuzzycmeans_ADMM(train_data.fea_d, n_rules, n_agent,rho_s,beta);
    plot(trainInfo_fc.error(1:trainInfo_fc.consensus_steps),'b-*','LineWidth',1.3);
%     h_leg =legend('FCM_ADMM','FNN_ADMM','FNN_G_ADMM','FNN_M_ADMM','FNN_GM_ADMM');
    xlabel('Iteration','fontsize',16);
%     xticklabels({'10%','','','','','20%', '','','','','30%'})
    ylabel('Loss','fontsize',16);
    title('Loss of FCM ADMM methods');
    set (gcf,'Position',[0,0,800,500], 'color','w');
%     set(h_leg,'position',[0.164 0.783 0.160714285714286 0.0555555555555556]);
%     set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%sconv_fcm_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
    title('');
    print -dpdf conv_para.pdf;
    close;

    rule_g_arr = cell(1, n_agent);
    H_train_arr_g = cell(1, n_agent);
    w_train_arr_g = zeros(n_rules*(train_data.n_fea+1), n_agent);

    width_tmp_g = zeros(size(rule_g.width));
    for ii = 1:n_agent
        rule_g_arr{ii}= rule_g.update_fuzzyc(train_data.fea_d{ii}, mu_optimal_g, beta);
        width_tmp_g = width_tmp_g + train_data.n_smpl_d*(rule_g_arr{ii}.width.^2);
    end
    width_g = sqrt(width_tmp_g/(n_agent*train_data.n_smpl_d));

    for ii = 1:n_agent
        rule_g_arr{ii}.width = width_g;
        H_train_arr_g{ii} = ComputeH(train_data.fea_d{ii},rule_g_arr{ii});
        [~, w_train_arr_g(:,ii)] = FNN_solve(H_train_arr_g{ii}, train_data.gnd_d{ii}', mu);
    end
    [~, trainInfo_g] = FNN_ADMM(train_data.gnd_d, n_agent, w_train_arr_g, H_train_arr_g, mu, rho_p);
    plot(trainInfo_g.error(1:trainInfo_g.consensus_steps),'b-*','LineWidth',1.3);
    %     h_leg =legend('FCM_ADMM','FNN_ADMM','FNN_G_ADMM','FNN_M_ADMM','FNN_GM_ADMM');
    xlabel('Iteration','fontsize',16);
%     xticklabels({'10%','','','','','20%', '','','','','30%'})
    ylabel('Loss','fontsize',16);
    title('Loss of FNN ADMM methods');
    set (gcf,'Position',[0,0,800,500], 'color','w');
%     set(h_leg,'position',[0.164 0.783 0.160714285714286 0.0555555555555556]);
%     set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%sconv_fnn_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
    title('');
    print -dpdf conv_para.pdf;
    close;
    
    % ============== semi-fnn all data involved in kmeans============

    rule_lg = rule_g.init_fuzzyc(train_data.fea_d{ii}, n_rules,beta);
%         [mu_optimal_lg, ~] = Kmeans_ADMM(train_data.fea_d, n_rules, n_agent);
%         rule_g = rule_g.update_kmeans(train_data.fea, mu_optimal_g);
    rule_lg_arr = cell(1, n_agent);
    H_train_arr_lg = cell(1, n_agent);
    w_train_arr_lg = zeros(n_rules*(train_data.n_fea+1), n_agent);

    width_tmp_lg = zeros(size(rule_g.width));
    for ii = 1:n_agent
        rule_lg_arr{ii}= rule_lg.update_fuzzyc(train_data.fea_d{ii}, mu_optimal_g, beta);
        width_tmp_lg = width_tmp_lg + train_data.n_smpl_d*(rule_lg_arr{ii}.width.^2);
    end
    width_lg = sqrt(width_tmp_lg/(n_agent*train_data.n_smpl_d));

    for ii = 1:n_agent
        rule_lg_arr{ii}.width = width_lg;
        H_train_arr_lg{ii} = ComputeH(train_data.fea_d_l{ii},rule_lg_arr{ii});
        [~, w_train_arr_lg(:,ii)] = FNN_solve(H_train_arr_lg{ii}, train_data.gnd_d_l{ii}', mu);
    end
    
    % ==============all data involved using graph============% labeled data and unlabeled data
    [~, trainInfo_ug] = graph_FNN_ADMM(train_data, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p,...
         H_train_arr_g, eta);
    plot(trainInfo_ug.error(1:trainInfo_ug.consensus_steps),'b-*','LineWidth',1.3);
    %     h_leg =legend('FCM_ADMM','FNN_ADMM','FNN_G_ADMM','FNN_M_ADMM','FNN_GM_ADMM');
    xlabel('Iteration','fontsize',16);
%     xticklabels({'10%','','','','','20%', '','','','','30%'})
    ylabel('Loss','fontsize',16);
    title('Loss of FNN\_G ADMM methods');
    set (gcf,'Position',[0,0,800,500], 'color','w');
%     set(h_leg,'position',[0.164 0.783 0.160714285714286 0.0555555555555556]);
%     set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%sconv_fnn_g_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
    title('');
    print -dpdf conv_para.pdf;
    close;
    % ==============semi-fnn using mix-up============% labeled data and unlabeled data
    rule_um = rule_g;
    [~, trainInfo_um] = mixup_FNN_ADMM(train_data, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p,...
        rule_um, n_mixup, gamma, alpha);
    plot(trainInfo_um.error(1:trainInfo_um.consensus_steps),'b-*','LineWidth',1.3);    %     h_leg =legend('FCM_ADMM','FNN_ADMM','FNN_G_ADMM','FNN_M_ADMM','FNN_GM_ADMM');    xlabel('Iteration','fontsize',16);
%     xticklabels({'10%','','','','','20%', '','','','','30%'})
    xlabel('Iteration','fontsize',16);
    ylabel('Loss','fontsize',16);
    title('Loss of FNN\_M ADMM methods');
    set (gcf,'Position',[0,0,800,500], 'color','w');
%     set(h_leg,'position',[0.164 0.783 0.160714285714286 0.0555555555555556]);
%     set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%sconv_fnn_m_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
    title('');
    print -dpdf conv_para.pdf;
    close;
    % ==============semi-fnn using mix-up and graph============% rule_umg = rule_g;
    rule_umg = rule_g;
    [~, trainInfo_umg] = graph_mixup_FNN_ADMM(train_data, n_agent, w_train_arr_lg, H_train_arr_lg, mu, rho_p,...
        rule_umg, n_mixup, gamma,alpha,...
        H_train_arr_g, eta);
    plot(trainInfo_umg.error(1:trainInfo_umg.consensus_steps),'b-*','LineWidth',1.3);
    %     h_leg =legend('FCM_ADMM','FNN_ADMM','FNN_G_ADMM','FNN_M_ADMM','FNN_GM_ADMM');
    xlabel('Iteration','fontsize',16);
%     xticklabels({'10%','','','','','20%', '','','','','30%'})
    ylabel('Loss','fontsize',16);
    title('Loss of FNN\_MG ADMM methods');
    set (gcf,'Position',[0,0,800,500], 'color','w');
%     set(h_leg,'position',[0.164 0.783 0.160714285714286 0.0555555555555556]);
%     set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%sconv_fnn_mg_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
    title('');
    print -dpdf conv_para.pdf;
    close;
    save_dir = sprintf('%s%s_conv_ana.mat',...
        fig_path, data_name);
    save(save_dir, 'trainInfo_fc', 'trainInfo_g', 'trainInfo_um', 'trainInfo_ug','trainInfo_umg');
end