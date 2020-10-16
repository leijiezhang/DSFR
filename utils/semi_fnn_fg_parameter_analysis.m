function [rslt_para_train, rslt_para_test]=semi_fnn_fg_parameter_analysis(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,gamma,eta,alpha)
    
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
    fig_path = sprintf('./fig/%s/para/', data_name);
    if exist(fig_path,'dir')==0
       mkdir(fig_path);
    end
    para_list = [1e-5, 1e-4, 1e-3,1e-2,1e-1, 1];
    rslt_para_train = zeros(5, length(para_list)); % each row stands for a parameter (mu, rho_p, rho_s, gamma, eta)
    rslt_para_test = zeros(5, length(para_list));
    % test para mu
    for ii=1:length(para_list)
        [train_mean_umg, ~, test_mean_umg, ~, ~] = semi_fnn_fg_func_d_um(data_name,...
            kfolds, n_rules, n_agent, n_mixup, labeled_num, para_list(ii),rho_p,rho_s,beta,...
            gamma,alpha);
        rslt_para_train(1,ii) = train_mean_umg;
        rslt_para_test(1,ii) = test_mean_umg;
    end
    % test para rho_p
    for ii=1:length(para_list)
        [train_mean_umg, ~, test_mean_umg, ~, ~] = semi_fnn_fg_func_d_um(data_name,...
            kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,para_list(ii),rho_s,beta,...
            gamma,alpha);
        rslt_para_train(2,ii) = train_mean_umg;
        rslt_para_test(2,ii) = test_mean_umg;
    end
    % test para rho_s
    for ii=1:length(para_list)
        [train_mean_umg, ~, test_mean_umg, ~, ~] = semi_fnn_fg_func_d_um(data_name,...
            kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,para_list(ii),beta,...
            gamma,alpha);
        rslt_para_train(3,ii) = train_mean_umg;
        rslt_para_test(3,ii) = test_mean_umg;
    end
    % test para gamma
    for ii=1:length(para_list)
        [train_mean_umg, ~, test_mean_umg, ~, ~] = semi_fnn_fg_func_d_um(data_name,...
            kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
            para_list(ii),alpha);
        rslt_para_train(4,ii) = train_mean_umg;
        rslt_para_test(4,ii) = test_mean_umg;
    end
    
    figure(1);
    car_idx = 1:length(para_list);
    plot(car_idx',rslt_para_train(2,:)','b-s',...
        car_idx',rslt_para_train(3,:)','r-*','LineWidth',1.3);
    
    h_leg =legend('\rho_p', '\rho_s');
    xlabel('Values of Parameters','fontsize',16);
    xticklabels({'1E-5','1E-4','1E-3','1E-2','1E-1','1'});
    ylabel('NRMSE','fontsize',16);
    title('Training result on LapLacian parameter');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%spara_lap_ana_train_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
%     title('');
%     save_name = sprintf('%spara_lap_ana_train_%s.pdf', fig_path, data_name);
%     print(save_name, '-dpdf')
%     close;
    figure(11);
    car_idx = 1:length(para_list);
    plot(car_idx',rslt_para_train(1,:)','m-v',...
        car_idx',rslt_para_train(4,:)','g-+','LineWidth',1.3);
    
    h_leg =legend('\mu', '\gamma');
    xlabel('Values of Parameters','fontsize',16);
    xticklabels({'1E-5','1E-4','1E-3','1E-2','1E-1','1'});
    ylabel('NRMSE','fontsize',16);
    title('training result on ADMM parameter');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%spara_admm_ana_train_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
%     title('');
%     save_name = sprintf('%spara_admm_ana_train_%s.pdf', fig_path, data_name);
%     print(save_name, '-dpdf')
%     close;
    
    figure(2);
    car_idx = 1:length(para_list);
    plot(car_idx',rslt_para_test(2,:)','b-s',...
        car_idx',rslt_para_test(3,:)','r-*','LineWidth',1.3);
    
    h_leg =legend('\rho_p', '\rho_s');
    xlabel('Values of Parameters','fontsize',16);
    xticklabels({'1E-5','1E-4','1E-3','1E-2','1E-1','1'});
    ylabel('NRMSE','fontsize',16);
    title('Test result on LapLacian parameter');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%spara_lap_ana_test_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
%     title('');
%     save_name = sprintf('%spara_lap_ana_test_%s.pdf', fig_path, data_name);
%     print(save_name, '-dpdf')
%     close;
    figure(22);
    car_idx = 1:length(para_list);
    plot(car_idx',rslt_para_test(1,:)','m-v',...
        car_idx',rslt_para_test(4,:)','g-+','LineWidth',1.3);
    
    h_leg =legend('\mu', '\gamma');
    xlabel('Values of Parameters','fontsize',16);
    xticklabels({'1E-5','1E-4','1E-3','1E-2','1E-1','1'});
    ylabel('NRMSE','fontsize',16);
    title('Testing result of ADMM parameter');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%spara_admm_ana_test_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
%     title('');
%     save_name = sprintf('%spara_admm_ana_test_%s.pdf', fig_path, data_name);
%     print(save_name, '-dpdf')
%     close;
    
    save_dir = sprintf('%s%s_para_ana.mat',...
        fig_path, data_name);
    save(save_dir, 'rslt_para_test', 'rslt_para_test');
end