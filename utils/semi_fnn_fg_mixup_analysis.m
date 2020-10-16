function [rslt_mixup]=semi_fnn_fg_mixup_analysis(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,gamma,eta,alpha,...
    n_mixup_list,n_mixup_list_str)
    
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
    fig_path = sprintf('./fig/%s/mixup/', data_name);
    if exist(fig_path,'dir')==0
       mkdir(fig_path);
    end
%     n_mixup_list = [1000,5000,10000, 50000,100000, 500000];
%     n_mixup_list = [100,500,1000, 5000,10000, 50000];
    rslt_mixup = zeros(4, length(n_mixup_list)); % each row stands for a parameter (mu, rho_p, rho_s, gamma, eta)

    % test different mixup numbers
    for ii=1:length(n_mixup_list)
        [result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, ~]=...
        semi_fnn_fg_func_d_um(data_name, kfolds, n_rules, n_agent, n_mixup_list(ii), labeled_num, mu,rho_p,rho_s,beta,...
        gamma,alpha);
        rslt_mixup(1,ii) = result_train_mean_um;
        rslt_mixup(2,ii) = result_test_mean_um;
        rslt_mixup(3,ii) = result_train_std_um;
        rslt_mixup(4,ii) = result_test_std_um;
    end
    
    figure(11);
    car_idx = 1:length(n_mixup_list);
    plot(car_idx',rslt_mixup(1,:)','b-s',...
        car_idx',rslt_mixup(2,:)','g-+','LineWidth',1.3);
%     errorbar(car_idx',rslt_mixup(1,:)', rslt_mixup(3,:)','b-s','LineWidth',1.3);
    
    h_leg =legend('training', 'test');
    xlabel('Mix-up Sample Number','fontsize',16);
    xticks(car_idx);
    xticklabels(n_mixup_list_str);
    ylabel('NRMSE','fontsize',16);
    title('Influence of mixup numbers');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',16);
    save_name = sprintf('%smixup_ana_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
%     title('');
%     save_name = sprintf('%smixup_ana_%s.pdf', fig_path, data_name);
%     print(save_name, '-dpdf')
%     close;

    figure(22);
    car_idx = 1:length(n_mixup_list);
    plot(car_idx',rslt_mixup(2,:)','b-*','LineWidth',1.3);

    xlabel('Mix-up Sample Number','fontsize',16);
    xticks(car_idx);
    xticklabels(n_mixup_list_str);
    ylabel('NRMSE','fontsize',16);
    title('Influence of mixup numbers');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(gca,'FontSize',16);
    save_name = sprintf('%smixup_ana_%s_t.fig', fig_path, data_name);
    saveas(gcf,save_name);

    figure(1);
    car_idx = 1:length(n_mixup_list);
%     errorbar(car_idx',rslt_mixup(1,:)','b-s',...
%         car_idx',rslt_mixup(2,:)','g-+','LineWidth',1.3);
    errorbar(car_idx',rslt_mixup(1,:)', rslt_mixup(3,:)','b-s','LineWidth',1.3);
    hold on;
    errorbar(car_idx',rslt_mixup(2,:)', rslt_mixup(4,:)','g-+','LineWidth',1.3);
    
    h_leg =legend('training', 'test');
    xlabel('Mix-up Sample Number','fontsize',16);
    xticks(car_idx);
    xticklabels(n_mixup_list_str);
    ylabel('NRMSE','fontsize',16);
    title('Influence of mixup numbers');
    set (gcf,'Position',[0,0,700,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',16);
    save_name = sprintf('%smixup_ana_%s_e.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
%     title('');
%     save_name = sprintf('%smixup_ana_%s.pdf', fig_path, data_name);
%     print(save_name, '-dpdf')
%     close;

    figure(2);
    car_idx = 1:length(n_mixup_list);
    errorbar(car_idx',rslt_mixup(2,:)', rslt_mixup(4,:)','b-*','LineWidth',1.3);
    
    xlabel('Mix-up Sample Number','fontsize',16);
    xticks(car_idx);
    xticklabels(n_mixup_list_str);
    ylabel('NRMSE','fontsize',16);
    title('Influence of mixup numbers');
    set (gcf,'Position',[0,0,700,500], 'color','w');
    set(gca,'FontSize',16);
    save_name = sprintf('%smixup_ana_%s_e_t.fig', fig_path, data_name);
    saveas(gcf,save_name);
    
    
    save_dir = sprintf('%s%s_mixup_ana.mat',...
        fig_path, data_name);
    save(save_dir, 'rslt_mixup');
end