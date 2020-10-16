function [rslt_agent]=semi_fnn_fg_agent_analysis(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,gamma,eta,alpha,...
    n_agent_list)
    
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
    fig_path = sprintf('./fig/%s/agent/', data_name);
    if exist(fig_path,'dir')==0
       mkdir(fig_path);
    end
%     n_agent_list = [2, 5,10,15,20,25,30,35,40,45,50];
    rslt_agent = zeros(4, length(n_agent_list)); % each row stands for a parameter (mu, rho_p, rho_s, gamma, eta)

    % test different mixup numbers
    for ii=1:length(n_agent_list)
        [result_train_mean_um, result_train_std_um, result_test_mean_um, result_test_std_um, ~]=...
        semi_fnn_fg_func_d_um(data_name, kfolds, n_rules, n_agent_list(ii), n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
        gamma,alpha);
        rslt_agent(1,ii) = result_train_mean_um;
        rslt_agent(2,ii) = result_test_mean_um;
        rslt_agent(3,ii) = result_train_std_um;
        rslt_agent(4,ii) = result_test_std_um;
    end
    
    figure(1);
    car_idx = 1:length(n_agent_list);
    plot(n_agent_list',rslt_agent(1,:)','b-s',...
        n_agent_list',rslt_agent(2,:)','g-+','LineWidth',1.3);
    
    h_leg =legend('training', 'test');
    xlabel('Agent Number','fontsize',16);
    ylabel('NRMSE','fontsize',16);
    xticks(car_idx);
    title('Performances of semi-FNN on different agent numbers');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%sagent_ana_%s.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
%     title('');
%     save_name = sprintf('%sagent_ana_%s.pdf', fig_path, data_name);
%     print(save_name, '-dpdf')
%     close;
    
    figure(11);
    car_idx = 1:length(n_agent_list);
    errorbar(n_agent_list',rslt_agent(1,:)', rslt_agent(3,:)','b-s','LineWidth',1.3);
    hold on;
    errorbar(n_agent_list',rslt_agent(2,:)', rslt_agent(4,:)','g-+','LineWidth',1.3);
    h_leg =legend('training', 'test');
    xlabel('Agent Number','fontsize',16);
    ylabel('NRMSE','fontsize',16);
    xticks(car_idx);
    title('Performances of semi-FNN on different agent numbers');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
    set(gca,'FontSize',20);
    save_name = sprintf('%sagent_ana_%s_e.fig', fig_path, data_name);
    saveas(gcf,save_name);
    
    figure(2);
    car_idx = 1:length(n_agent_list);
    plot(n_agent_list',rslt_agent(2,:)','g-+','LineWidth',1.3);
    
    xlabel('Agent Number','fontsize',16);
    ylabel('NRMSE','fontsize',16);
    xticks(car_idx);
    title('Performances of semi-FNN on different agent numbers');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(gca,'FontSize',20);
    save_name = sprintf('%sagent_ana_%s_t.fig', fig_path, data_name);
    saveas(gcf,save_name);
    
    figure(22);
    car_idx = 1:length(n_agent_list);
    errorbar(n_agent_list',rslt_agent(2,:)', rslt_agent(4,:)','g-+','LineWidth',1.3);
    
    xlabel('Agent Number','fontsize',16);
    ylabel('NRMSE','fontsize',16);
    xticks(car_idx);
    title('Performances of semi-FNN on different agent numbers');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(gca,'FontSize',20);
    save_name = sprintf('%sagent_ana_%s_t_e.fig', fig_path, data_name);
    saveas(gcf,save_name);
    
    save_dir = sprintf('%s%s_agent_ana.mat',...
        fig_path, data_name);
    save(save_dir, 'rslt_agent');
end