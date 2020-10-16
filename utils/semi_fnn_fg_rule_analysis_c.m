function [rslt_rule]=semi_fnn_fg_rule_analysis_c(data_name, kfolds, n_rules, n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,gamma,eta,alpha)
    
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
    fig_path = sprintf('./fig/%s/rule_c/', data_name);
    if exist(fig_path,'dir')==0
       mkdir(fig_path);
    end
    n_rule_list = 2:30;
    rslt_rule = zeros(2, length(n_rule_list)); % each row stands for a parameter (mu, rho_p, rho_s, gamma, eta)

    % test different mixup numbers
    for ii=1:length(n_rule_list)
        [train_mean_um, ~, test_mean_um, ~, ~] = semi_fnn_fg_func_c_um(data_name, kfolds, n_rule_list(ii), n_mixup, labeled_num, mu, gamma, alpha, beta);
%         semi_fnn_fg_func_d_um(data_name,...
%             kfolds, n_rule_list(ii), n_agent, n_mixup, labeled_num, mu,rho_p,rho_s,beta,...
%             gamma,alpha);
        rslt_rule(1,ii) = train_mean_um;
        rslt_rule(2,ii) = test_mean_um;
    end
    
    figure(1);
    car_idx = 1:length(n_rule_list);
    plot(car_idx',rslt_rule(1,:)','b-s',...
        car_idx',rslt_rule(2,:)','g-+','LineWidth',1.3);
    
    h_leg =legend('training', 'test');
    xlabel('Rule Number','fontsize',16);
%     xticklabels({'2','5','10','15','20','30'});
    ylabel('NRMSE','fontsize',16);
    title('Rule Performance C');
    set (gcf,'Position',[0,0,800,500], 'color','w');
    set(h_leg,'position',[0.060 0.783 0.160714285714286 0.0555555555555556]);
    set(h_leg,'Units','Normalized','FontUnits','Normalized')%ÕâÊÇ·ÀÖ¹±ä»¯Ê±£¬²úÉú½Ï´óµÄÐÎ±ä¡£
%     set(gca,'FontSize',20);
    save_name = sprintf('%srule_ana_%s_c.fig', fig_path, data_name);
    saveas(gcf,save_name);
    % axis([0 200 0.1 0.8]); 
    title('');
    save_name = sprintf('%srule_ana_%s_c.pdf', fig_path, data_name);
    print(save_name, '-dpdf')
%     close;
    
    save_dir = sprintf('%s%s_rule_ana_c.mat',...
        fig_path, data_name);
    save(save_dir, 'rslt_rule');
end