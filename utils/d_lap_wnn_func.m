function d_lap_wnn_func(data_name, kfolds, n_rules, n_agent, labeled_num, lambda, eta, gamma)
log_path = './log/';
    if exist(log_path,'dir')==0
       mkdir(log_path);
    end
    logfilename = sprintf('%s%s_d_lap_wnn_r%d_l%.1f_lambda%.4f_eta%.4f_gamma%.4f.txt',...
        log_path, data_name, n_rules, labeled_num, lambda, eta, gamma);
    time_local = datestr(now,0);
    fid = fopen(logfilename,'at');
    fprintf(fid, "==================================dataset: %s================================ \n", data_name);
    fprintf(fid, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf(fid, "============================================================================= \n");
    fprintf(fid, "==========rule number: %d, agent number %d, k-fold: %d, labeled rate: %.2f ================ \n",...
        n_rules, n_agent, kfolds, labeled_num);
    fprintf(fid, "========gamma: %f, lambda: %f, eta: %f: =========== \n",...
         gamma, lambda, eta);
    fprintf(fid, "============================================================================= \n");
    
    fprintf("==================================dataset: %s================================ \n", data_name);
    fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
    fprintf("============================================================================= \n");
    fprintf( "========gamma: %f, lambda: %f, eta: %f: =========== \n",...
         gamma, lambda, eta);
  
    fprintf("============================================================================= \n");
    fprintf(fid, "===================time: %s==============================\n", time_local);
    fprintf("===================time: %s==============================\n", time_local);
    load_dir = sprintf('./data_norm/%s.mat', data_name);
    load(load_dir);
    % Preprocess dataset and select parameters for the simulation

    dataset = Dataset(X, Y, task, name);    % Load and preprocess dataset Data
    % set partition strategy
    partition_strategy = Partition(kfolds);
    n_smpl = size(dataset.fea,1);
    partition_strategy = partition_strategy.partition(n_smpl, true, 0);
    dataset = dataset.set_partition(partition_strategy); 

    result_test_g_arr = zeros(1,kfolds); 
    result_train_g_arr = zeros(1,kfolds);
    result_test_d_arr = zeros(1,kfolds); 
    result_train_d_arr = zeros(1,kfolds);
    l = 100;                                                           
    K = 10000; 

    for i=1:kfolds
        [train_data, test_data] = dataset.get_kfold_data_d(n_agent, i);
        train_data.n_smpl_d_l = labeled_num;
        train_data.n_smpl_d_l = ceil(labeled_num/n_agent);
        shuffle_idx = randperm(train_data.n_smpl_d);
        train_data.fea_d_l = train_data.fea_d(:,shuffle_idx(1:train_data.n_smpl_d_l),:);
        train_data.gnd_d_l = train_data.gnd_d(:,shuffle_idx(1:train_data.n_smpl_d_l));
        train_data.fea_d_u = train_data.fea_d(:,shuffle_idx(train_data.n_smpl_d_l+1:end),:);
        train_data.gnd_d_u = train_data.gnd_d(:,shuffle_idx(train_data.n_smpl_d_l+1:end));
        train_data.n_smpl_d_u = size(train_data.fea_d_u,2);
        train_data.n_smpl_l = labeled_num*n_agent;
        
        train_data.fea_l = train_data.fea(1:train_data.n_smpl_l,:);
        train_data.gnd_l = train_data.gnd(1:train_data.n_smpl_l);
        train_data.fea_u = train_data.fea(train_data.n_smpl_l+1:end,:);
        train_data.gnd_u = train_data.gnd(train_data.n_smpl_l+1:end);
        train_data.n_smpl_u = size(train_data.fea_u,1);
        a = rand(l, train_data.n_fea);
        b = rand(l, train_data.n_fea);
        c = train_data.fea(1:l,:)';
        H_train_g = zeros(train_data.n_smpl,l);
        for ii = 1:l
%             H_train_g(:,ii) = prod(Gauss((train_data.fea-b(ii,:))./a(ii,:))./sqrt(a(ii,:)),2);    % wavelet basis function
            H_train_g(:,i) = RBFN(train_data.fea,a(:,i),b(:,i),c(:,i));    % SLFNN
        end
        H_train_g_lab = H_train_g(1:train_data.n_smpl_l,:);
        W_Lap = squareform(pdist(train_data.fea));
        W_Lap = exp(-W_Lap.^2/2);
        D_Lap = diag(sum(W_Lap,2));
        Lap = D_Lap - W_Lap;
        D_Lap(D_Lap>0) = 1./sqrt(D_Lap(D_Lap>0));
        Lap = D_Lap^(-1/2)*Lap*D_Lap^(-1/2);
        Lap = D_Lap*Lap*D_Lap;
        W_C_lab = (H_train_g_lab'*H_train_g_lab + lambda*eye(l) + eta*H_train_g'*Lap*H_train_g )^(-1)*H_train_g_lab'*train_data.gnd_l;
        y_hat_train_g = H_train_g*W_C_lab;
        H_test_g = zeros(test_data.n_smpl,l);
        for ii = 1:l
%             H_test_g(:,ii) = prod(Gauss((test_data.fea-b(ii,:))./a(ii,:))./sqrt(a(ii,:)),2);  % wavelet basis function
            H_test_g(:,i) = RBFN(test_data.fea,a(:,i),b(:,i),c(:,i));    % SLFNN
        end
        y_hat_test_g = H_test_g*W_C_lab;
        if task=='C'
            rslt_train_g = calculate_acc(train_data.gnd, y_hat_train_g);
            fprintf(fid, "==>Train acc of c_lapwnn: %f\n", rslt_train_g);
            fprintf("==>Train acc of c_lapwnn: %f\n", rslt_train_g);
        else
            rslt_train_g = calculate_nrmse(train_data.gnd, y_hat_train_g);
            fprintf(fid, "==>Train NRMSE of c_lapwnn: %f\n", rslt_train_g);
            fprintf("==>Train NRMSE of c_lapwnn: %f\n", rslt_train_g);
        end
        if task=='C'
            rslt_test_g = calculate_acc(test_data.gnd, y_hat_test_g);
            fprintf(fid, "==>Test acc  of c_lapwnn: %f\n", rslt_test_g);
            fprintf("==>Test acc  of c_lapwnn: %f\n", rslt_test_g);
        else
            rslt_test_g = calculate_nrmse(test_data.gnd, y_hat_test_g);
            fprintf(fid, "==>Test NRMSE  of c_lapwnn: %f\n", rslt_test_g);
            fprintf("==>Test NRMSE of c_lapwnn: %f\n", rslt_test_g);
        end
        result_test_g_arr(i) = rslt_test_g;
        result_train_g_arr(i) = rslt_train_g;
        
%         A = [0 1 1 1 1;
%             1 0 1 1 1;
%             1 1 0 1 1;
%             1 1 1 1 0];
        A = ones(n_agent);
        A = A-diag(ones(n_agent, 1));
        d = max(sum(A));
        L = diag(sum(A)) - A;
        V = size(A,1);

        %%  ��ʼ��
        W0 = zeros(l,V);
        Hinv = cell(V,1);
        Hi = cell(V,1);
        Yi = cell(V,1);

        for ii = 1:V
            xi = squeeze(train_data.fea_d(ii,:,:));
            yi = train_data.gnd_d(ii,:)';
            H_train_d_g = zeros(train_data.n_smpl_d,l);
            c = xi(1:l,:)';
            for kk = 1:l
%                 H_train_d_g(:,kk) = prod(Gauss((xi-b(ii,:))./a(ii,:))./sqrt(a(ii,:)),2); % wavelet basis function
                H_train_d_g(:,i) = RBFN(xi,a(:,i),b(:,i),c(:,i));    % SLFNN
            end
            Hi{ii} = H_train_d_g;
            Yi{ii} = yi;

            Hi_lab = Hi{ii}(1:train_data.n_smpl_d_l,:);
            W_l = squareform(pdist(xi));
            W_l = exp(-W_l.^2/2);
            D_l = diag(sum(W_l,2));
            Lap = D_l - W_l;
            Hinv{ii} = (Hi_lab'*Hi_lab + lambda*eye(l) + eta*Hi{ii}'*Lap*Hi{ii} )^(-1);
            W0(:,ii) = Hinv{ii}*(Hi_lab'*yi(1:train_data.n_smpl_d_l));
        end

        %% D-Lap

        % ��ʼ��
        W0_dac = W0;
        W_dac = W0;

        W_admm = W0;
        t0 = W0 - W0;
        z0 = t0(:,1);
        t = t0;
        z = z0;

        W0_zgs = W0;
        W_zgs = W0;

        W0_dlms = W0 - W0;
        W_dlms = W0_dlms;

        %% ����
        for k = 1:K

            for l = 1:V
                W_zgs(:,i) = W0_zgs(:,i) - gamma*Hinv{l}*W0_zgs*L(:,l);
            end
        end
        w_d_g = W_zgs;
        for ii = 1:n_agent
            y_hat_train_d{ii} = Hi{ii} * w_d_g(:,ii);
            
            if task=='C'
                rslt_train_d(ii) = calculate_acc(train_data.gnd_d(ii,:)', y_hat_train_d{ii});
                fprintf(fid, "==>%d-th agent Train acc  of d_lapwnn: %f\n", ii, rslt_train_d(ii));
                fprintf("==>%d-th agent Train acc  of d_lapwnn: %f\n", ii, rslt_train_d(ii));
                
            else
                rslt_train_d(ii) = calculate_nrmse(train_data.gnd_d(ii,:)', y_hat_train_d{ii});
                fprintf(fid, "==>%d-th agent  of d_lapwnn: %f\n", ii, rslt_train_d(ii));
                fprintf("==>%d-th agent Train  of d_lapwnn: %f\n", ii, rslt_train_d(ii));
                
            end
            
        end
        y_hat_test_d = H_test_g*w_d_g(:,1);
        if task=='C'
            rslt_test_d = calculate_acc(test_data.gnd, y_hat_test_d);
            fprintf(fid, "==>Test acc  of d_lapwnn: %f\n", rslt_test_d);
            fprintf("==>Test acc  of d_lapwnn: %f\n", rslt_test_d);
        else
            rslt_test_d = calculate_nrmse(test_data.gnd, y_hat_test_d);
            fprintf(fid, "==>Test NRMSE  of d_lapwnn: %f\n", rslt_test_d);
            fprintf("==>Test NRMSE of d_lapwnn: %f\n", rslt_test_d);
        end
        result_test_d_arr(i) = mean(rslt_train_d);
        result_train_d_arr(i) = rslt_test_d;
        
        fprintf(fid, "==> %d-th fold finished!\n", i);
        fprintf("==> %d-th fold finished!\n", i);
    end
    result_test_mean_g = mean(result_test_g_arr); 
    result_test_std_g = std(result_test_g_arr);
    result_train_mean_g = mean(result_train_g_arr); 
    result_train_std_g = std(result_train_g_arr);
    
    result_test_mean_d = mean(result_test_d_arr); 
    result_test_std_d = std(result_test_d_arr);
    result_train_mean_d = mean(result_train_d_arr); 
    result_train_std_d = std(result_train_d_arr);
    fprintf(fid, "Train result of c_lapwnn: %.4f/%.4f\n", result_train_mean_g, result_train_std_g);
    fprintf(fid, "Test result of c_lapwnn: %.4f/%.4f\n", result_test_mean_g, result_test_std_g);
    
    fprintf(fid, "Train result of d_lapwnn: %.4f/%.4f\n", result_train_mean_d, result_train_std_d);
    fprintf(fid, "Test result of d_lapwnn: %.4f/%.4f\n", result_test_mean_d, result_test_std_d);
    
    fprintf("Train result of c_lapwnn: %.4f/%.4f\n", result_train_mean_g, result_train_std_g);
    fprintf("Test result of c_lapwnn: %.4f/%.4f\n", result_test_mean_g, result_test_std_g);
    
    fprintf("Train result of d_lapwnn: %.4f/%.4f\n", result_train_mean_d, result_train_std_d);
    fprintf("Test result of d_lapwnn: %.4f/%.4f\n", result_test_mean_d, result_test_std_d);
    if exist('./results','dir')==0
       mkdir('./results');
    end
    save_dir = sprintf('./results/%s_d_lap_wnn_r%d_l%.1f_lambda%.4f_eta%.4f_gamma%.4f.mat',...
        data_name, n_rules, labeled_num, lambda, eta, gamma);
    save(save_dir, 'result_test_mean_g', 'result_test_std_g', 'result_train_mean_g', 'result_train_std_g',...
        'result_test_mean_d', 'result_test_std_d', 'result_train_mean_d', 'result_train_std_d');
        
end