function [boptimal, trainInfo] = mixup_FNN_ADMM(train_data,n_agents, w, H_l, mu, rho,...
    rule, n_mixup, gamma, alpha)

%% define ADMM parameters 
    admm_max_steps = 300;
    admm_reltol = 0.001;
    admm_abstol = 0.001;
%     gamma=0.01;
            
%     [~, d] = size(squeeze(fea_d(1,:,:)));

    N_params = size(w, 1);

    error = zeros(admm_max_steps, n_agents);

     % Global term
    z = zeros(N_params, 1);

    % Lagrange multipliers
    t = zeros(N_params, n_agents);

    % Parameters
    steps = admm_max_steps;
    
    % Precompute the matrices
    Hinv = cell(n_agents, 1);
    HY = cell(n_agents, 1);

    for ii = 1:n_agents
        n_smpl_d_l_agent = size(train_data.fea_d_l{ii},1);
        n_smpl_d_u_agent = train_data.n_smpl_d - n_smpl_d_l_agent;
        % setting mixup
        mix_index1 = mod(randperm(n_mixup),n_smpl_d_u_agent)+1;
        mix_data1 = squeeze(train_data.fea_d_u{ii}(mix_index1, :));
        mix_index2 = mod(randperm(n_mixup),n_smpl_d_u_agent)+1;
        mix_data2 = squeeze(train_data.fea_d_u{ii}(mix_index2, :));


        lambda_1 = betarnd(alpha, alpha,n_mixup,1);

        H_train_mix1 = ComputeH(lambda_1.*mix_data1+(1-lambda_1).*mix_data2,rule);
        H_train_mix2 = ComputeH(mix_data1,rule);
        H_train_mix3 = ComputeH(mix_data2,rule);
        B = H_train_mix1 - lambda_1.*H_train_mix2 - (1-lambda_1).*H_train_mix3;
        Hinv{ii} = inv(gamma*B'*B + eye(size(w, 1))*rho + H_l{ii}' * H_l{ii});
        HY{ii} = H_l{ii}'*train_data.gnd_d_l{ii}';
    end

    beta = w;

    for ii = 1:steps

        for jj = 1:n_agents

            % Compute current weights
            beta(:, jj) = Hinv{jj}*(HY{jj} + rho*z - t(:, jj));

        end

        % Store the old z and update it
        zold = z;
        z = (rho*sum(beta,2) + sum(t,2))/(mu + rho*n_agents);

%               z = (rho*beta_avg + t_avg)/(lambda/n_agents + rho);

        % Compute the update for the Lagrangian multipliers
        for jj = 1:n_agents
            t(:, jj) = t(:, jj) + rho*(beta(:, jj) - z);
        end

        % Check stopping criterion
        s = - rho*(z - zold);
        t_norm = zeros(n_agents, 1);

        primal_criterion = zeros(n_agents, 1);

        for kk = 1:n_agents
            r = beta(:, kk) - z;
            error(ii,kk) = norm(r);
            if norm(r) < sqrt(n_agents)*admm_abstol + admm_reltol*max(norm(beta(:,kk), 2), norm(z, 2)) 
                primal_criterion(kk) = 1;
            end

            t_norm(kk) = norm(t(:, kk), 2);
        end

        if norm(s) < sqrt(n_agents)*admm_abstol + admm_reltol*max(t_norm) && max(primal_criterion) == 1
            break;
        end
    end

    boptimal = beta(:, 1);

    trainInfo.error = error;
    trainInfo.consensus_steps = ii;
end