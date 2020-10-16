function [boptimal, trainInfo] = graph_FNN_ADMM(train_data,n_agents, w, H_l, mu, rho,...
    H_g, eta)

%% define ADMM parameters 
    admm_max_steps = 300;
    admm_reltol = 0.001;
    admm_abstol = 0.001;
%     mu=0.001;
%     eta = 0.0001;
%             
%     [~, d] = size(squeeze(fea_l(1,:,:)));

    N_params = size(w, 1);

    error = zeros(admm_max_steps, n_agents);

     % Global term
    z = zeros(N_params, 1);

    % Lagrange multipliers
    t = zeros(N_params, n_agents);

    % Parameters
%     rho = admm_rho;
    steps = admm_max_steps;

    % Precompute the matrices
    Hinv = cell(n_agents, 1);
    HY = cell(n_agents, 1);

    for ii = 1:n_agents
        dist_tmp = EuDist2(train_data.fea_d{ii});
        t_tmp = mean(mean(dist_tmp));
        W_tmp = exp(-dist_tmp/(2*t_tmp^2));
%         W_tmp = constructW(squeeze(fea_g(ii,:,:)));
%         W_tmp = full(W_tmp);
        D_tmp = diag(sum(W_tmp,2));
%         D_tmp = full(diag(sum(W_tmp,2)));
        L_tmp = D_tmp - W_tmp;
%         D_half = sparse(diag(1./diag(D_tmp.^2)));
        D_half = D_tmp^(-1/2);
        L_hat = D_half*L_tmp*D_half;
        Hinv{ii} = inv(eye(length(w))*rho  + H_l{ii}' * H_l{ii} + eta*H_g{ii}'*L_hat*H_g{ii});
%         Hinv{ii} = inv(eye(length(w))*(rho)  + H_l{ii}' * H_l{ii});
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