function [boptimal, trainInfo] = semi_FNN_ADMM(gnd_l, fea_g,N_nodes, beta0, H_g, H_l, lambda, mu, eta)

%% define ADMM parameters 
    consensus_max_steps = 300;
    consensus_thres = 0.001;
    admm_max_steps = 3000;
    admm_rho = 1;
    admm_reltol = 0.001;
    admm_abstol = 0.001;
%     mu=0.001;
%     eta = 0.0001;
%             
%     [~, d] = size(squeeze(fea_l(1,:,:)));

    N_params = size(beta0, 1);

    error = zeros(admm_max_steps, N_nodes);
    consensus_steps = zeros(admm_max_steps, 1);

     % Global term
    z = zeros(N_params, 1);

    % Lagrange multipliers
    t = zeros(N_params, N_nodes);

    % Parameters
    rho = admm_rho;
    steps = admm_max_steps;

    % Precompute the matrices
    Hinv = cell(N_nodes, 1);
    HY = cell(N_nodes, 1);

    for ii = 1:N_nodes
        dist_tmp = EuDist2(squeeze(fea_g(ii,:,:)));
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
        Hinv{ii} = inv(eye(size(beta0, 1))*(rho+mu)  + H_l{ii}' * H_l{ii} + eta*H_g{ii}'*L_hat*H_g{ii});
%         Hinv{ii} = inv(eye(length(beta0))*(rho+mu)  + H_l{ii}' * H_l{ii});
        HY{ii} = H_l{ii}'*gnd_l(ii,:)';
    end

    beta = beta0;

    for ii = 1:steps

        for jj = 1:N_nodes

            % Compute current weights
            beta(:, jj) = Hinv{jj}*(HY{jj} + rho*z - t(:, jj));

        end

        % Store the old z and update it
        zold = z;
        z = (rho*sum(beta,2) + sum(t,2))/(lambda + rho*N_nodes);

%               z = (rho*beta_avg + t_avg)/(lambda/N_nodes + rho);

        % Compute the update for the Lagrangian multipliers
        for jj = 1:N_nodes
            t(:, jj) = t(:, jj) + rho*(beta(:, jj) - z);
        end

        % Check stopping criterion
        s = - rho*(z - zold);
        t_norm = zeros(N_nodes, 1);

        primal_criterion = zeros(N_nodes, 1);

        for kk = 1:N_nodes
            r = beta(:, kk) - z;
            error(ii,kk) = norm(r);
            if norm(r) < sqrt(N_nodes)*admm_abstol + admm_reltol*max(norm(beta(:,kk), 2), norm(z, 2)) 
                primal_criterion(kk) = 1;
            end

            t_norm(kk) = norm(t(:, kk), 2);
        end

        if norm(s) < sqrt(N_nodes)*admm_abstol + admm_reltol*max(t_norm) && max(primal_criterion) == 1
            break;
        end
    end

    boptimal = beta(:, 1);

    trainInfo.error = error;
    trainInfo.consensus_steps = ii;
end