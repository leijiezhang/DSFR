function [mu_optimal, trainInfo] = Kmeans_ADMM(fea_d, nRules, N_nodes)

        %% define ADMM parameters 
            consensus_max_steps = 300;
            consensus_thres = 0.001;
            admm_max_steps = 300;
            admm_rho = 1;
            admm_reltol = 0.001;
            admm_abstol = 0.001;
            
            for ii = 1:N_nodes
                [idx(ii), d] = size(squeeze(fea_d(ii,:,:)));
            end
            
            error = zeros(admm_max_steps, 1);
            consensus_steps = zeros(admm_max_steps, 1);
            
             % Global term
            mu = zeros(nRules, d); 
            
            % Lagrange multipliers 
            t = zeros(nRules, d, N_nodes); 
            
            % Parameters
            rho = admm_rho;
            steps = admm_max_steps;
            
            mu_node = zeros(nRules, d, N_nodes);
            
            %% initialize the centroids mu
            fea_ds = squeeze(fea_d(1,:,:));
            gm = fitgmdist(fea_ds,nRules);
            mu = gm.mu;
            for ii = 1:steps  
                
                for jj = 1:N_nodes
                    % assign clusters for each node based on the global centoids
                    [~,label{jj}] = min(dot(mu,mu,2)/2-mu*squeeze(fea_d(jj,:,:))',[],1);
                    mu_node(:,:,jj) = normalize(sparse(label{jj},1:idx(jj),1),2)*squeeze(fea_d(jj,:,:));
                end
                
                % Store the old z and update it
                mu_old = mu;
                
                % for each cluster
                for kk = 1:nRules
                    mu(kk,:) = (rho*sum(mu_node(kk,:,:),3) + sum(t(kk,:,:),3))/(rho*N_nodes);
                end
                
                % Compute the update for the Lagrangian multipliers
                for kk = 1:nRules
                    for jj = 1:N_nodes
                        t(kk, :, jj) = t(kk, :, jj) + rho*(mu_node(kk,:,jj) - mu(kk,:));
                    end
                end
                
                % Check stopping criterion
                s = - rho*(mu - mu_old);
                t_norm = zeros(N_nodes, 1);
                
                error(ii,1) = norm(s,'fro');
                
                if error(ii,1) < sqrt(N_nodes)*admm_abstol
                    break;
                end
                %ii
            end
            
            mu_optimal = mu;

            trainInfo.error = error;
            trainInfo.consensus_steps = ii;
end

function Y = normalize(X, dim)
% Normalize the vectors to be summing to one
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end
Y = X./sum(X,dim);
end
