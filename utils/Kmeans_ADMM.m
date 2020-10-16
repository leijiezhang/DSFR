function [mu_optimal, trainInfo] = Kmeans_ADMM(fea_d, n_rules, n_agents, rho)

        %% define ADMM parameters 
%             consensus_max_steps = 300;
%             consensus_thres = 0.001;
            admm_max_steps = 300;
%             admm_rho = 1;
%             admm_reltol = 0.001;
            admm_abstol = 1e-7;
            
            for ii = 1:n_agents
                [idx(ii), d] = size(squeeze(fea_d(ii,:,:)));
            end
            
            error = zeros(admm_max_steps, 1);
            consensus_steps = zeros(admm_max_steps, 1);
            
             % Global term
            mu = zeros(n_rules, d); 
            
            % Lagrange multipliers 
            t = zeros(n_rules, d, n_agents); 
            
            % Parameters
%             rho = admm_rho;
            steps = admm_max_steps;
            
            mu_agent = zeros(n_rules, d, n_agents);
            
            %% initialize the centroids mu
            [~, mu] = kmeans(squeeze(fea_d(1,:,:)),n_rules);
            for ii = 1:steps  
                
                for jj = 1:n_agents
                    % assign clusters for each node based on the global centoids
                    [~,label{jj}] = min(dot(mu,mu,2)/2-mu*squeeze(fea_d(jj,:,:))',[],1);
                    norm_label = normalize(sparse(label{jj},1:idx(jj),1),2);
                    if(size(norm_label, 1)<n_rules)
                        zero_comple = zeros(n_rules-size(norm_label, 1), size(norm_label, 2));
                        norm_label = [norm_label;zero_comple];
                    end
                    mu_agent(:,:,jj) = norm_label*squeeze(fea_d(jj,:,:));
                end
                
                % Store the old z and update it
                mu_old = mu;
                
                % for each cluster
                for kk = 1:n_rules
                    mu(kk,:) = (rho*sum(mu_agent(kk,:,:),3) + sum(t(kk,:,:),3))/(rho*n_agents);
                end
                
                % Compute the update for the Lagrangian multipliers
                for kk = 1:n_rules
                    for jj = 1:n_agents
                        t(kk, :, jj) = t(kk, :, jj) + rho*(mu_agent(kk,:,jj) - mu(kk,:));
                    end
                end
                
                % Check stopping criterion
                s = - rho*(mu - mu_old);
                t_norm = zeros(n_agents, 1);
                
                error(ii,1) = norm(s,'fro');
                
                if error(ii,1) < sqrt(n_agents)*admm_abstol
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
