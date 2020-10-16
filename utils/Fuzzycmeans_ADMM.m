function [mu_optimal, trainInfo] = Fuzzycmeans_ADMM(fea_d, n_rules, n_agents, rho, beta)

        %% define ADMM parameters 
            admm_max_steps = 300;
%             admm_rho = 1;
            admm_abstol = 1e-3;
            
            for agent_idx = 1:n_agents
                [idx(agent_idx), d] = size(fea_d{agent_idx});
            end
            
            error = zeros(admm_max_steps, 1);
%             
%              % Global term
%             mu = zeros(n_rules, d); 
            
            % Lagrange multipliers 
            t = zeros(n_rules, d, n_agents); 
            
%             % Parameters
%             rho = admm_rho;
            steps = admm_max_steps;
            
            %% initialize the centroids mu
            options = [beta 100 0.00001 0];
            [mu, ~] = fcm(fea_d{1},n_rules,options);
            mu_agent = zeros(n_rules, d, n_agents);
%             mu_agent = repmat(reshape(mu,n_rules, d, 1),1, 1, n_agents);
%             for jj = 1:n_agents
%                 % assign clusters for each node based on the global centoids
%                 [~,label{jj}] = min(dot(mu,mu,2)/2-mu*fea_d{ii}',[],1);
%                 norm_label = normalize(sparse(label{jj},1:idx(jj),1),2);
%                 if(size(norm_label, 1)<n_rules)
%                     zero_comple = zeros(n_rules-size(norm_label, 1), size(norm_label, 2));
%                     norm_label = [norm_label;zero_comple];
%                 end
% %                     norm_label=full(norm_label);
%                 mu_agent(:,:,jj) = norm_label*fea_d{ii};
%                 if(sum(sum(isnan(squeeze(mu_agent(:,:,jj))))))
%                     disp("lei");
%                 end
%             end
            for step_idx = 1:steps  
                
                for agent_idx = 1:n_agents
                    % update u for each agent based on the global centoids
                    X = fea_d{agent_idx};
                    n_smpl_d = size(fea_d{agent_idx},1);
%                     u_agent = zeros(n_rules, n_smpl_d);
                    var_x = zeros(n_rules, n_smpl_d);
                    u_ini = zeros(n_rules, n_smpl_d);

                    for rule_idx=1:n_rules
                        var_x(rule_idx,:)=sqrt(sum((X - mu(rule_idx,:)).^2, 2));
                    end
                    var_x(find(var_x<1e-6))=1e-6;
                    for rule_idx=1:n_rules
                        u_tmp = zeros(1, size(X,1));
                        for j=1:n_rules
                            var_x_i = var_x(rule_idx,:);
                            var_x_j = var_x(j,:);
                            u_tmp = u_tmp+(var_x_i./var_x_j).^(2/(beta-1));
                        end

                        u_ini(rule_idx,:) = 1./u_tmp;
                        if(sum(isnan(squeeze(u_ini(rule_idx,:)))))
                            disp("lei");
                        end
                    end
%                     [mu, ~] = fcm(X,n_rules,options);
                    % assign clusters for each node based on the global centoids
                    for rule_idx = 1:n_rules
                        mu_agent(rule_idx,:,agent_idx) = (sum(u_ini(rule_idx,:)'.*X,1)-t(rule_idx,:,agent_idx)+rho*mu(rule_idx,:))/(sum(u_ini(rule_idx,:))+rho);
%                         mu_agent(kk,:,jj) = (sum(u_ini(kk,:)'.*X,1))/(sum(u_ini(kk,:)));
                    end
                    
                end
                
                % Store the old z and update it
                mu_old = mu;
                
                % for each cluster
                for rule_idx = 1:n_rules
                    mu(rule_idx,:) = (rho*sum(mu_agent(rule_idx,:,:),3) + sum(t(rule_idx,:,:),3))/(rho*n_agents);
                end
                
                % Compute the update for the Lagrangian multipliers
                for rule_idx = 1:n_rules
                    for agent_idx = 1:n_agents
                        t(rule_idx, :, agent_idx) = t(rule_idx, :, agent_idx) + rho*(mu_agent(rule_idx,:,agent_idx) - mu(rule_idx,:));
                    end
                end
                
                % Check stopping criterion
                s = -(mu - mu_old);
                t_norm = zeros(n_agents, 1);
                
                error(step_idx,1) = norm(s,'fro');
%                 disp(mu(:,1:5));
%                 disp(std(mu(:,1:5),1));
                if step_idx>1
                    if abs(error(step_idx,1)-error(step_idx-1,1)) < admm_abstol
                        break;
                    end
                end
                %ii
            end
            
            mu_optimal = mu;

            trainInfo.error = error;
            trainInfo.consensus_steps = step_idx;
end

% function Y = normalize(X, dim)
% % Normalize the vectors to be summing to one
% %   By default dim = 1 (columns).
% % Written by Michael Chen (sth4nth@gmail.com).
%     if nargin == 1
%         % Determine which dimension sum will use
%         dim = find(size(X)~=1,1);
%         if isempty(dim), dim = 1; end
%     end
%     
%     Y = X./sum(X,dim);
%     Y(isnan(Y))=0;
% 
% end
