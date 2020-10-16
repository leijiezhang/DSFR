classdef Rules
    
    properties
        n_rules;
        center; % [ n_rules, n_fea]
        width; % [n_rules, n_fea]
        u; % [n_rules, n_smpl]

        consq; % [n_rules, n_fea+1]
    end
    
    methods
        function obj = Rules(n_rules, center, width, consq)
            
            if nargin >3
                obj.consq = consq;
            end
            if nargin >2
                obj.width = width;
            end
            if nargin >1
                obj.center = center;
            end
            if nargin >0
                obj.n_rules = n_rules;
            end
            
        end
        
        % update the center and width value using given center value with
        % kmeans
        function obj = update_kmeans(obj, X,center)
            [~,idx] = min(dot(center,center,2)/2-center*X',[],1);
            inVarl=size(center,2);

            for i=1:obj.n_rules
                cind=find(idx==i)'; 
                for j=1:inVarl
                   obj.center(i,j) = center(i,j);
                   obj.width(i,j) = std(X(cind,j));
                   if(isnan(obj.width(i,j)))
                       print("woooooops nan!");
                   end
%                    obj.consq(i,j) = 0;
                end
            end

        end
        
            
        % update the u and using the given data and its centers
        % fuzzy c means
        function obj = update_u(obj, X, center, beta)
            var_x = zeros(obj.n_rules, size(X,1));
            u_ini = zeros(obj.n_rules, size(X, 1));
            
            for i=1:obj.n_rules
                var_x(i,:)=sqrt(sum((X - center(i,:)).^2, 2));
            end
            for i=1:obj.n_rules
                u_tmp = zeros(1, size(X,1));
                for j=1:obj.n_rules
                    var_x_i = var_x(i,:);
                    var_x_j = var_x(j,:);
                    u_tmp = u_tmp+(var_x_i./var_x_j).^(2/(beta-1));
                end

                u_ini(i,:) = 1./u_tmp;
            end
            obj.u = u_ini;
        end
        
        % update the center and width value using given center value with
        % fuzzy c means
        function obj = update_fuzzyc(obj, X, center, beta)
            obj = update_u(obj, X, center, beta);
            for i=1:size(center,1)
                obj.center(i,:) = center(i,:);
                obj.width(i,:) = sqrt((sum(obj.u(i,:)'.*(X-center(i,:)).^2, 1))/sum(obj.u(i,:)));
                
            end

        end
        
        
        % initiate the Rule objective using kmeans
        function obj = init_kmeans(obj, X, nRules)
            [idx, center_init] = kmeans(X,nRules);      
            inVarl=size(center_init,2);

            for i=1:nRules
                cind=find(idx==i)'; 
                for j=1:inVarl
                   obj.center(i,j) = center_init(i,j);
                   obj.width(i,j) = std(X(cind,j));
                   obj.consq(i,j) = rand(1);
                end
                obj.consq(i,j+1) = rand(1);
            end

        end
        
        % initiate the Rule objective using fuzzyc means
        function obj = init_fuzzyc(obj, X, nRules,beta)
            options = [beta,100,1e-5,0];
            [center_init, u_init] = fcm(X,nRules,options);      

            for i=1:nRules
                obj.center(i,:) = center_init(i,:);
                obj.u(i,:) = u_init(i,:);
                obj.width(i,:) = sqrt((sum(u_init(i,:)'.*(X-center_init(i,:)).^2, 1))/sum(u_init(i,:)));
                obj.consq(i,:) = rand(1, size(center_init,2)+1);
                
            end

        end
        
    end
    
end