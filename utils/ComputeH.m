function H = ComputeH(X,rule)

    [N, d] = size(X);
    nRules = rule.n_rules;
    a = zeros(nRules, N, d);
    
    %% make sure the sigmal value of Gauss membership function is not 0 
    for ii = 1:nRules
        for jj = 1:d
            if rule.width(ii,jj) == 0 || isnan(rule.width(ii,jj))
                rule.width(ii,jj) = 1e-3;
            end
            if isnan(rule.center(ii,jj))
                rule.center(ii,jj) = 1;
            end
        end
    end
    
    for ii = 1:nRules
        for jj = 1:d
            mf = fismf(@gaussmf,[rule.width(ii,jj), rule.center(ii,jj)]);
            a(ii, :, jj) = evalmf(mf,X(:, jj));
        end
    end
%             
%             [N, d] = size(X);
%             nRules = size(fis.rule, 2);
%             
%             a = zeros(nRules, N, d);
%             
%             for ii = 1:nRules
%                 for jj = 1:d
%                     mf = fis.input(jj).mf(fis.rule(ii).antecedent(jj));
%                     a(ii, :, jj) = evalmf(mf,X(:, jj));
%                 end
%             end
            
            w = prod(a, 3); %Computes the unnormalized firing strengths
            w_hat = w./(repmat(sum(w, 1), nRules, 1)); %Normalizes the firing strengths
            w_hat(find(isnan(w_hat))) = 1/nRules; 
            H = [];
            for c = 1:size(w_hat, 1)
                H = [H repmat(w_hat(c, :)', 1, d+1).*[ones(N, 1) X]]; %Computes the hidden matrix
            end
end