function P  = ComputeP(X,rule)

    [n_sample, n_dim] = size(X);
    nRules = rule.n_rules;
    rule = rule.update_u(X, rule.center, 2);
    U = rule.u';
    P = []; % (n_sample,(n_dim+1)*nRules)
    for i = 1:nRules
        P = [P, repmat(U(:,i), 1, n_dim+1).*[ones(n_sample,1), X]];
    end
    
end