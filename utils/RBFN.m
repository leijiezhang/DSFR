function Y = RBFN(X,a,b,c)
sig = 1;
Y = exp(-sum((X-repmat(c',size(X,1),1)).^2,2)/(2*sig^2));                    % Gaussian
end