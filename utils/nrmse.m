function loss = nrmse(y,y_hat)
    loss = sqrt(sum((y_hat - y).^2)/(length(y)*var(y)));
end