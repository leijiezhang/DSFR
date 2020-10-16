function acc = calculate_acc(y,y_hat)
    n_class = size(unique(y),1);
    y_hat = round(y_hat);
    y_hat(find(y_hat > n_class-1)) = n_class-1;
    y_hat(find(y_hat < 0)) = 0;
    acc = sum(y_hat == y)/size(y,1);
end