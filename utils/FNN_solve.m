function [Yhat, boptimal] = FNN_solve(H, Y, lambda)

            [~, p] = size(H);
            
            boptimal = (H'*H + lambda*eye(p)) \ (H'*Y);

            %% Calculate Yhat
            Yhat = H * boptimal;
end