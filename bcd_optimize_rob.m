function [P, w, loss_history] = bcd_optimize_rob(X, Y, D, lambda_reg, alpha, initial_step, max_iter, tol, reg, seed, visualize_iter, epsilon)
    % Set default parameters
    if nargin < 6, initial_step = 0.01; end
    if nargin < 7, max_iter = 1000; end
    if nargin < 8, tol = 1e-6; end
    if nargin < 9, reg = 1e-6; end
    if nargin < 10, seed = []; end
    if nargin < 11, visualize_iter = []; end
    if nargin < 12, epsilon = 1e-6; end
    
    % Get dimensions
    [d, N] = size(X);
    [C, N_y] = size(Y);
    if N ~= N_y
        error('Sample count mismatch: X has %d samples and Y has %d samples', N, N_y);
    end
    
    % Initialize P and w
    if ~isempty(seed), rng(seed); end
    P = 0.5* randn(C, d);
    w = 0.5 * randn(d, 1);
    loss_history = [];
    
    % Initialize boundary parameter records, storing only lambda and alpha
    boundary_params = struct('lambda', {}, 'alpha', {});
    
    % Outer iterations
    for iter = 1:max_iter
        % Step 1: fix w and optimize P
        diag_w = diag(w);
        X_weighted = diag_w * X;
        P_intermediate = X_weighted * X_weighted' + reg * eye(d);
        P = (pinv(P_intermediate) * X_weighted * Y')';
    
        % Step 2: fix P and optimize w
        obj_w = @(w) compute_obj_w_rob(w, P, X, Y, D, lambda_reg, alpha, epsilon); % Wrap compute_obj_w as a function that depends only on w
        max_sub_iter = 1000;
        subgrad_values = zeros(max_sub_iter, d);  % Preallocate a matrix to store subgradients
        subloss_values = zeros(max_sub_iter + 1, 1);  % Preallocate the loss array
    
        % Initialize subgrad_components as a scalar structure with cell array fields
        subgrad_components = struct();
        subgrad_components.grad_f1 = cell(1, max_sub_iter);
        subgrad_components.subgrad_f2 = cell(1, max_sub_iter);
        subgrad_components.grad_f3 = cell(1, max_sub_iter);
    
        % Compute the initial loss
        current_loss = obj_w(w);
        subloss_values(1) = current_loss;
    
        % Inner iterations (subgradient method)
        for sub_iter = 1:max_sub_iter
            [g, grad_f1, subgrad_f2, grad_f3] = subgradient_w_rob(w, P, X, Y, D, lambda_reg, alpha, true, epsilon);
    
            % Store subgradient components
            subgrad_components.grad_f1{sub_iter} = grad_f1;  % Assign directly by index
            subgrad_components.subgrad_f2{sub_iter} = subgrad_f2;
            subgrad_components.grad_f3{sub_iter} = grad_f3;
    
            subgrad_values(sub_iter, :) = g';  % Store the total subgradient
            subgrad_norm = norm(g);
    
            % Update the step size (Armijo line search)
            c = 0.5; % Parameter controlling the amount of decrease
            alpha_k = initial_step;
            while obj_w(w - alpha_k * g) > obj_w(w) - c * alpha_k * norm(g)^2
                alpha_k = alpha_k * 0.7;
            end
            w_new = w - alpha_k * g; % Candidate update
    
            % Compute the new loss
            new_loss = obj_w(w_new);
            subloss_values(sub_iter + 1) = new_loss;
    
            % Check for early stopping in the inner iterations
            if sub_iter > 1 && abs(new_loss - current_loss) < tol   % Use 1/10 of tol as the inner iteration threshold
                w = w_new;
                fprintf('Seed %d, Iter %d/%d, Sub-Iter %d/%d, Loss: %.6f (Early Stopping), alpha_k: %.6f , Subgradient Norm: %.6f\n', ...
                    seed, iter, max_iter, sub_iter, max_sub_iter, new_loss, alpha_k, subgrad_norm);
                break;
            end
    
            % Update w and the current loss
            w = w_new;
            current_loss = new_loss;
            subloss_values(sub_iter + 1) = current_loss;
    
            fprintf('Seed %d, Iter %d/%d, Sub-Iter %d/%d, Loss: %.6f, alpha_k: %.6f , Subgradient Norm: %.6f\n', ...
                seed, iter, max_iter, sub_iter, max_sub_iter, current_loss, alpha_k, subgrad_norm);
        end
    
        % Record the history
        loss_history = [loss_history; current_loss];
        fprintf('Seed %d, Iteration %d/%d, Loss: %.6f\n', seed, iter, max_iter, current_loss);
    
        %         % Early stopping check (based on absolute error)
        %         if iter > 1 && abs(loss_history(end) - loss_history(end-1)) < tol
        %             fprintf('Converged after %d iterations\n', iter);
        %             break;
        %         end
    
        % Early stopping check (based on relative error)
        if iter > 1
            prev_loss = loss_history(end-1);
            curr_loss = loss_history(end);
            relative_change = abs(curr_loss - prev_loss) / max(abs(prev_loss), eps); % Avoid division by zero
    
            if relative_change < tol * 0.1
                fprintf('Converged after %d iterations, relative error: %.6e\n', iter, relative_change);
                break;
            end
    
        end
    end
    
    % Save boundary parameters to a file
    if ~isempty(boundary_params)
        fid = fopen('boundary_params.txt', 'a', 'n', 'UTF-8');
        for i = 1:length(boundary_params)
            fprintf(fid, 'Parameter set: lambda=%.6f, alpha=%.6f\n', ...
                boundary_params(i).lambda, boundary_params(i).alpha);
        end
        fclose(fid);
    end


end
