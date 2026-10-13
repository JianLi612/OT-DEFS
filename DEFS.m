function res_model = DEFS(~, train_data, train_target, Df, Ds, lambda_reg, alpha, eta, norm_type)
%% function [ranking, w] = defs_feature_selection(X_train, Y_train, normalization_type, mat_file)

X = train_data;

train_target = train_target'; % n*1

if isvector(train_target) || size(train_target, 2) == 1
    unique_labels = unique(train_target);
    Y = dummyvar(categorical(train_target, unique_labels));
end
Y = Y'; % c*n

[~, ~] = size(X);


switch norm_type
    case 1
        D = Df;  
    case 2
        D = Ds;  
end

%% Run BCD optimization
% Define all available optimization methods
optimization_methods = {'subgradient'};  


optimization_method = optimization_methods{1};
%     fprintf('Optimization method: %s\n', optimization_method);

% Define common parameters
max_iter = 1000;        % Maximum number of iterations
reg = 1e-6;             % Specific parameter: regularization parameter (regularization of the matrix pseudoinverse when optimizing P)
visualize_iter = 1;     % Enable visualization (1 for yes, 0 for no)

% Perform different BCD optimizations according to the optimization method
switch lower(optimization_method)
        % 1. Subgradient method
    case 'subgradient'

        tol = 0.000001;           % Convergence parameter
        seed = 7;
        epsilon = 1e-4;         % Smoothing parameter for the divergence term

        [~, w_opt, loss_history] = bcd_optimize_rob(X, Y, D, lambda_reg, alpha, eta, max_iter, tol, reg, seed, visualize_iter, epsilon);
        

end

fprintf('Optimization completed, optimization method: %s, lambda_reg = %.2f, alpha = %.2f\n', optimization_method, lambda_reg, alpha);

%% Compute feature rankings (sorted by absolute value)
[~, ranking] = sort(abs(w_opt), 'descend');  % Modification: use abs(w_opt) to sort by absolute value
w = w_opt;
res_model = struct('W', w, 'lambda_reg', lambda_reg, 'alpha', alpha, 'eta', eta, 'loss_history', loss_history);
end
