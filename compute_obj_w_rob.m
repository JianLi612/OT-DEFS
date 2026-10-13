function loss = compute_obj_w_rob(w, P, X, Y, D, lambda_reg, alpha, epsilon)
% Compute the objective function value
% Inputs:
%   w - Current weight vector
%   P - Projection matrix
%   X - Data matrix
%   Y - Label matrix
%   D - Divergence matrix
%   lambda_reg - Regularization parameter
%   alpha - Divergence term weight
%   epsilon - Divergence term smoothing parameter

% Outputs:
%   loss - Objective function value
    if nargin < 8, epsilon = 1e-6; end

    diag_w = diag(w);
    projection = P * diag_w * X;
    loss_emp = norm(projection - Y, 'fro')^2 + lambda_reg * sum(abs(w));

    % Divergence term: alpha / (0.5 |w|^T D |w| + epsilon)
    abs_w = abs(w);
    div_w = 0.5 * (abs_w' * D * abs_w); 
    loss = loss_emp + alpha / (div_w + epsilon);

end
