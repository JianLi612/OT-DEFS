function [g, grad_f1, subgrad_f2, grad_f3] = subgradient_w_rob(w, P, X, Y, D, lambda_reg, alpha, return_components, epsilon)
    % Compute the subgradient
    % Inputs:
    %   w - Current weights
    %   P - Projection matrix
    %   X - Data matrix
    %   Y - Label matrix
    %   D - Divergence matrix
    %   lambda_reg - Regularization parameter
    %   alpha - Divergence term weight
    %   return_components - Whether to return gradient components
    %   epsilon - Divergence term smoothing parameter

    % Outputs:
    %   g - Total subgradient
    %   grad_f1, subgrad_f2, grad_f3 - Gradient/subgradient of each component

    if nargin < 8, return_components = false; end
    if nargin < 9, epsilon = 1e-6; end

    d = length(w);
    diag_w = diag(w);
    projection = P * diag_w * X;  % Projection
    residual = projection - Y;  % Residual

    grad_f1 = 2 * sum((P' * residual) .* X, 2);   % Gradient of the empirical loss
    subgrad_f2 = lambda_reg * sign(w);            % Subgradient of the regularization term

    % Subgradient of the divergence term
    abs_w = abs(w);
    div_w = 0.5 * (abs_w' * D * abs_w);              % d(w) = 0.5 |w|^T D |w|
    grad_d = (D * abs_w) .* sign(w);                 % \partial d(w) = D |w| .* sign(w)
    grad_f3 = -alpha * grad_d / (div_w + epsilon)^2; % \partial f_3

    g = grad_f1 + subgrad_f2 + grad_f3;  % Total subgradient

    if return_components
        varargout = {g, grad_f1, subgrad_f2, grad_f3};  % Return all components
    else
        varargout = {g};  % Return only the total subgradient
    end
end