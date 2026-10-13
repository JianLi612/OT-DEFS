function dist = wasserstein_distance(u, v, p)
    % Compute the p-Wasserstein distance for one-dimensional data (based on the cumulative distribution function)
    % Inputs:
    %   u, v - Vectors for two features (sample data)
    %   p - Order of the Wasserstein distance (default p=1, the 1-Wasserstein distance)
    % Outputs:
    %   dist - p-Wasserstein distance
    %
    % Notes:
    %   This implementation assumes u and v are data points sampled from two distributions and uses the empirical cumulative distribution function (ECDF).
    %   For one-dimensional data, the 1-Wasserstein distance equals the L1 norm of the difference between the two cumulative distribution functions.

    % Default parameters
    if nargin < 3
        p = 1; % Use the 1-Wasserstein distance by default
    end

    % Input validation
    if length(u) ~= length(v)
        error('Input vectors u and v must have the same length');
    end
    if ~isvector(u) || ~isvector(v)
        error('Inputs u and v must be one-dimensional vectors');
    end

    % Convert inputs to column vectors
    u = u(:);
    v = v(:);
    n = length(u);

    % Sort the data to construct empirical distributions
    u_sorted = sort(u);
    v_sorted = sort(v);

    % Compute the cumulative distribution function (CDF)
    % With uniform weights, each sample has a weight of 1/n
    cdf_u = (1:n)' / n; % Empirical CDF of u (assuming a uniform distribution)
    cdf_v = (1:n)' / n; % Empirical CDF of v (assuming a uniform distribution)

    % Compute the distance between sample points
    if p == 1
        % 1-Wasserstein distance: L1 norm of the sorted samples
        dist = sum(abs(u_sorted - v_sorted)) / n;
    elseif p == 2
        % 2-Wasserstein distance: Square root of the L2 norm of the sorted samples
        dist = sqrt(sum((u_sorted - v_sorted).^2) / n);
    else
        % General p-Wasserstein distance
        dist = (sum(abs(u_sorted - v_sorted).^p) / n)^(1/p);
    end

    % Weights can be introduced for nonuniform distributions (uniform weights are assumed here)
    % For more complex distributions, pass weight parameters and adjust the CDF calculation
end