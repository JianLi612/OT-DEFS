% pre_wd.m
% Generate the Wasserstein divergence matrix and save it in the dataset's df field

%% Initialization
clear; % Clear workspace variables
data_folder = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'synthetic_small'); % Data folder relative to this script

% Get all .mat files in the folder and its subfolders
mat_files = dir(fullfile(data_folder, '**', '*.mat')); % Recursively search for all .mat files
datasets = {mat_files.folder}; % Extract the folder path for each file
file_names = {mat_files.name}; % Extract file names (including the .mat extension)

% Construct full paths and remove the .mat extension
full_paths = cellfun(@(folder, name) fullfile(folder, name), datasets, file_names, 'UniformOutput', false);

% full_paths = {fullfile(data_folder, 'CARML.mat'), ...
%               fullfile(data_folder, 'MALDIML.mat'), ...
%               fullfile(data_folder, 'MLLML.mat')};
%
failed_datasets = cell(0, 1); % Initially an empty column vector

%% Main loop: Process each dataset
for dataset_idx = 1:length(full_paths)
    % Load the dataset
    full_file_path = full_paths{dataset_idx}; % Full path
    fprintf('1. Processing dataset: %s\n', full_file_path);

    % Extract X and Y
    try
        [X, Y] = extractXY(full_file_path);
    catch ME
        fprintf('Failed to extract X and Y: %s\nSkipping this file...\n', ME.message);
        % Record the failed dataset (append to the column vector)
        failed_datasets{end+1, 1} = full_file_path; % Add as a column entry
        continue; % Skip the file with an error
    end
    % X,N*D  Y,N*1
    % Default row = 1 -> Normalize by row (sample)
    X = NormalizeFea(X); % normalization!!
    % X = NormalizeFea(X, 0);   % row = 0 -> Normalize by column (feature)

    [n,d] = size(X);
    X = X';  % d*n

    mat_data = load(full_file_path);
    norm_type = 1; % Select Df:1 or Ds:2 (on 8.21, this was found to be incorrect; the data are actually normalized by sample)
    % 3. Check and load or compute the divergence matrix D
    switch norm_type
        case 2
            D_var_name = 'Ds';  % Divergence matrix normalized by sample
        case 1
            D_var_name = 'Df';  % Divergence matrix normalized by feature (on 8.21, this was found to be incorrect; the data are actually normalized by sample)
    end

    % Check whether the corresponding divergence matrix already exists in the .mat file

    if isfield(mat_data, D_var_name)
        D = mat_data.(D_var_name);
        fprintf('2. %s already contains divergence matrix %s\n', full_file_path, D_var_name);
        fprintf('\n');
    else
        % Compute the divergence matrix D (in matrix form)
        D = zeros(d, d);
        % Get the Wasserstein distances for all feature pairs
        [i, j] = meshgrid(1:d, 1:d);
        mask = (i ~= j);              % Create a mask to compute only elements where i ~= j
        % Extract all feature pairs that need to be computed
        idx_i = i(mask);
        idx_j = j(mask);
        % Compute the Wasserstein distances for all off-diagonal elements at once
        distances = arrayfun(@(k) wasserstein_distance(X(idx_i(k), :), X(idx_j(k), :)), 1:length(idx_i));
        % Fill the matrix with the computed results
        D(sub2ind([d, d], idx_i, idx_j)) = distances;

        % Assign D to the corresponding variable name (Ds or Df)
        eval([D_var_name ' = D;']);
        % Save the divergence matrix to the .mat file
        save(full_file_path, D_var_name, '-append');  % Append to the existing file
        fprintf('2. Computed and saved divergence matrix %s to %s\n', D_var_name, full_file_path);
        fprintf('\n');
    end
end

%% Save the collection of failed datasets (as a column vector)
failed_file = fullfile(data_folder, 'failed_datasets.mat'); % Path to the failure log file
if ~isempty(failed_datasets)
    % Ensure failed_datasets is a column vector
    failed_datasets = failed_datasets(:); % Force conversion to a column vector
    save(failed_file, 'failed_datasets', '-v7.3'); % Save the list of failed datasets
    fprintf('Saved %d failed dataset records (as a column) to %s\n', length(failed_datasets), failed_file);
else
    fprintf('No dataset extraction failed; no log needs to be saved.\n');
end

%% Complete
disp('All datasets have been processed!');