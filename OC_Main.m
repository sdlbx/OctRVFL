clear;
clc;
warning off;

%% 1. Path Setup
addpath(genpath('./utils'));

%% 2. Load Dataset
% Place your custom .mat file inside the './Datasets' folder.
% Data format: [N samples x (D features + 1 label)], where the last column is the class label.
dataPath = './Datasets/your_dataset.mat'; 

if ~exist(dataPath, 'file')
    error('File "%s" not found. Please place your .mat dataset in the ./Datasets/ folder.', dataPath);
end

raw = importdata(dataPath);
if isstruct(raw)
    fields = fieldnames(raw);
    raw = raw.(fields{1});
end

X_raw = raw(:, 1:end-1);
Y_raw = raw(:, end);

% Map labels to 1, 2, ..., K
[~, ~, Y] = unique(Y_raw);
classes = unique(Y);
nclass = length(classes);

%% 3. Train/Test Split (80/20)
n = size(X_raw, 1);
rng(0); 
idx = randperm(n);
nTrain = round(0.8 * n);

trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

trainX_raw = X_raw(trainIdx, :);
trainY     = Y(trainIdx);
testX_raw  = X_raw(testIdx, :);
testY      = Y(testIdx);

%% 4. Z-Score Standardization
mu = mean(trainX_raw, 1);
sigma = std(trainX_raw, 0, 1);
sigma(sigma == 0) = 1; % Prevent division by zero for invariant features

trainX = (trainX_raw - mu) ./ sigma;
testX  = (testX_raw  - mu) ./ sigma;

%% 5. Model Configuration
option = struct();
option.nclass = nclass;
if size(trainX, 1) < 200
    option.m = 2^4;
else
    option.m = 2^7; 
end
option.C = 1;
option.N = 183;
option.activation = 6;   
option.a = -0.5;
option.b = 0.5;
option.alpha = 0.001;

%% 6. Model Training and Evaluation
rng(0); 

[TrainAcc, TestMetrics, train_time, test_time] = ...
    OC_RVFL_Model(trainX, trainY, testX, testY, option);

%% 7. Output Results
fprintf('================ Demo Finished ================\n');
fprintf('Test Accuracy  : %.4f\n', TestMetrics(1));
fprintf('Train Time     : %.4f s\n', train_time);
fprintf('Test Time      : %.4f s\n', test_time);
fprintf('===============================================\n');