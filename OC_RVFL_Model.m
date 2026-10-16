function [TrainAcc,TestAcc,TrainTime,TestTime]  = OC_RVFL_Model(trainX,trainY,testX,testY,option)

%%%%%%%%%%%%%%%%% Training Starts %%%%%%%%%%%%%%%%%
N = option.N;
C = option.C;
s = 1;
activation = option.activation;
nclass      = option.nclass;
alpha=option.alpha;   
m=option.m;         
a=option.a;           
b=option.b;
%%%%%%%%%%%%%%%%%

% 1. Convert Labels to One-Hot
trainY_onehot = zeros(length(trainY), nclass);
for i = 1:nclass
    trainY_onehot(trainY == i, i) = 1;
end

[numTrainSamples, numFeat] = size(trainX);

tic; % Start training timer

% === 1. Map features for the FULL training dataset outside the loop ===
W = (rand(numFeat, N) * 2 * s - 1);
bias = s * rand(1, N);

X1_full = trainX * W + repmat(bias, numTrainSamples, 1);

switch activation
    case 1, X1_full = sigmoid(X1_full);
    case 2, X1_full = sin(X1_full);
    case 3, X1_full = tribas(X1_full);
    case 4, X1_full = radbas(X1_full);
    case 5, X1_full = tansig(X1_full);
    case 6, X1_full = relu(X1_full);
end

% Direct Link + Hidden Layer + Bias
X_full = [trainX, X1_full, ones(numTrainSamples, 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%
num_hidden_neurons = size(X_full, 2);

% Initialize parameters for Adam
beta = 0.01 * rand(num_hidden_neurons, nclass); 
r    = zeros(num_hidden_neurons, nclass);
v    = zeros(num_hidden_neurons, nclass);

beta1 = 0.9;       % exponential decay rates for the first moment estimate
beta2 = 0.999;     % exponential decay rates for the second moment estimate
epsilon = 10^-8;   % small constant used to avoid division by zero
max_iter = 500;    % maximum iteration number
tol = 10^-5;

betaPrevious = inf;

% Determine actual available batch size
real_m = min(m, numTrainSamples);

% === 2. Iterative Training ===
for t = 1:max_iter
    
    batch_idx = randi(numTrainSamples, real_m, 1);
    
    X_batch = X_full(batch_idx, :);
    Y_batch = trainY_onehot(batch_idx, :);
    
    U = X_batch * beta - Y_batch; 
    
    % Gradient Calculation
    exp_au = exp(a * U);            
    numer  = U .* exp_au .* (2 - a * U);          
    denom_inner = exp_au + b * (U.^2);            
    denom  = denom_inner .^ 2;
    dL_dU  = numer ./ (denom + epsilon);
    
    grad_data = C * (X_batch' * dL_dU);
    gradient  = beta + grad_data; 
    
    % Adam Update
    r = beta1 .* r + (1 - beta1) .* gradient;
    v = (beta2 .* v) + ((1 - beta2) .* (gradient.^2));
    
    r_hat = r ./ (1 - beta1^t);
    v_hat = v ./ (1 - beta2^t);
    
    beta = beta - ((alpha * r_hat) ./ (sqrt(v_hat) + epsilon));
    
    % Convergence Check
    if norm(beta - betaPrevious, 'fro') < tol
        break
    else
        betaPrevious = beta;
    end
end

% === 3. Training Set Evaluation ===
train_scores = X_full * beta;
[~, pred_train_labels] = max(train_scores, [], 2);
[~, true_train_labels] = max(trainY_onehot, [], 2);

EVAL_Train = Evaluate(true_train_labels, pred_train_labels, nclass);
TrainAcc = EVAL_Train(1); 
TrainTime = toc;

%%%%%%%%%%%%%%%%%%%% Testing Starts %%%%%%%%%%%%%%%%%%%%%
tic;
numTestSamples = size(testX, 1);

% Apply the same mapping to the test set
X1_test = testX * W + repmat(bias, numTestSamples, 1);

switch activation
    case 1, X1_test = sigmoid(X1_test);
    case 2, X1_test = sin(X1_test);
    case 3, X1_test = tribas(X1_test);
    case 4, X1_test = radbas(X1_test);
    case 5, X1_test = tansig(X1_test);
    case 6, X1_test = relu(X1_test);
end

X_test = [testX, X1_test, ones(numTestSamples, 1)];

rawScore = X_test * beta;
[~, pred_test_labels] = max(rawScore, [], 2);

TestAcc = Evaluate(testY, pred_test_labels, nclass);
TestTime = toc;
end
