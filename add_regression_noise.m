function [trainX_noisy, trainY_noisy, corrupt_idx] = add_regression_noise(trainX, trainY, noise_ratio, noise_scale)

    if nargin < 3 || isempty(noise_ratio)
        noise_ratio = 0.30;
    end
    if nargin < 4 || isempty(noise_scale)
        noise_scale = 5;
    end

    [N, num_features] = size(trainX);
    n_corrupt = round(noise_ratio * N);

    trainX_noisy = trainX;
    trainY_noisy = trainY;

    if n_corrupt == 0
        corrupt_idx = [];
        return;
    end
 
    std_vals = std(trainX, 0, 1);

    corrupt_idx = randperm(N, n_corrupt);

    trainY_noisy(corrupt_idx) = trainY(corrupt_idx) * 10;

    for i = 1:n_corrupt
        r = corrupt_idx(i);
        c = randi(num_features);
        trainX_noisy(r, c) = trainX(r, c) + noise_scale * std_vals(c) * randn();
    end
end