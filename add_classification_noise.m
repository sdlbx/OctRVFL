function [trainY_noisy, corrupt_idx] = add_classification_noise(trainY, noise_ratio, n_class)

    if nargin < 3 || isempty(n_class)
        n_class = max(trainY);
    end

    trainY_noisy = trainY;
    N = length(trainY);
    n_corrupt = round(noise_ratio * N);

    if n_corrupt == 0 || n_class < 2
        corrupt_idx = [];
        return;
    end

    corrupt_idx = randperm(N, n_corrupt);
    all_classes = 1:n_class;

    for i = 1:n_corrupt
        idx = corrupt_idx(i);
        curr_label = trainY(idx);
        candidates = setdiff(all_classes, curr_label);
        trainY_noisy(idx) = candidates(randi(length(candidates)));
    end
end