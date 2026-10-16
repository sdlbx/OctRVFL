function METRICS = Evaluate(ACTUAL, PREDICTED, nclass)

    ACTUAL = ACTUAL(:); PREDICTED = PREDICTED(:);
    CM = confusionmat(ACTUAL, PREDICTED, 'Order', 1:nclass);
    
    accuracy = sum(diag(CM)) / sum(CM(:));
 
    METRICS = accuracy*100;
end