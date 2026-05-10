function model = trainRF(X, y)
    numTrees = 200;
    model = TreeBagger(numTrees, X, y, ...
        'Method','classification', ...
        'OOBPrediction', 'on');

end
