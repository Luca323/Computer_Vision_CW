function model = trainRF(X, y)
    numTrees = 100;
    model = TreeBagger(numTrees, X, y, ...
        'Method','classification', ...
        'OOBPrediction', 'on');

end

function yhat = PredictRF(model, Xtest)
    yhat = predict(model, Xtest);
    yhat = categorical(yhat);

end