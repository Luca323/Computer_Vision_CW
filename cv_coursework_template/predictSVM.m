function yhat = predictSVM(modelStruct, Xtest)
    numImages = size(Xtest, 1);
    numClasses = numel(modelStruct.BinaryModels);
    scores = zeros(numImages, numClasses);

    for i = 1:numClasses
        currentModel = modelStruct.BinaryModels{i};
        [predLabels, binaryScore] = predict(currentModel, Xtest);
        
        % Safely find which column corresponds to the positive class (label == 1)
        posClassCol = find(currentModel.ClassNames == 1);
        scores(:, i) = binaryScore(:, posClassCol);
    end

    [~, maxIdx] = max(scores, [], 2);
    yhat = modelStruct.ClassNames(maxIdx);
end