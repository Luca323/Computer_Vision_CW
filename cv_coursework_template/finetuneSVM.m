function bestParams = finetuneSVM(X, y)

    boxContraints = [0.01 0.1 1 10 100];
    
    bestParams = struct();
    bestAcc = 0;
    
    cv = cvpartition(y, 'kFold', 5);

    classes = categories(y);
    numClasses = numel(classes);

    
    for b = boxContraints
        
        foldAcc = zeros(cv.NumTestSets, 1);
        
        %K folds
        for fold = 1:cv.NumTestSets
            trainIdx = training(cv, fold);
            testIdx  = test(cv, fold);

            Xtrain = X(trainIdx, :);
            ytrain = y(trainIdx);

            Xtest = X(testIdx, :);
            ytest = y(testIdx);
            
            modelStruct.BinaryModels = cell(numClasses, 1);
            modelStruct.ClassNames = classes;

            for i = 1:numClasses
                current = classes{i};
                binaryLabels = double(ytrain == current);   % 1 where current class, 0 elsewhere
                binaryLabels(binaryLabels == 0) = -1;  % convert 0s to -1
                
                modelStruct.BinaryModels{i} = fitcsvm(Xtrain, binaryLabels, 'KernelFunction', 'linear', ...
                    'Standardize', true, 'BoxConstraint', b);
            end

            preds = predictSVM(modelStruct, Xtest);


            % accuracy for this fold
            foldAcc(fold) = mean(preds == ytest);
        end

        acc = mean(foldAcc);

        fprintf('boxConstraint=%.2f, acc=%.4f\n', b, acc);


        if acc > bestAcc
            bestAcc = acc;
            bestParams.acc = acc;
            bestParams.boxConstraint = b;
        end
    end
    
    fprintf('Best: boxConstraint=%.2f, acc=%.4f\n', bestParams.boxConstraint, bestParams.acc);
end