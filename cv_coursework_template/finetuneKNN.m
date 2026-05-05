function bestParams = finetuneKNN(X, y)

    kValues = [1 3 5 7 9];
    distances = {'euclidean', 'minkowski', 'chebychev'};
    
    bestParams = struct();
    bestAcc = 0;
    
    cv = cvpartition(y, 'kFold', 5);
    
    for k = kValues
        for d = 1:length(distances)
            mdlCV = fitcknn(X, y, 'NumNeighbors', k, 'Distance', distances{d}, 'CVPartition', cv);
            acc = 1 - kfoldLoss(mdlCV);
            if acc > bestAcc
                bestParams.acc = acc;
                bestParams.k = k;
                bestParams.distance = distances{d};
            end
        end
    end

    disp(bestParams)
end