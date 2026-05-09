function bestParams = finetuneKNN(X, y)
    kValues   = [1 3 5 7 9 11 13];
    distances = {'euclidean', 'minkowski', 'chebychev'};
    bestParams = struct();
    bestAcc    = 0;
    
    cv = cvpartition(y, 'KFold', 5);
    
    for k = kValues
        for d = 1:length(distances)
            mdlCV = fitcknn(X, y, ...
                'NumNeighbors', k, ...
                'Distance',     distances{d}, ...
                'CVPartition',  cv);
            
            acc = 1 - kfoldLoss(mdlCV);
            fprintf('k=%d, distance=%s, acc=%.4f\n', k, distances{d}, acc);
            
            if acc > bestAcc || (acc == bestAcc && k > bestParams.k)
                bestAcc            = acc;   % ← fix
                bestParams.acc     = acc;
                bestParams.k       = k;
                bestParams.distance = distances{d};
            end
        end
    end
    
    fprintf('Best: k=%d, distance=%s, acc=%.4f\n', bestParams.k, bestParams.distance, bestParams.acc);
end