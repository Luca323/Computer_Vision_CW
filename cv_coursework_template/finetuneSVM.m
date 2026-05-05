function bestParams = finetuneSVM(X, y);

kernels

bestParams = struct();
bestAcc = 0;

cv = cvpartition(y, 'kFold', 5);

for k = kValues
    for d = 1:length(distances)
        mdlCV = fitcknn(X, y, 'NumNeighbours', k, 'Distance', d, 'CVPartition', cv);
        acc = 1 - kfoldLoss(mdlCV);
        if acc > bestAcc
            bestAcc = acc;
            bestParams.k = k;
            bestParams.distance = distances{d};
        end
    end
end