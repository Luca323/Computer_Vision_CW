function model = trainKNN(X, y, k)
    
    model = fitcknn(X, y, 'Distance', 'euclidean', 'NumNeighbors', k);

end