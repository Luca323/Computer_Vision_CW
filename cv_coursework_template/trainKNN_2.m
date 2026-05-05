function model = trainKNN_2(X, y, d, k)
    
    model = fitcknn(X, y, 'Distance', d, 'NumNeighbors', k);

end