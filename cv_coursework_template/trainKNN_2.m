function model = trainKNN(X, y, k)

    model = fitcknn(X, y, 'NumNeighbors', k);

end