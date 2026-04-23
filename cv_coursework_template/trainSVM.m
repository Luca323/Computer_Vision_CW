function model = trainSVM(X, y)

    t = templateSVM("BoxConstraint",1, Standardize=true, KernelFunction="linear");

    model = fitcsvm(X, y, 'Learners', t);

end