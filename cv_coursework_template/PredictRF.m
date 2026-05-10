function yhat = PredictRF(model, Xtest)
    yhat = predict(model, Xtest);
    yhat = categorical(yhat);

end