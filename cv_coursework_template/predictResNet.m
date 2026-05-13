function yhat = predictTransferCNN(netStruct, imdsTest)
    augimdsTest = augmentedImageDatastore(netStruct.inputSize, imdsTest);
    yhat = classify(netStruct.net, augimdsTest);
end