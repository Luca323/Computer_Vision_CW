function [features, labels] = extractHOG(imds, imageSize, cellSize)
    labels = imds.Labels;
    numImages = numel(imds.Files);
    imageFiles = imds.Files;

    sample = imread(imageFiles{1});
    sample = imresize(sample, imageSize);
    sampleFeatures = extractHOGFeatures(sample, 'CellSize', cellSize);
    numFeatures = length(sampleFeatures);

    features = zeros(numImages, numFeatures);
    features(1, :) = sampleFeatures;

    for i = 2:numImages
        img = imread(imageFiles{i});
        img = imresize(img, imageSize);
        hogFeatures = extractHOGFeatures(img, 'CellSize', cellSize);  % ← renamed
        features(i, :) = hogFeatures;
    end
end