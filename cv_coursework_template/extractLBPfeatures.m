function [features, labels] = extractLBP(imds, imageSize, lbp_params)
    %requires a parameter struct containing numNeighbours, radius and
    %upright
    
    numImages = numel(imds.Files);
    labels = imds.Labels;

    numNeighbours = lbp_params.numNeighbours;
    radius = lbp_params.radius;
    upright = lbp_params.upright;

    sample = readimage(imds, 1);
    sample = imresize(sample, imageSize);
    if size(sample, 3) ==3
        sample = rgb2gray(sample);
    end
    sampleFeatures = extractLBPFeatures(sample, ...
        "NumNeighbors", numNeighbours, ...
        "Radius", radius,"Upright", upright);
    numFeatures = numel(sampleFeatures);

    features = zeros(numImages, numFeatures);
    features(1,:) = sampleFeatures;

    for i = 2:numImages
        img = readimage(imds, i);
        img = imresize(img, imageSize);

        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        features(i, :) = extractLBPFeatures(img, ...
            "NumNeighbors", numNeighbours, ...
            "Radius", radius, "Upright", upright);
    end

end