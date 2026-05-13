function [features, labels] = extractTinyImages_2(imds, thumbnailSize)
    labels = imds.Labels;
    numImages = numel(imds.Files);
    tSize = thumbnailSize(1); 
    
    numFeatures = tSize * tSize;
    features = zeros(numImages, numFeatures);
    
    for i = 1:numImages
        img = imread(imds.Files{i});
        
        %convert to Greyscale
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        imgTiny = imresize(img, [tSize, tSize]);
        
        %Normalise the matrix
        imgVector = double(imgTiny(:)) / 255;
 
        
        features(i, :) = imgVector';
    end
end