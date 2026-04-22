function [features, labels] = extractTinyImages_2(imds, thumbnailSize)
    labels = imds.Labels;
    numImages = numel(imds.Files);
    tSize = thumbnailSize(1); 
    
    % Now only 256 features (16x16x1)
    numFeatures = tSize * tSize;
    features = zeros(numImages, numFeatures);
    
    for i = 1:numImages
        img = imread(imds.Files{i});
        
        % 1. Convert to Grayscale immediately
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % 2. Resize to the tiny thumbnail
        imgTiny = imresize(img, [tSize, tSize]);
        
        % 3. Flatten and use simple 0-1 scaling
        % Dividing by 255 is often more stable for k-NN than unit-length norm
        imgVector = double(imgTiny(:)) / 255;
        
        % 4. Optional: Simple Zero-mean (uncomment if accuracy stays at 6%)
        % imgVector = imgVector - mean(imgVector);
        
        features(i, :) = imgVector';
    end
end