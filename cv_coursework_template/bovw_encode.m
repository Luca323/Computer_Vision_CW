function [features, labels] = bovw_encode(imds, imgSize, vocab, bovw_details)
    step = bovw_details.stepSize;
    numWords = bovw_details.numWords;

    labels = imds.Labels;
    numImgs = numel(imds.Files);
    features = zeros(numImgs, numWords);
    for i = 1:numImgs
        img = imread(imds.Files{i});
        img = imresize(img, imgSize);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        img = im2single(img);

        % duplicated from buildVocab
        [rows, cols] = size(img);
        [X, Y] = meshgrid(1:step:cols, 1:step:rows);
        points = SURFPoints([X(:), Y(:)]);
        
        [desc, ~] = extractFeatures(img, points, 'Method', 'SURF');
        desc = double(desc);

        dists = pdist2(desc, vocab);
        [~, wordIdx] = min(dists, [], 2);

        h = histcounts(wordIdx, 1:numWords+1);
        if sum(h) > 0
            h = h/ sum(h);
        end

        features(i, :) = h;
    
    end


end