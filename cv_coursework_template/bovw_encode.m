function [features, labels] = bovw_encode(imds, imgSize, vocab, bovw_params)
    step      = bovw_params.stepSize;
    numWords  = bovw_params.numWords;
    useColour = bovw_params.useColour;
    
    labels    = imds.Labels;
    numImages = numel(imds.Files);
    features  = zeros(numImages, numWords);

    for i = 1:numImages
        img = readimage(imds, i);
        img = imresize(img, imgSize);
        img = im2single(img);

        % Define points FIRST before any feature extraction
        [rows, cols] = size(img(:,:,1));
        [X, Y] = meshgrid(1:step:cols, 1:step:rows);
        points = SURFPoints([X(:), Y(:)], 'Scale', 4);

        % Extract descriptors
        if useColour && size(img, 3) == 3
            desc = [];
            for ch = 1:3
                [d, ~] = extractFeatures(img(:,:,ch), points, 'Method', 'SURF');
                desc = [desc, double(d)];
            end
        else
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            [desc, ~] = extractFeatures(img, points, 'Method', 'SURF');
            desc = double(desc);
        end

        if isempty(desc)
            continue;
        end

        wordIdx = knnsearch(vocab, desc, 'K', 1);
        h = histcounts(wordIdx, 1:numWords+1);
        if sum(h) > 0
            h = h / sum(h);
        end

        features(i, :) = h;

        if mod(i, 100) == 0
            fprintf('Encoding %d/%d\n', i, numImages);
        end
    end
end