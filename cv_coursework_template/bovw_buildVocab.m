function vocab = bovw_buildVocab(imds, imgSize, bovw)
    numWords   = bovw.numWords;
    step       = bovw.stepSize;
    useColour  = bovw.useColour;
    
    allDescriptors = {};
    numImgs = numel(imds.Files);

    for i = 1:numImgs
        img = readimage(imds, i);
        img = imresize(img, imgSize);
        img = im2single(img);

        % Define points FIRST
        [rows, cols] = size(img(:,:,1));
        [X, Y] = meshgrid(1:step:cols, 1:step:rows);
        pts = SURFPoints([X(:), Y(:)], 'Scale', 4);

        if pts.Count == 0
            continue;
        end

        % Extract descriptors
        if useColour && size(img, 3) == 3
            desc = [];
            for ch = 1:3
                [d, ~] = extractFeatures(img(:,:,ch), pts, 'Method', 'SURF');
                desc = [desc, double(d)];
            end
        else
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            [desc, ~] = extractFeatures(img, pts, 'Method', 'SURF');
            desc = double(desc);
        end

        if ~isempty(desc)
            allDescriptors{end+1} = desc;
        end

        if mod(i, 100) == 0
            fprintf('Processed %d/%d\n', i, numImgs);
        end
    end

    allDescriptors = vertcat(allDescriptors{:});
    fprintf('Total descriptors: %d\n', size(allDescriptors, 1));

    if isempty(allDescriptors)
        error('No descriptors extracted - check images are loading correctly');
    end

    [~, vocab] = kmeans(allDescriptors, numWords, ...
        'Replicates', 1, ...
        'MaxIter',    300, ...
        'Display',    'final');
end