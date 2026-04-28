function vocab = bovw_buildVocab(imds, imgSize, bovw)
    numWords = bovw.numWords;
    step     = bovw.stepSize;
    
    allDescriptors = [];
    numImgs = numel(imds.Files);
    
    for i = 1:numImgs
        img = readimage(imds, i);
        img = imresize(img, imgSize);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        img = im2single(img);
        
        [rows, cols] = size(img);
        [X, Y] = meshgrid(1:step:cols, 1:step:rows);
        
        % SURFPoints needs Scale specified - this is the common fix
        pts = SURFPoints([X(:), Y(:)], 'Scale', 4);
        
        if pts.Count == 0
            continue;
        end
        
        [desc, validPts] = extractFeatures(img, pts, 'Method', 'SURF');
        desc = double(desc);
        
        if ~isempty(desc)
            allDescriptors = [allDescriptors; desc];
        end
        
        if mod(i, 100) == 0
            fprintf('Processed %d/%d, descriptors so far: %d\n', i, numImgs, size(allDescriptors,1));
        end
    end
    
    fprintf('Total descriptors: %d\n', size(allDescriptors, 1));
    
    % Safety check
    if isempty(allDescriptors)
        error('No descriptors extracted - check images are loading correctly');
    end
    
    [~, vocab] = kmeans(allDescriptors, numWords, ...
        'Replicates', 1, ...
        'MaxIter', 300, ...
        'Display', 'final');
end