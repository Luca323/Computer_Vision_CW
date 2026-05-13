function lgraph = applyFreezing(lgraph, freezeMode)
layers = lgraph.Layers;
numLayers = numel(layers);

switch freezeMode
    case 'all'
        % Freeze everything except the new head (last 2 layers)
        freezeUpTo = numLayers - 2;

    case 'partial'
        % Freeze roughly the first half (early low-level features)
        % ResNet18 has ~71 layers; this freezes roughly the first two residual blocks
        freezeUpTo = round(numLayers * 0.5);

    case 'none'
        % Fine-tune the whole network
        freezeUpTo = 0;

    otherwise
        error('Unknown freezeMode: %s', freezeMode);
end
for i = 1:freezeUpTo
    layer = layers(i);

    % Freeze weights if available
    if isprop(layer,'WeightLearnRateFactor')
        layer.WeightLearnRateFactor = 0;
    end

    % Freeze bias if available
    if isprop(layer,'BiasLearnRateFactor')
        layer.BiasLearnRateFactor = 0;
    end

    % Replace updated layer back into graph
    lgraph = replaceLayer(lgraph, layer.Name, layer);
end

fprintf('Freeze mode: %s — froze %d of %d layers\n', freezeMode, freezeUpTo, numLayers);
end