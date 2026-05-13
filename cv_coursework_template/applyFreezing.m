function lgraph = applyFreezing(lgraph, freezeMode)
layers = lgraph.Layers;
numLayers = numel(layers);

switch freezeMode
    case 'all'
        %freeze everything except the last 2 layers
        freezeUpTo = numLayers - 2;

    case 'partial'
        %freeze roughly the first half (early low-level features)
        freezeUpTo = round(numLayers * 0.5);

    case 'none'
        %tune the whole network
        freezeUpTo = 0;

    otherwise
        error('Unknown freezeMode: %s', freezeMode);
end
for i = 1:freezeUpTo
    layer = layers(i);

    %Freeze weights if available
    if isprop(layer,'WeightLearnRateFactor')
        layer.WeightLearnRateFactor = 0;
    end

    if isprop(layer,'BiasLearnRateFactor')
        layer.BiasLearnRateFactor = 0;
    end

    lgraph = replaceLayer(lgraph, layer.Name, layer);
end

fprintf('Freeze mode: %s — froze %d of %d layers\n', freezeMode, freezeUpTo, numLayers);
end