function plotMonkeyData

%load the data in
red = load('~/data/interp/monkeyData/reddata.mat');
paul = load('~/data/interp/monkeyData/pauldata.mat');
venus = load('~/data/interp/monkeyData/venusdata.mat');



keyboard



%% plot reliability of each monkey
save = 0;

figure(120)
histogram(red.reliability, [0:0.1:1])
xlabel('Reliability (split-half r)')
ylabel('Count')
title('Monkey 1')
if save
    drawPublishAxis('labelFontSize=8','figSize=[4, 3]','lineWidth=0.5');
    legend('off')
    savepdf(figure(120), '~/Desktop/catfigs/comps/monkey/redReliability')
end

figure(121)
histogram(paul.reliability, [0:0.1:1])
xlabel('Reliability (split-half r)')
ylabel('Count')
title('Monkey 2')
if save
    drawPublishAxis('labelFontSize=8','figSize=[4, 3]','lineWidth=0.5');
    legend('off')
    savepdf(figure(121), '~/Desktop/catfigs/comps/monkey/paulReliability')
end

figure(122)
histogram(venus.reliability(strcmp(venus.areanums, 'Late')), [0:0.1:1])
xlabel('Reliability (split-half r)')
ylabel('Count')
title('Monkey 3')
if save
    drawPublishAxis('labelFontSize=8','figSize=[4, 3]','lineWidth=0.5');
    legend('off')
    savepdf(figure(122), '~/Desktop/catfigs/comps/monkey/venusReliability')
end



%% plot the RSMs

figure, imagesc(squeeze(mean(red.corr_mtx_results.l_aIT,1)))
colormap('hot'), colorbar

figure, imagesc(squeeze(mean(paul.corr_mtx_results.cIT,1)))
colormap('hot'), colorbar

figure, imagesc(squeeze(mean(venus.corr_mtx_results.venus_neuropixel.Late,1)))
colormap('hot'), colorbar



%% plot the categorical indices

redcs = []; redrs = []; redms = [];
for set = 1:45;
    [c r m] = computeCategoricalIndex(squeeze(red.corr_mtx_results.l_aIT(set,:,:)), 1);
    redcs = [redcs c]; redrs = [redrs r]; redms = [redms m];
end


paulcs = []; paulrs = []; paulms = [];
for set = 1:45;
    [c r m] = computeCategoricalIndex(squeeze(paul.corr_mtx_results.cIT(set,:,:)), 1);
    paulcs = [paulcs c]; paulrs = [paulrs r]; paulms = [paulms m];
end


venuscs = []; venusrs = []; venusms = [];
for set = 1:45;
    [c r m] = computeCategoricalIndex(squeeze(venus.corr_mtx_results.venus_neuropixel.V4(set,:,:)), 1);
    venuscs = [venuscs c]; venusrs = [venusrs r]; venusms = [venusms m];
end


cutoff = 0.6;
figure, hold on
%red
filtered_red = redcs(redrs > cutoff);
scatter(repmat(0.9,length(filtered_red),1), filtered_red, 'filled', 'markerFaceColor', 'b', 'markerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'XJitter', 'randn', 'XJitterWidth', 0.2);
errorbar(1+0.1, mean(filtered_red), std(filtered_red), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
scatter(1 + 0.1, mean(filtered_red), 60, 'k', 'filled', 'MarkerEdgeColor', 'w');
%paul
filtered_paul = paulcs(paulrs > cutoff);
scatter(repmat(1.9,length(filtered_paul),1), filtered_paul, 'filled', 'markerFaceColor', [0.13 0.55 0.13], 'markerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'XJitter', 'randn', 'XJitterWidth', 0.2);
errorbar(2+0.1, mean(filtered_paul), std(filtered_paul), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
scatter(2 + 0.1, mean(filtered_paul), 60, 'k', 'filled', 'MarkerEdgeColor', 'w');

xlabel('Monkey')
xticks([1 2])
ylabel('Categorical influence (index)')




%%
function out = normalize_slices(mat)
out = (mat - min(mat,[],[2 3])) ./ (max(mat,[],[2 3]) - min(mat,[],[2 3]));


function [catIndex, r2, midpoint] = computeCategoricalIndex(inputRSM, normalize)
    % Optional normalization
    if normalize
        mn = min(inputRSM(:)); mx = max(inputRSM(:));
        if mx > mn, inputRSM = (inputRSM - mn) / (mx - mn); end
    end

    N = size(inputRSM,1);
    d = abs((1:N)' - (1:N));                 % pairwise integer distances
    linearRSMCovar = max(0, 1 - d/(N-1));    % size-agnostic linear kernel in [0,1]

    % Scale models by per-item variance (diag) like before
    inputVar   = sqrt(abs(diag(inputRSM)));  % abs for safety if small negatives
    varMatrix  = inputVar * inputVar';
    linearRSM  = varMatrix .* linearRSMCovar;

    % Fit categorical/linear betas AND the categorical midpoint
    [categoricalBeta, linearBeta, r2, midpoint] = findCatLinearEvidence(inputRSM, linearRSM, varMatrix);

    % Categorical index
    total = categoricalBeta + linearBeta;
    if total <= 0
        catIndex = NaN;
    else
        catIndex = categoricalBeta / total;
    end



function [categoricalBeta, linearBeta, r2, midpoint] = findCatLinearEvidence(inputRSM, linearRSM, varMatrix)
    N = size(inputRSM,1);
    mask = ~eye(N);

    % Objective over [b_cat, b_lin, m]; m is rounded to integer inside
    obj = @(p) ...
        sum( ...
            ( inputRSM(mask) - modelFromParams(p, linearRSM, varMatrix, N, mask) ).^2 );

    % Init: betas 0.5/0.5, midpoint at half
    p0 = [0.5, 0.5, (N/2)];
    p  = fminsearch(obj, p0, optimset('Display','off'));

    % Extract fitted pieces
    m_fit = max(1, min(N-1, round(p(3))));
    categoricalCovar = blkdiag(ones(m_fit), ones(N-m_fit));   % block-ones by midpoint
    categoricalRSM   = varMatrix .* categoricalCovar;

    betas = [p(1), p(2)];
    fitRSM = betas(1)*categoricalRSM + betas(2)*linearRSM;
    r2 = corr(fitRSM(mask), inputRSM(mask))^2;

    total = sum(betas);
    if total > 0
        categoricalBeta = betas(1)/total;
        linearBeta      = betas(2)/total;
    else
        categoricalBeta = 0; linearBeta = 0;
    end

    midpoint = m_fit;



function diffVec = modelFromParams(p, linearRSM, varMatrix, N, mask)
    % Build categorical RSM from rounded midpoint, combine with betas, return diff on mask
    m = max(1, min(N-1, round(p(3))));
    categoricalCovar = blkdiag(ones(m), ones(N-m));
    categoricalRSM   = varMatrix .* categoricalCovar;
    fitRSM = p(1)*categoricalRSM + p(2)*linearRSM;
    diffVec = inputParser; %#ok<NASGU> % dummy to keep function local-only
    diffVec = fitRSM(mask) - 0;       % will be subtracted in obj with inputRSM(mask)












