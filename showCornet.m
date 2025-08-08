function showNNRSMs
interpSets = {[1:6], [7: 12], [13:18], [19:24], [25:30], [31:36], [37:42], [43:48]};


%load the cornet data in
load('~/data/interp/allStimuliCORnetCorrMatricesSoftmaxedMasked.mat') 

%get the unmapped layers
uEVC = layer_0;
uMVC = layer_2;
uVVS = layer_3;

%get the reweighted layers
EVC = layer_0 * 0.6949063860711535 + layer_2 * 0.6919466727995444 + layer_3 * 0.5732322238855516;
MVC = layer_0 * 0.5034904276278029 + layer_2 * 0.6533835111893872 + layer_3 * 0.6068859137391336;
VVS = layer_0 * 0.40375262721824523 + layer_2 * 0.60788598213458 + layer_3 * 0.6042013378392591;


keyboard
%%



plotMatrices(uVVS/8,interpSets)
%%


figure, hold on

plot(1:5, [0.6949063860711535 0.6986247996940109 0.6919466727995444 0.5732332945437436 0.4756879576494478], 'color', [0.2 0.7 0.3])
scatter(1:5, [0.6949063860711535 0.6986247996940109 0.6919466727995444 0.5732332945437436 0.4756879576494478], 'filled', 'MarkerFaceColor', [0.2 0.7 0.3], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5)

plot(1:5, [0.5034904276278029 0.5624616383719798 0.6533835111893872 0.6068859137391336 0.488622592583327], 'color', [0 0.15 0.85])
scatter(1:5, [0.5034904276278029 0.5624616383719798 0.6533835111893872 0.6068859137391336 0.488622592583327], 'filled', 'MarkerFaceColor', [0 0.15 0.85], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5)


plot(1:5, [0.40375262721824523 0.4804947990858432 0.60788598213458 0.6042013378392591 0.5921885249142957], 'color', [0.7 0.3 1])
scatter(1:5, [0.40375262721824523 0.4804947990858432 0.60788598213458 0.6042013378392591 0.5921885249142957], 'filled', 'MarkerFaceColor', [0.7 0.3 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5)

xticks([1 2 3 4 5])
xticklabels({'V1', 'V2', 'V4', 'IT', 'D'})
xlabel('Model layer')
ylabel('NSD Ridge prediction (pearson)')
xlim([0.5 5.5])
ylim([0.3 0.8])




%%
figure, hold on
colors = hsv(8);
colors(4,:) = [0.13 0.55 0.13];

n_sets = length(interpSets);
n_x = 6;
allCats = nan(n_sets, n_x);

% Collect all categorical indices
for set = 1:n_sets
    allCats(set, 1) = computeCategoricalIndex(uEVC(interpSets{set}, interpSets{set}), 1);
    allCats(set, 2) = computeCategoricalIndex(uMVC(interpSets{set}, interpSets{set}), 1);
    allCats(set, 3) = computeCategoricalIndex(uVVS(interpSets{set}, interpSets{set}), 1);
    allCats(set, 4) = computeCategoricalIndex(EVC(interpSets{set}, interpSets{set}), 1);
    allCats(set, 5) = computeCategoricalIndex(MVC(interpSets{set}, interpSets{set}), 1);
    allCats(set, 6) = computeCategoricalIndex(VVS(interpSets{set}, interpSets{set}), 1);
end

% Plot all data at once per x-value
for x = 1:n_x
    y = allCats(:, x);
    x_vals = repmat(x - 0.1, n_sets, 1);
    scatter(x_vals, y, 36, ...
        'filled', ...
        'CData', colors(1:n_sets, :), ...
        'MarkerEdgeColor', 'w', ...
        'MarkerFaceAlpha', 0.5, ...
        'XJitter', 'randn', ...
        'XJitterWidth', 0.2);

    % Mean point at x + 0.1
    meanVal = mean(y, 'omitnan');
    scatter(x + 0.1, meanVal, 60, 'k', 'filled', 'MarkerEdgeColor', 'w');
    errorbar(x+0.1, meanVal, std(y), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)

end

xlabel('Cornet layer')
xticks(1:6)
xticklabels({'V1', 'V4', 'IT', 'V1', 'V4', 'IT'})
ylabel('Categorical influence (index)')
xlim([0.5 6.5])
ylim([-0.2 1])










%%

function plotMatrices(rsm, interpSets)
figure
averageRSM = zeros(6);
for interpSet = 1:length(interpSets)
    subplot(6,2,interpSet + 4)
    imagesc(rsm(interpSets{interpSet},interpSets{interpSet})), colormap('hot'), axis('off')
    averageRSM = averageRSM + rsm(interpSets{interpSet},interpSets{interpSet});
end
keyboard
subplot(3,1,1), imagesc(averageRSM), colormap('hot'), axis('off')




function [catIndex, r2] = computeCategoricalIndex(inputRSM, normalize)
    % Optional normalization
    if normalize
        inputRSM = (inputRSM - min(inputRSM(:))) / (max(inputRSM(:)) - min(inputRSM(:)));
    end

    % Null hypothesis models
    categoricalRSMCovar = [ones(3) zeros(3); zeros(3) ones(3)];
    linearRSMCovar = max(0, 1 - 0.2 * abs((1:6)' - (1:6)));

    % Scale models based on input variance
    inputVar = sqrt(diag(inputRSM));
    varMatrix = inputVar * inputVar';
    categoricalRSM = varMatrix .* categoricalRSMCovar;
    linearRSM = varMatrix .* linearRSMCovar;

    % Compute betas and r2
    [categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM);
    catIndex = categoricalBeta / (categoricalBeta + linearBeta);



function [categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM)
    mask = ~eye(size(inputRSM));
    objective = @(b) sum((inputRSM(mask) - (b(1) * categoricalRSM(mask) + b(2) * linearRSM(mask))).^2);
    betas = fminsearch(objective, [0.5, 0.5]);

    fitRSM = betas(1) * categoricalRSM + betas(2) * linearRSM;
    r2 = corr(fitRSM(mask), inputRSM(mask))^2;

    total = betas(1) + betas(2);
    categoricalBeta = betas(1) / total;
    linearBeta = betas(2) / total;

