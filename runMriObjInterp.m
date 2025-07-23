function runMriObjInterp

%% first, get all the data
subs = ['sub=s0605'; 'sub=s0606'; 'sub=s0607'; 'sub=s0608'; 'sub=s0609'; 'sub=s0610'; 'sub=s0612'; 'sub=s0613'; 'sub=s0614'; 'sub=s0615'; 'sub=s0616'; 'sub=s0617']
%subs = ['sub=s0613'; 'sub=s0614'; 'sub=s0615'; 'sub=s0616'; 'sub=s0617']
numSubs = size(subs,1)

for sub = 1:numSubs

    %get the values
    [unaveragedBrainCatVals, unaveragedBrainR2Vals, unaveragedCornetCatVals, unaveragedCornetR2Vals, unaveragedMldsCatVals, unaveragedMldsR2Vals, unaveragedBigRoiCatVals, unaveragedBigRoiR2Vals, unaveragedNNCatVals, unaveragedNNR2Vals, EVCRSM, MVCRSM, VVSRSM, BigROIRSM, VVSDotProduct, unaveragedConeCatVals, unaveragedConeR2Vals, mldsRSM, categoryTaskRSM] = mriObjInterp(subs(sub,:));

    %save them
    brainCatVals{sub} = unaveragedBrainCatVals;
    brainR2Vals{sub} = unaveragedBrainR2Vals;
    cornetCatVals{sub} = unaveragedCornetCatVals;
    cornetR2Vals{sub} = unaveragedCornetR2Vals;
    mldsCatVals{sub} = unaveragedMldsCatVals;
    mldsR2Vals{sub} = unaveragedMldsR2Vals;
    bigRoiCatVals{sub} = unaveragedBigRoiCatVals;
    bigRoiR2Vals{sub} = unaveragedBigRoiR2Vals;
    EVCRSMs{sub} = EVCRSM;
    MVCRSMs{sub} = MVCRSM;
    VVSRSMs{sub} = VVSRSM;
    BigROIRSMs{sub} = BigROIRSM;
    VVSDotProducts{sub} = VVSDotProduct;
    NNCatVals{sub} = unaveragedNNCatVals;
    NNR2Vals{sub} = unaveragedNNR2Vals;
    coneCatVals{sub} = unaveragedConeCatVals;
    coneR2Vals{sub} = unaveragedConeR2Vals;
    mldsRSMs{sub} = mldsRSM;
    categoryTaskRSMs{sub} = categoryTaskRSM;
    close all
end
%%
labels = getLabels;
numLabels = 8;
%colors = [linspace(0,34,numLabels)', linspace(0,139,numLabels)', linspace(139,34,numLabels)'] / 255;
colors = hsv(8);
%%

keyboard


%% sort the data to be usable - get out of cells
brainCatVals = cell2mat(brainCatVals');
brainR2Vals = cell2mat(brainR2Vals');
cornetCatVals = cell2mat(cornetCatVals');
cornetR2Vals = cell2mat(cornetR2Vals');
mldsCatVals = cell2mat(mldsCatVals');
mldsR2Vals = cell2mat(mldsR2Vals');
bigRoiCatVals = cell2mat(bigRoiCatVals');
bigRoiR2Vals = cell2mat(bigRoiR2Vals');
NNCatVals = cell2mat(NNCatVals');
NNR2Vals = cell2mat(NNR2Vals');
coneR2Vals = cell2mat(coneR2Vals');
coneCatVals = cell2mat(coneCatVals');
catTaskCatVals = brainCatVals(:,4);
catTaskR2Vals = brainR2Vals(:,4);



%% set cutoffs
catTaskR2Cutoff = 0.9;
mldsR2Cutoff = 0.8;
brainR2Cutoff = 0.5; 
NNR2Cutoff = 0.9;
coneR2Cutoff = 0.8;




%% Do some analyses
%brain vs mlds
figure, hold on
%plotMldsVals = []; plotBrainCatVals = [];
for area = 1:3,
    %subplot(1,3,area)
    scatter(brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area) , mldsCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff)), 'filled')
    plot([0 1], [0 1], '--k')
    %plotMldsVals = [plotMldsVals mldsCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff))']; plotBrainCatVals = [plotBrainCatVals brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area)'];
end
xlabel('Brain categorical index'), ylabel('Mlds categorical index'), title('Brain and MLDS categorical indices')
xlim([-0.2 1]), ylim([-0.2 1])
legend({'Early', '', 'Middle', '', 'Late'})


%brain vs categorization
figure, hold on
plotCatVals = []; plotBrainCatVals = [];
for area = 1:3,
    %subplot(1,3,area)
    scatter(brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area) , brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), 4), 'filled')
    plot([0 1], [0 1], '--k')
    plotCatVals = [plotCatVals brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), 4)']; plotBrainCatVals = [plotBrainCatVals brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area)'];
end
xlabel('Brain categorical index'), ylabel('Cat task categorical index'), title('Brain and categorization categorical indices')
xlim([-0.2 1]), ylim([-0.2 1])
legend({'Early', '', 'Middle', '', 'Late'})

[C,P]=corrcoef(plotBrainCatVals,plotCatVals); plot(plotBrainCatVals,polyval(polyfit(plotBrainCatVals,plotCatVals,1),plotBrainCatVals),'b-');
legend({'Early', '', 'Middle', '', 'Late', '', sprintf('Fit (r=%.2f, p=%.3f)',C(1,2),P(1,2))},'Location','best');

%mlds vs categorization
figure, hold on
scatter(mldsCatVals((mldsR2Vals > mldsR2Cutoff) & (catTaskR2Vals > brainR2Cutoff)), catTaskCatVals((mldsR2Vals > mldsR2Cutoff) & (catTaskR2Vals > brainR2Cutoff)), 'filled')
xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
xlabel('Mlds categorical index'), ylabel('Categorization task categorical index'), title('Mlds and Categorization task categorical indices')



%Whole brain vs mlds
figure, hold on
scatter(mldsCatVals((mldsR2Vals > mldsR2Cutoff) & (bigRoiR2Vals > brainR2Cutoff)), bigRoiCatVals((mldsR2Vals > mldsR2Cutoff) & (bigRoiR2Vals > brainR2Cutoff)), 'filled')
xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
xlabel('Mlds categorical index'), ylabel('Brain categorical index'), title('Whole brain categorical index vs mlds')



%% do "classification" with the brain data by comparing how close the endpoints are

plotPsychometricsFromRSMs(VVSRSMs, numSubs)




%% Make the big plot
figure, hold on, xlim([0.5 4.5]), ylim([-0.1 1.1])


% Initialize storage
filtered_NNCatVals = cell(size(NNR2Vals,3), 4);
filtered_cornetCatVals = cell(1,4);
filtered_brainCatVals = cell(1,3);
filtered_mldsCatVals = [];
filtered_coneCatVals = [];
filtered_catTaskCatVals = [];

% Plot the models and store values
for model = 1:size(NNR2Vals,3)
    for area = 1:4
        temp_vals = [];
        for interp = 1:length(brainR2Vals)
            if (NNR2Vals(interp, area, model) > NNR2Cutoff)
                val = NNCatVals(interp, area, model);
                scatter(area-0.1, val, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
                temp_vals(end+1) = val;
            end
        end
        filtered_NNCatVals{model, area} = temp_vals;
    end
end

% Cornet model
for area = 1:4
    temp_vals = [];
    for interp = 1:length(cornetR2Vals)
        if (cornetR2Vals(interp, area) > NNR2Cutoff)
            val = cornetCatVals(interp, area);
            scatter(area-0.1, val, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
            temp_vals(end+1) = val;
        end
    end
    filtered_cornetCatVals{area} = temp_vals;
end

% Brain data
for area = 1:3
    temp_vals = [];
    for interp = 1:length(brainR2Vals)
        if (brainR2Vals(interp, area) > brainR2Cutoff) && ...
           (mldsR2Vals(interp) > mldsR2Cutoff) && ...
           (catTaskR2Vals(interp) > catTaskR2Cutoff)
            val = brainCatVals(interp, area);
            scatter(area+0.1, val, 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
            temp_vals(end+1) = val;
        end
    end
    filtered_brainCatVals{area} = temp_vals;
end

% MLDS
for interp = 1:length(mldsR2Vals)
    if (mldsR2Vals(interp) > mldsR2Cutoff)
        val = mldsCatVals(interp);
        scatter(4.1, val, 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
        filtered_mldsCatVals(end+1) = val;
    end
end

% CONES
for interp = 1:length(coneR2Vals)
    if (coneR2Vals(interp) > coneR2Cutoff)
        val = coneCatVals(interp);
        scatter(0.8, val, 36, 'filled', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
        filtered_coneCatVals(end+1) = val;
    end
end


% Categorization task
for interp = 1:length(catTaskR2Vals)
    if (catTaskR2Vals(interp) > catTaskR2Cutoff)
        val = catTaskCatVals(interp);
        scatter(4.1, val, 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
        filtered_catTaskCatVals(end+1) = val;
    end
end

% Axis labels
xticks(1:4), xticklabels({'Early', 'Middle', 'Late', 'Behavior'}), xlabel('ROI/Layer')
ylabel('Categorical index')
title('Categorical influence on RSM')

%legend
h1 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Neural network');
h2 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Brain');
h3 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'MLDS');
h4 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Categorization task');
h5 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Cone mosaic');
legend([h1, h2, h3, h4, h5]);


save = 0;
if save
    drawPublishAxis('labelFontSize=8','figSize=[8, 7]','lineWidth=0.5');
    savepdf(figure(2),'~/Desktop/catFigs/comps/bigFigure') ;
end


%% show the human brain RSMs
figure
for sub = 1:length(VVSRSMs)
    subplot(4,3,sub)
    imagesc(VVSRSMs{sub}), colorbar, colormap('hot')
end





%% plot the behavioral data
interpSets = {[1:6], [7:12], [13:18], [19:24]};
k=1;
for sub = 1:length(mldsRSMs)
    for interpSet = 1:length(mldsRSMs{sub})/6
        %get the mlds curve
        y = mldsRSMs{sub}(interpSets{interpSet},max(interpSets{interpSet}));
        x = 1:6;
        mldsFit = fitCumulativeGaussian(x,y);
        %plot
        figure(70), hold on
        plot(mldsFit.fitX, mldsFit.fitY, 'Color', [colors(labels{sub}(interpSet),:) 0.5]);

        %categorization
        y = categoryTaskRSMs{sub}(interpSets{interpSet},max(interpSets{interpSet}));
        x = 1:6;
        catFit = fitCumulativeGaussian(x,y);
        %plot
        figure(71), hold on
        plot(catFit.fitX, catFit.fitY, 'Color', [colors(labels{sub}(interpSet),:) 0.5]);

        %scatters
        figure(72), hold on
        scatter(mldsFit.mean, catFit.mean, [], colors(labels{sub}(interpSet),:), 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w')

        figure(73), hold on
        scatter(mldsFit.std, catFit.std, [], colors(labels{sub}(interpSet),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w')
        
        %mlds vs categorical index
        figure(74), hold on, scatter(mldsCatVals(k), mldsFit.std, 'MarkerFaceColor', colors(labels{sub}(interpSet),:), 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w'), k = k+1;
    end
end

figure(70), title('MLDS Curves'), xlabel('Interpolation number'), ylabel('Scale value'), xlim([1 6])
figure(71), title('Categorization Curves'), xlabel('Interpolation number'), ylabel('Frequency identified as object 2'), xlim([1 6])
figure(72), title('Mu values'), xlabel('Mlds Mu value'), ylabel('Categorization Mu value'), xlim([2.5 4.5]), ylim([2.5 4.5]), plot([2.5 4.5], [2.5 4.5], 'k-')
figure(73), title('Sigma values'), xlabel('Mlds sigma value'), ylabel('Categorization sigma value'), xlim([0 2]), ylim([0 2]), plot([0 6], [0 6], 'k-')
figure(74), title('Sigma vs categorical index'), xlabel('MLDS categorical Index'), ylabel('MLDS sigma Value'), xlim([-0.1 0.8]), ylim([0 1.5]);


save = 0;
if save
    for f = 70:73;                                                      
    figure(f);
    drawPublishAxis('labelFontSize=8','figSize=[6, 5]','lineWidth=0.5');
    legend('off')
    end

    savepdf(figure(70),'~/Desktop/catFigs/comps/mldsCurves') 
    savepdf(figure(71),'~/Desktop/catFigs/comps/catCurves')  
    savepdf(figure(72),'~/Desktop/catFigs/comps/muValues') 
    savepdf(figure(73),'~/Desktop/catFigs/comps/sigmaValues')    

    figure, hold on, for k = 1:8, scatter(k,k,'markerFaceColor', colors(k,:), 'MarkerEdgeColor', 'w'), end
end









%%
keyboard



function labels = getLabels;

labels{1} = [1 2 3 4]
labels{2} = [1 2 3 4];
labels{3} = [1 2 3 4];
labels{4} = [5 6 7];
labels{5} = [1 3 4];
labels{6} = [1 3 4];
% ADD S0611 HERE LATER labels{7} = [5 6 8];
labels{7} = [5 1 8];
labels{8} = [5 1 8];
labels{9} = [1 3 4];
labels{10} = [1 3 4];
labels{11} = [1 8 4];
labels{12} = [1 8 4]



%% 
function plotPsychometricsFromRSMs(RSMs, numSubs)

figure

interpSets = {[1:6], [7:12], [13:18], [19:24]};

x = []; y = []; trueY = [];
for sub = 1:numSubs
    for set = 1:length(RSMs{sub})/6
        low = min(interpSets{set}); high = max(interpSets{set});
        dists = RSMs{sub}(low:high,high) - RSMs{sub}(low:high,low);

        subplot(1,3,1), hold on, plot([1 6], [0 0], '--k'), xlim([0.5, 6.5])
        scatter(1:6, dists, 'k', 'filled', 'MarkerFaceAlpha', 0.2, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w')
        ylabel('Corr(interp,6) - Corr(interp, 1)'), xlabel('Interpolation point')
        title('Relative distance to endpoints')

        subplot(1,3,2), hold on, ylim([0.8 2.2]), xlim([0.5, 6.5])
        scatter(1:6, (dists>0)+1, 'k', 'filled', 'MarkerFaceAlpha', 0.1, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w')
        ylabel('Closest endpoint'), xlabel('Interpolation point')
        title('Classification of interpolations')
        x = [x 1:6]; y = [y dists>0]; trueY = [trueY dists];

    end
end

%plot the medians
subplot(1,3,1), scatter(1:6, mean(trueY,2), 48, 'cyan', 'filled')
subplot(1,3,2), scatter(1:6, mean(y,2)+1, 48, 'cyan', 'filled')


%normalize the correlation values between 0 and 1 so you can fit psychometrics
subplot(1,3,3), hold on, ylim([-0.1 1.1]), xlim([-0.1 6.1]), title('Relative distance (normalized)')
ylabel('Corr(interp,6) - Corr(interp, 1) (normalized)'), xlabel('Interpolation point')

scaledTrueY = trueY;
scaledTrueY = (scaledTrueY - min(min(scaledTrueY)))/(max(max(scaledTrueY)) - min(min(scaledTrueY)));

g = fitCumulativeGaussian(x,scaledTrueY(:)');
plot(g.fitX,g.fitY, 'k'), hold on
scatter(x, scaledTrueY(:), 'k', 'filled', 'MarkerFaceAlpha', 0.1, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w');
scatter(1:6, mean(scaledTrueY,2), 48, 'cyan', 'filled')

g2 = fitCumulativeGaussian(x,y(:)');
subplot(1,3,2), plot(g2.fitX,g2.fitY+1, 'k')

for interp = 1:6, choices(interp) = g.fitY(g.fitX == interp); end
brainCatTaskMatrix = 1- dist(choices);
[c r] = compareCatRSM(brainCatTaskMatrix, 1);






%%
function [catVals, r2s] = compareCatRSM(inputRSM, normalize)

inputRSMs{1} = inputRSM;

%init empty array
catVals = [];
r2s = [];
% create the 2 null hypothesis matrices
categoricalRSMCovar = [ones(3) zeros(3); zeros(3) ones(3)];
linearRSMCovar = max(0, 1 - 0.2 * abs((1:6)' - (1:6)));

for inputNum = 1:length(inputRSMs)
    %get the RSM
    inputRSM = inputRSMs{inputNum};
    %if not already normalized, normalize (unaveraged RSMs aren't normalized)
    if normalize
        inputRSM = (inputRSM - min(inputRSM(:))) / (max(inputRSM(:)) - min(inputRSM(:)));
    end
    %compute the idealized matrices based on measured variance
    inputVar = sqrt(diag(inputRSM));
    varMatrix = inputVar*inputVar';
    linearRSM = varMatrix .* linearRSMCovar;
    categoricalRSM = varMatrix .* categoricalRSMCovar;
    
    %find betas that describe input of linear/categorical matrices to observed matrix
    [categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM, inputNum);
    catVals = [catVals categoricalBeta / (categoricalBeta + linearBeta)];
    r2s = [r2s r2];

end



%%%%%%%%%%%%%%%%%%
%% findCatLinearEvidence %%
%%%%%%%%%%%%%%%%%%%%%
function [categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM, inputNum)

%mask to ignore diagonal - it's the same between both conditions but might cause a shitty fit in the neural networks
mask = ~eye(size(inputRSM));

%define the objective function to minimize absolute difference
objective = @(betas) sum(sum(abs(inputRSM(mask) - ...
    (betas(1) * categoricalRSM(mask) + betas(2) * linearRSM(mask))).^2));

% Optimize Betas using fminsearch
betas = fminsearch(objective, [.5, .5]); % Initial guesses for Betas


%r-squared of the fit matrix
fitRSM = categoricalRSM * betas(1) + linearRSM * betas(2);
r2 = corr(fitRSM(~eye(size(fitRSM))), inputRSM(~eye(size(inputRSM))))^2;

categoricalBeta = betas(1)/sum(betas);
linearBeta = betas(2)/sum(betas);








