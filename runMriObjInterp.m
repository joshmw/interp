function runMriObjInterp

%% first, get all the data
subs = ['sub=s0605'; 'sub=s0606'; 'sub=s0607'; 'sub=s0608'; 'sub=s0609'; 'sub=s0610'; 'sub=s0612'; 'sub=s0613'; 'sub=s0614'; 'sub=s0615'; 'sub=s0616'; 'sub=s0617']
%subs = ['sub=s0613'; 'sub=s0614'; 'sub=s0615'; 'sub=s0616'; 'sub=s0617']
numSubs = size(subs,1)

for sub = 1:numSubs

    %get the values
    [unaveragedBrainCatVals, unaveragedBrainR2Vals, unaveragedCornetCatVals, unaveragedCornetR2Vals, unaveragedMldsCatVals, unaveragedMldsR2Vals, unaveragedBigRoiCatVals, unaveragedBigRoiR2Vals, unaveragedNNCatVals, unaveragedNNR2Vals, EVCRSM, MVCRSM, VVSRSM, BigROIRSM, EVCRSMBoot, MVCRSMBoot, VVSRSMBoot, BigROIRSMBoot, VVSDotProduct, unaveragedConeCatVals, unaveragedConeR2Vals, mldsRSM, categoryTaskRSM] = mriObjInterp(subs(sub,:));

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
    EVCRSMBoots{sub} = EVCRSMBoot;
    MVCRSMBoots{sub} = MVCRSMBoot;
    VVSRSMBoots{sub} = VVSRSMBoot;
    BigROIRSMBoots{sub} = BigROIRSMBoot;
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
%load('summaryDataMiniV1.mat'); noise = load('~/Data/interp/noiseFloorMiniV1.mat');
%load('summaryDataBenson10000Boots.mat'); noise = load('~/Data/interp/noiseFloorBenson100.mat');
load('~/Data/interp/summaryDataV1V4NewVentral10000Boots.mat');  noise = load('~/Data/interp/noiseFloorBenson100.mat');
labels = getLabels;
numLabels = 8;
%colors = [linspace(0,34,numLabels)', linspace(0,139,numLabels)', linspace(139,34,numLabels)'] / 255;
colors = hsv(8);
%colors(3,:) = [0.8 1.0 0.4];
colors(4,:) = [0.13 0.55 0.13];
interpSets = {[1:6], [7:12], [13:18], [19:24]};
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
catTaskR2Cutoff = 0.7;
mldsR2Cutoff = 0.7;
brainR2Cutoff = 0.3; 
NNR2Cutoff = 0.7;
coneR2Cutoff = 0.8;




%% do "classification" with the brain data by comparing how close the endpoints are
[allColors, allLabels, vvsMldsGaussFits, vvsCatGaussFits] = plotPsychometricsFromRSMs(VVSRSMs, VVSRSMBoots, numSubs, colors, labels);
%[allColors, allLabels, evcMldsGaussFits, evcCatGaussFits] = plotPsychometricsFromRSMs(EVCRSMs, EVCRSMBoots, numSubs, colors, labels);


save = 1;
if save
    figure(83), drawPublishAxis('labelFontSize=8','figSize=[6, 5]','lineWidth=0.5', 'xtick=[1:6]'); legend('off')
    savepdf(figure(83),'~/Desktop/catFigs/comps/brainMLDS')
    figure(84), drawPublishAxis('labelFontSize=8','figSize=[6, 5]','lineWidth=0.5', 'xtick=[1:6]'); legend('off')
    savepdf(figure(84),'~/Desktop/catFigs/comps/brainCategorization')
end



%% show the human brain RSMs
plotHumanBrainRSMs(EVCRSMs, interpSets, colors, labels)



%% plot brain reliability
plotHumanBrainReliability(EVCRSMs, noise.EVCRSMBoots, [0.2 0.7 0.3], -0.1)
plotHumanBrainReliability(MVCRSMs, noise.MVCRSMBoots, [0 0.15 0.85], 0)
plotHumanBrainReliability(VVSRSMs, noise.VVSRSMBoots, [0.7 0.3 1], 0.1)

save = 0;
if save
    drawPublishAxis('labelFontSize=8','figSize=[12, 5]','lineWidth=0.5', 'xtick=[1 3 5 7 9 11 13]');
    legend('off')
    savepdf(figure(60),'~/Desktop/catFigs/comps/brainReliability')
end



%% Make the big plot
figure(100), hold on,


% Initialize storage
filtered_NNCatVals = cell(size(NNR2Vals,3), 4);
filtered_cornetCatVals = cell(1,4);
filtered_brainCatVals = cell(1,3);
filtered_mldsCatVals = [];
filtered_coneCatVals = [];
filtered_catTaskCatVals = [];
orderedColors = allColors(1:6:end,:);

% Plot the models and store values
for area = 1:4
    all_models = [];
    for model = 1:size(NNR2Vals,3)
        temp_vals = []; tempColors = []; dupes = [100 100 100];
        for interp = 1:length(brainR2Vals)
            if sum(ismember(dupes,orderedColors(interp,:),'rows')) < 1
                if (NNR2Vals(interp, area, model) > NNR2Cutoff)
                    val = NNCatVals(interp, area, model);
                    tempColors = [tempColors; orderedColors(interp,:)];
                    temp_vals(end+1) = val;
                    dupes = [dupes; orderedColors(interp,:)];
                    all_models = [all_models val];
                end
            end
        end
        scatter(repmat(area-0.1,1,length(temp_vals)), temp_vals, 36, 'filled', 'CData', tempColors, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'XJitter', 'randn', 'XJitterWidth', 0.15);
        filtered_NNCatVals{model, area} = temp_vals;
    end
    scatter(area+0.1, mean(temp_vals), 72, 'black','filled', 'markerEdgeColor', 'w')
    errorbar(area+0.1, mean(temp_vals), std(temp_vals), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
end



% Cornet model
for area = 1:4
    temp_vals = []; tempColors = []; dupes = [100 100 100];
    for interp = 1:length(cornetR2Vals)
        if sum(ismember(dupes,orderedColors(interp,:),'rows')) < 1
            if (cornetR2Vals(interp, area) > NNR2Cutoff)
                val = cornetCatVals(interp, area);
                tempColors = [tempColors; orderedColors(interp,:)];
                temp_vals(end+1) = val;
                dupes = [dupes; orderedColors(interp,:)];
            end
        end
    end
    scatter(repmat(area,1,length(temp_vals)), temp_vals, 36, 'filled', 'CData', tempColors, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'XJitter', 'randn', 'XJitterWidth', 0.15);
    filtered_cornetCatVals{area} = temp_vals;
end
xticks(1:4), xticklabels({'Early', 'Middle', 'Late', 'Choice'}), xlabel('Network Layer')
ylabel('Categorical index')
title('Neural Network Representation Categoricalness')



%figure(101), hold on, xlim([0.5 5.5]), ylim([-0.1 1.1])

% Brain data
for area = 1:3
    temp_vals = []; tempColors = [];
    for interp = 1:length(brainR2Vals)
        if (brainR2Vals(interp, area) > brainR2Cutoff) && ...
           (mldsR2Vals(interp) > mldsR2Cutoff) && ...
           (catTaskR2Vals(interp) > catTaskR2Cutoff)
            val = brainCatVals(interp, area);
            tempColors = [tempColors; orderedColors(interp,:)];
            temp_vals(end+1) = val;
        end
    end
    scatter(repmat(area+4-0.1,1,length(temp_vals)), temp_vals, 36, 'filled', 'CData', tempColors, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'XJitter', 'randn', 'XJitterWidth', 0.15);
    filtered_brainCatVals{area} = temp_vals;
    scatter(area+4+0.1, mean(temp_vals), 72, 'black','filled', 'markerEdgeColor', 'w')
    errorbar(area+4+0.1, mean(temp_vals), std(temp_vals), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
end

% MLDS
temp_vals = []; tempColors = [];
for interp = 1:length(mldsR2Vals)
    if (mldsR2Vals(interp) > mldsR2Cutoff)
        val = mldsCatVals(interp);
        tempColors = [tempColors; orderedColors(interp,:)];
        temp_vals(end+1) = val;
        filtered_mldsCatVals(end+1) = val;
    end
end
scatter(repmat(5+4-0.1,1,length(temp_vals)), temp_vals, 36, 'filled', 'CData', tempColors, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'XJitter', 'randn', 'XJitterWidth', 0.15);
scatter(5+4+0.1, mean(temp_vals), 72, 'black','filled', 'markerEdgeColor', 'w')
errorbar(5+4+0.1, mean(temp_vals), std(temp_vals), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)


% % CONES
% for interp = 1:length(coneR2Vals)
%     if (coneR2Vals(interp) > coneR2Cutoff)
%         val = coneCatVals(interp);
%         scatter(0.8, val, 36, 'filled', 'MarkerFaceColor', orderedColors(interp,:), 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
%         filtered_coneCatVals(end+1) = val;
%     end
% end


% Categorization task
temp_vals = []; tempColors = [];
for interp = 1:length(catTaskR2Vals)
    if (catTaskR2Vals(interp) > catTaskR2Cutoff)
        val = catTaskCatVals(interp);
        tempColors = [tempColors; orderedColors(interp,:)];
        temp_vals(end+1) = val;
        filtered_catTaskCatVals(end+1) = val;
    end
end
scatter(repmat(4+4-0.1,1,length(temp_vals)), temp_vals, 36, 'filled', 'CData', tempColors, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'XJitter', 'randn', 'XJitterWidth', 0.2);
scatter(4+4+0.1, mean(temp_vals), 72, 'black','filled', 'markerEdgeColor', 'w')
errorbar(4+4+0.1, mean(temp_vals), std(temp_vals), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)


% Axis labels
%xticks(1:9), xticklabels({'Early', 'Middle', 'Late', 'Cat', 'Early', 'Middle', 'Late', 'Cat', 'MLDS Task'}), xlabel('Visual area/Behavioral task')
ylabel('Categorical index')
title('Human Brain Representational/Behavioral Categoricalness')

xlim([0.5 9.5]), ylim([-0.2 1.1])
save = 1;
if save
    drawPublishAxis('labelFontSize=8','figSize=[20, 8]','lineWidth=0.5'); legend('off')
    savepdf(figure(100),'~/Desktop/catFigs/comps/bigFigure') ;
end




%% plot the behavioral data
interpSets = {[1:6], [7:12], [13:18], [19:24]};
k=1;
behavioral_r2_cutoff = 0.7;
for sub = 1:length(mldsRSMs)
    for interpSet = 1:length(mldsRSMs{sub})/6

        %get the mlds curve
        y = mldsRSMs{sub}(interpSets{interpSet},max(interpSets{interpSet}));
        x = 1:6;
        behaviorMldsGaussFits{sub}{interpSet} = fitCumulativeGaussian(x,y);

        %categorization
        y = categoryTaskRSMs{sub}(interpSets{interpSet},max(interpSets{interpSet}));
        x = 1:6;
        behaviorCatGaussFits{sub}{interpSet} = fitCumulativeGaussian(x,y);


        %plot mlds and categorization
        if (behaviorMldsGaussFits{sub}{interpSet}.r2 > behavioral_r2_cutoff) & (behaviorCatGaussFits{sub}{interpSet}.r2 > behavioral_r2_cutoff)
            figure(70), hold on
            plot(behaviorMldsGaussFits{sub}{interpSet}.fitX, behaviorMldsGaussFits{sub}{interpSet}.fitY, 'Color', [colors(labels{sub}(interpSet),:) 0.5]);
            figure(71), hold on
            plot(behaviorCatGaussFits{sub}{interpSet}.fitX, behaviorCatGaussFits{sub}{interpSet}.fitY, 'Color', [colors(labels{sub}(interpSet),:) 0.5]);
        end 

        %scatters
        figure(72), hold on
        scatter(behaviorMldsGaussFits{sub}{interpSet}.mean, behaviorCatGaussFits{sub}{interpSet}.mean, [], colors(labels{sub}(interpSet),:), 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w')

        figure(73), hold on
        scatter(behaviorMldsGaussFits{sub}{interpSet}.std, behaviorCatGaussFits{sub}{interpSet}.std, [], colors(labels{sub}(interpSet),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w')
        
        %mlds vs categorical index
        if (behaviorMldsGaussFits{sub}{interpSet}.r2 > behavioral_r2_cutoff) & (mldsR2Vals(k) > mldsR2Cutoff)
            figure(74), hold on, scatter(mldsCatVals(k), behaviorMldsGaussFits{sub}{interpSet}.std, 'MarkerFaceColor', colors(labels{sub}(interpSet),:), 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w'),
            k = k+1;
        end
    end
end

figure(70), title('MLDS Curves'), xlabel('Interpolation number'), ylabel('Scale value'), xlim([1 6])
figure(71), title('Categorization Curves'), xlabel('Interpolation number'), ylabel('Frequency identified as object 2'), xlim([1 6])
figure(72), title('Mu values'), xlabel('Mlds Mu value'), ylabel('Categorization Mu value'), xlim([2.5 4.5]), ylim([2.5 4.5]), plot([2.5 4.5], [2.5 4.5], 'k-')
figure(73), title('Sigma values'), xlabel('Mlds sigma value'), ylabel('Categorization sigma value'), xlim([0 2]), ylim([0 2]), plot([0 6], [0 6], 'k-')
figure(74), title('Sigma vs categorical index'), xlabel('MLDS categorical Index'), ylabel('MLDS sigma Value'), xlim([-0.2 0.8]), ylim([0.3 1.8]);


save = 0;
if save
    for f = 70:74;                                                      
        figure(f);
        if sum(f == [70 71])
            drawPublishAxis('labelFontSize=8','figSize=[6, 5]','lineWidth=0.5', 'xtick=[1 2 3 4 5 6]');
        else
            drawPublishAxis('labelFontSize=8','figSize=[6, 5]','lineWidth=0.5'), end
            
        legend('off')
    end

    savepdf(figure(70),'~/Desktop/catFigs/comps/mldsCurves') 
    savepdf(figure(71),'~/Desktop/catFigs/comps/catCurves')  
    savepdf(figure(72),'~/Desktop/catFigs/comps/muValues') 
    savepdf(figure(73),'~/Desktop/catFigs/comps/sigmaValues') 
    savepdf(figure(74),'~/Desktop/catFigs/comps/sigmaVsCatIndex')  

    figure, hold on, for k = 1:8, scatter(k,k,'markerFaceColor', colors(k,:), 'MarkerEdgeColor', 'w'), end
end



%%
% final plot linking psychometrics and brain
%%
figure(96), hold on, xlim([0.5 4.5])
behavioral_r2_cutoff = 0.8;

m = []; mb = []; c = []; cb = []; usedColors = [];
for sub = 1:13
    for set = 1:length(behaviorCatGaussFits{sub})
        vvsCatGaussFits{sub}{set}.r2
        if (behaviorMldsGaussFits{sub}{set}.r2 > behavioral_r2_cutoff) & (behaviorCatGaussFits{sub}{set}.r2 > behavioral_r2_cutoff) & (vvsMldsGaussFits{sub}{set}.r2 > behavioral_r2_cutoff) & (vvsCatGaussFits{sub}{set}.r2 > behavioral_r2_cutoff)
            m = [m behaviorMldsGaussFits{sub}{set}.std];
            mb = [mb vvsMldsGaussFits{sub}{set}.std];
            c = [c behaviorCatGaussFits{sub}{set}.std];
            cb = [cb vvsCatGaussFits{sub}{set}.std];
            usedColors = [usedColors; colors(labels{sub}(set),:)];
        end
    end
end

scatter(repmat(1-0.1,1,length(m)), m, [], usedColors, 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w', 'XJitter', 'randn', 'XJitterWidth', 0.2);
scatter(repmat(2-0.1,1,length(m)), mb, [], usedColors, 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w', 'XJitter', 'randn', 'XJitterWidth', 0.2);
scatter(repmat(3-0.1,1,length(m)), c, [], usedColors, 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w', 'XJitter', 'randn', 'XJitterWidth', 0.2);
scatter(repmat(4-0.1,1,length(m)), cb, [], usedColors, 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w', 'XJitter', 'randn', 'XJitterWidth', 0.2);



ylabel('Cumulative Gaussian sigma')
xlim([0.5 4.5]), ylim([0 2])
title('')
xticks([1 2 3 4]);


scatter(1.1, mean(m), 72, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', 'w')
errorbar(1.1, mean(m), std(m), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
scatter(2.1, mean(mb), 72, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', 'w')
errorbar(2.1, mean(mb), std(mb), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
scatter(3.1, mean(c), 72, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', 'w')
errorbar(3.1, mean(c), std(c), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
scatter(4.1, mean(cb), 72, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', 'w')
errorbar(4.1, mean(cb), std(cb), 'LineStyle', 'none', 'Color', [0.5 0.5 0.5], 'CapSize', 0, 'LineWidth', 1)
%xticklabels({'MLDS', 'Brain MLDS', 'Categorization', 'Brain categorization'})


drawPublishAxis('labelFontSize=8','figSize=[11, 9]','lineWidth=0.5');
legend('off')
savepdf(figure(96),'~/Desktop/catFigs/comps/stdComps')
 



%% NUMBERS
getDistances(MVCRSMs, interpSets)





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% END OF SCRIPT %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
keyboard




%%%%%%%%%%%%%%%
%% get reliability/differences %%
%%%%%%%%%%%%%%%
function getDistances(RSMs, interpSets)
diagonal = [];
corner = [];
for sub = 1:length(RSMs)
    for interpSet = 1:length(RSMs{sub})/6
        diagonal = [diagonal RSMs{sub}(interpSets{interpSet}(1), interpSets{interpSet}(1))];
        corner = [corner RSMs{sub}(interpSets{interpSet}(1), interpSets{interpSet}(6))];
    end
end

sprintf('Same stimulus correlation - mean: %1.4f, std: %0.4f', mean(diagonal), std(diagonal))
sprintf('Different stimulus correlation - mean: %1.4f, std: %0.4f', mean(corner), std(corner))






%%%%%%%%%%%%%%%%%
%% get labels %%%
%%%%%%%%%%%%%%%%%%
function labels = getLabels;

labels{1} = [1 2 3 4]
labels{2} = [1 2 3 4];
labels{3} = [1 2 3 4];
labels{4} = [5 6 7];
labels{5} = [1 3 4];
labels{6} = [1 3 4];
labels{7} = [5 6 8];
labels{8} = [5 1 8];
labels{9} = [5 1 8];
labels{10} = [1 3 4];
labels{11} = [1 3 4];
labels{12} = [1 8 4];
labels{13} = [1 8 4];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plotPsychometricsFromRSMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [allColors, allLabels, vvsMldsGaussFits, vvsCatGaussFits] = plotPsychometricsFromRSMs(RSMs, RSMBoots, numSubs, colors, labels)
behavioral_r2_cutoff = 0.8;

interpSets = {[1:6], [7:12], [13:18], [19:24]};

x = []; y = []; trueY = []; allColors = []; allLabels = []
for sub = 1:numSubs
    for set = 1:length(RSMs{sub})/6

        %clear bootstrap classification
        bootDists = zeros(6,size(RSMBoots{sub},3));

        %interp idexes
        low = min(interpSets{set}); high = max(interpSets{set});
        dists = RSMs{sub}(low:high,high) - RSMs{sub}(low:high,low);

        %plot unnormalized mlds
        figure(80), hold on, plot([1 6], [0 0], '--k'), xlim([0.5, 6.5])
        scatter(1:6, dists, [], colors(labels{sub}(set),:), 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w')
        ylabel('Scale value'), xlabel('Interpolation number')
        title('Brain MLDS')

        %plot unbootstrapped classification
        figure(81), hold on, xlim([0.5, 6.5])
        scatter(1:6, (dists>0), [], colors(labels{sub}(set),:), 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w')
        ylabel('Percent closest to endpoint 2'), xlabel('Interpolation number')
        title('Brain Classification')
        x = [x 1:6]; y = [y dists>0]; trueY = [trueY dists];
        allColors = [allColors; repmat(colors(labels{sub}(set),:),6,1)];
        allLabels = [allLabels labels{sub}(set)];
        
        %normalize each curve individually
        figure(83), hold on, xlabel('Interpolation number'), ylabel('Relative distance to endpoints'); title('Brain MLDS'), xlim([1 6])
        normalizedDists = (dists - min(dists))/max(dists - min(dists));
        vvsMldsGaussFits{sub}{set} = fitCumulativeGaussian([1:6],normalizedDists);
        if (vvsMldsGaussFits{sub}{set}.r2 > behavioral_r2_cutoff);
            plot(vvsMldsGaussFits{sub}{set}.fitX, vvsMldsGaussFits{sub}{set}.fitY, 'color', [colors(labels{sub}(set),:) 0.5])
        end

        % do "classification" on the bootstraps
        figure(84), hold on, xlabel('Interpolation number'), ylabel('Percent classified object 2'); title('Brain classification')
        for boot = 1:size(RSMBoots{sub},3)
            bootDists(:,boot) = RSMBoots{sub}(low:high,high,boot) - RSMBoots{sub}(low:high,low,boot);
        end
        numClassified2 = sum(bootDists'>0)/1000;
        vvsCatGaussFits{sub}{set} = fitCumulativeGaussian([1:6], numClassified2);
        if (vvsCatGaussFits{sub}{set}.r2 > behavioral_r2_cutoff);
            plot(vvsCatGaussFits{sub}{set}.fitX, vvsCatGaussFits{sub}{set}.fitY, 'color', [colors(labels{sub}(set),:) 0.5])
        end
        xlim([1 6])
    end
end

%plot the medians
figure(80), scatter(1:6, mean(trueY,2), 60, [0 0 0], 'filled', 'MarkerEdgeColor', 'w')
figure(81), scatter(1:6, mean(y,2), 60, [0 0 0], 'filled', 'markerEdgeColor', 'w')


%normalize the correlation values between 0 and 1 so you can fit psychometrics
figure(82), hold on, title('Relative distance (normalized)')
ylabel('Corr(interp,6) - Corr(interp, 1) (normalized)'), xlabel('Interpolation point')

%scale the values to 0 1
scaledTrueY = trueY;
scaledTrueY = (scaledTrueY - min(mean(trueY,2)))/(max(mean(trueY,2)) - min(mean(trueY,2)));

%calculate threshold for calling something 1 vs 2
threshold = (0 - min(mean(trueY,2)))/(max(mean(trueY,2)) - min(mean(trueY,2)));
plot([1 6], [threshold threshold], 'Color', [0 0 0 0.5], 'LineStyle', '--')

%plot psychometrics
g = fitCumulativeGaussian(x,scaledTrueY(:)');
plot(g.fitX,g.fitY, 'k'), hold on
scatter(x, scaledTrueY(:), [], allColors, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
scatter(1:6, mean(scaledTrueY,2), 60, [0 0 0], 'filled')

g2 = fitCumulativeGaussian(x,y(:)');
figure(81), plot(g2.fitX,g2.fitY, 'k')

for interp = 1:6, choices(interp) = g.fitY(g.fitX == interp); end
brainCatTaskMatrix = 1- dist(choices);
[c r] = compareCatRSM(brainCatTaskMatrix, 1);






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plotHumanRSMS %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotHumanBrainRSMs(RSMs, interpSets, colors, labels, allColors)

totalRSMs = 0;
averageRSM = zeros(6);
totalStimuliRSMs = zeros(1,8);
stimuliRSMs = zeros(6,6,8);

for sub = 1:length(RSMs)
    for set = 1:length(RSMs{sub})/6

        rsm = RSMs{sub}(interpSets{set}, interpSets{set});

        averageRSM = averageRSM + rsm;

        stimuliRSMs(:,:,labels{sub}(set)) = stimuliRSMs(:,:,labels{sub}(set)) + rsm;

        totalRSMs = totalRSMs + 1;
        totalStimuliRSMs(labels{sub}(set)) = totalStimuliRSMs(labels{sub}(set)) + 1;

    end
end

%plot
figure, imagesc(averageRSM/totalRSMs), colormap('hot'), colorbar
figure, for k = 1:8, subplot(2,4,k), imagesc(stimuliRSMs(:,:,k)/totalStimuliRSMs(k)), colormap('hot'), end 




%%%%%%%%%%%%%%%%%%%%%%
%% plot human rsm reliability %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotHumanBrainReliability(RSMs, noise, c, offset)

%plot reliability
x = []; y = []; ynoise = []; ynoisestd = [];

for sub = 1:length(RSMs)
    x = [x repmat(sub,1,size(RSMs{sub},1))];
    y = [y diag(RSMs{sub})'];
    ynoise = [ynoise diag(mean(noise{sub},3))'];
    ynoisestd = [ynoisestd diag(std(noise{sub},0,3))'];
end
figure(60), hold on
scatter(x+offset, y, 12, c, 'filled', 'markerFaceAlpha', 0.5, 'markerEdgeColor', 'w')
plot([1 sub], [median(y) median(y)], 'color', c, 'LineStyle', '--')
xlabel('Subject')
ylabel('Split-half reliability of patterns')
ylim([0 1]), xlim([0.5 length(RSMs) + 0.5]);

figure(61), hold on
scatter(y, ynoise, 12, c, 'filled', 'markerFaceAlpha', 0.5, 'markerEdgeColor', 'w')










%%%%%%%%%%%%%%%%%%%%%%%%
%% compareCatRSM 
%%%%%%%%%%%%%%%%%%%%%%%%%
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







%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXTRA ANALYSES I DON'T END UP USING %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %% Do some comparisons between behavioral and brain data
% %compare brain cat/brain mlds means
% figure(90), hold on, xlabel('Brain MLDS Mean'), ylabel('Brain Cat Mean')
% x = []; y = [];
% for sub = 1:12
%     for set = 1:length(vvsMldsGaussFits{sub})
%         scatter(vvsMldsGaussFits{sub}{set}.mean, vvsCatGaussFits{sub}{set}.mean, [], colors(labels{sub}(set),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
%         x = [x vvsMldsGaussFits{sub}{set}.mean]; y = [y vvsCatGaussFits{sub}{set}.mean];
%     end
% end
% [r p] = corr(x',y')
% title(sprintf('Correlation: %0.2f, P-value: %0.3f', r, p));
% 
% 
% %compare mlds/brain mlds means
% figure(91), hold on, xlabel('Behavior MLDS Mean'), ylabel('Brain MLDS Mean')
% x = []; y = [];
% for sub = 1:12
%     for set = 1:length(behaviorMldsGaussFits{sub})
%         scatter(behaviorMldsGaussFits{sub}{set}.mean, vvsMldsGaussFits{sub}{set}.mean, [], colors(labels{sub}(set),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
%         x = [x behaviorMldsGaussFits{sub}{set}.mean]; y = [y vvsMldsGaussFits{sub}{set}.mean];
%     end
% end
% [r p] = corr(x',y')
% title(sprintf('Correlation: %0.2f, P-value: %0.3f', r, p));
% 
% 
% %compare behavior cat/brain cat mean
% figure(92), hold on, xlabel('Behavior Cat std'), ylabel('Brain Cat std')
% x = []; y = [];
% for sub = 1:12
%     for set = 1:length(behaviorCatGaussFits{sub})
%         scatter(behaviorCatGaussFits{sub}{set}.std, vvsCatGaussFits{sub}{set}.std, [], colors(labels{sub}(set),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
%         x = [x behaviorCatGaussFits{sub}{set}.std]; y = [y vvsCatGaussFits{sub}{set}.std];
%     end
% end
% [r p] = corr(x',y')
% title(sprintf('Correlation: %0.2f, P-value: %0.3f', r, p));
% xlim([0 2]), ylim([0 2]), plot([0 2], [0 2], 'k')
% 
% 
% 
% %compare behavior mlds/brain mlds std
% figure(93), hold on, xlabel('Behavior MLDS Std'), ylabel('Brain MLDS Std')
% x = []; y = [];
% for sub = 1:12
%     for set = 1:length(behaviorMldsGaussFits{sub})
%         scatter(behaviorMldsGaussFits{sub}{set}.std, vvsMldsGaussFits{sub}{set}.std, [], colors(labels{sub}(set),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
%         x = [x behaviorMldsGaussFits{sub}{set}.std]; y = [y vvsMldsGaussFits{sub}{set}.std];
%     end
% end
% [r p] = corr(x',y')
% title(sprintf('Correlation: %0.2f, P-value: %0.3f', r, p));
% xlim([0 2]), ylim([0 2]), plot([0 2], [0 2], 'k')
% 
% %compare behavior mlds/brain mlds std
% figure(94), hold on, xlabel('evc mlds std'), ylabel('vvs mlds std')
% x = []; y = [];
% for sub = 1:12
%     for set = 1:length(evcMldsGaussFits{sub})
%         scatter(evcMldsGaussFits{sub}{set}.std, vvsMldsGaussFits{sub}{set}.std, [], colors(labels{sub}(set),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
%         x = [x evcMldsGaussFits{sub}{set}.std]; y = [y vvsMldsGaussFits{sub}{set}.std];
%     end
% end
% [r p] = corr(x',y')
% title(sprintf('Correlation: %0.2f, P-value: %0.3f', r, p));
% 
% 
% %compare behavior mlds/brain cat mean
% figure(95), hold on, xlabel('Behavior cat Mean'), ylabel('Brain Cat Mean')
% x = []; y = [];
% for sub = 1:12
%     for set = 1:length(behaviorCatGaussFits{sub})
%         scatter(behaviorCatGaussFits{sub}{set}.mean, vvsCatGaussFits{sub}{set}.mean, [], colors(labels{sub}(set),:), 'filled','MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
%         x = [x behaviorCatGaussFits{sub}{set}.mean]; y = [y vvsCatGaussFits{sub}{set}.mean];
%     end
% end
% [r p] = corr(x',y')
% title(sprintf('Correlation: %0.2f, P-value: %0.3f', r, p));
% 




% 
% %% Do some analyses
% %brain vs mlds
% figure, hold on
% %plotMldsVals = []; plotBrainCatVals = [];
% for area = 1:3,
%     %subplot(1,3,area)
%     scatter(brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area) , mldsCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff)), 'filled')
%     plot([0 1], [0 1], '--k')
%     %plotMldsVals = [plotMldsVals mldsCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff))']; plotBrainCatVals = [plotBrainCatVals brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area)'];
% end
% xlabel('Brain categorical index'), ylabel('Mlds categorical index'), title('Brain and MLDS categorical indices')
% xlim([-0.2 1]), ylim([-0.2 1])
% legend({'Early', '', 'Middle', '', 'Late'})
% 
% 
% %brain vs categorization
% figure, hold on
% plotCatVals = []; plotBrainCatVals = [];
% for area = 1:3,
%     %subplot(1,3,area)
%     scatter(brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area) , brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), 4), 'filled')
%     plot([0 1], [0 1], '--k')
%     plotCatVals = [plotCatVals brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), 4)']; plotBrainCatVals = [plotBrainCatVals brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area)'];
% end
% xlabel('Brain categorical index'), ylabel('Cat task categorical index'), title('Brain and categorization categorical indices')
% xlim([-0.2 1]), ylim([-0.2 1])
% legend({'Early', '', 'Middle', '', 'Late'})
% 
% [C,P]=corrcoef(plotBrainCatVals,plotCatVals); plot(plotBrainCatVals,polyval(polyfit(plotBrainCatVals,plotCatVals,1),plotBrainCatVals),'b-');
% legend({'Early', '', 'Middle', '', 'Late', '', sprintf('Fit (r=%.2f, p=%.3f)',C(1,2),P(1,2))},'Location','best');
% 
% %mlds vs categorization
% figure, hold on
% scatter(mldsCatVals((mldsR2Vals > mldsR2Cutoff) & (catTaskR2Vals > brainR2Cutoff)), catTaskCatVals((mldsR2Vals > mldsR2Cutoff) & (catTaskR2Vals > brainR2Cutoff)), 'filled')
% xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
% xlabel('Mlds categorical index'), ylabel('Categorization task categorical index'), title('Mlds and Categorization task categorical indices')
% 
% 
% 
% %Whole brain vs mlds
% figure, hold on
% scatter(mldsCatVals((mldsR2Vals > mldsR2Cutoff) & (bigRoiR2Vals > brainR2Cutoff)), bigRoiCatVals((mldsR2Vals > mldsR2Cutoff) & (bigRoiR2Vals > brainR2Cutoff)), 'filled')
% xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
% xlabel('Mlds categorical index'), ylabel('Brain categorical index'), title('Whole brain categorical index vs mlds')
% 
% 
% 
% 










% %% Make the big plot
% figure, hold on, xlim([0.5 4.5]), ylim([-0.1 1.1])
% 
% 
% % Initialize storage
% filtered_NNCatVals = cell(size(NNR2Vals,3), 4);
% filtered_cornetCatVals = cell(1,4);
% filtered_brainCatVals = cell(1,3);
% filtered_mldsCatVals = [];
% filtered_coneCatVals = [];
% filtered_catTaskCatVals = [];
% 
% % Plot the models and store values
% for model = 1:size(NNR2Vals,3)
%     for area = 1:4
%         temp_vals = [];
%         for interp = 1:length(brainR2Vals)
%             if (NNR2Vals(interp, area, model) > NNR2Cutoff)
%                 val = NNCatVals(interp, area, model);
%                 scatter(area-0.1, val, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
%                 temp_vals(end+1) = val;
%             end
%         end
%         filtered_NNCatVals{model, area} = temp_vals;
%     end
% end
% 
% % Cornet model
% for area = 1:4
%     temp_vals = [];
%     for interp = 1:length(cornetR2Vals)
%         if (cornetR2Vals(interp, area) > NNR2Cutoff)
%             val = cornetCatVals(interp, area);
%             scatter(area-0.1, val, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
%             temp_vals(end+1) = val;
%         end
%     end
%     filtered_cornetCatVals{area} = temp_vals;
% end
% 
% % Brain data
% for area = 1:3
%     temp_vals = [];
%     for interp = 1:length(brainR2Vals)
%         if (brainR2Vals(interp, area) > brainR2Cutoff) && ...
%            (mldsR2Vals(interp) > mldsR2Cutoff) && ...
%            (catTaskR2Vals(interp) > catTaskR2Cutoff)
%             val = brainCatVals(interp, area);
%             scatter(area+0.1, val, 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
%             temp_vals(end+1) = val;
%         end
%     end
%     filtered_brainCatVals{area} = temp_vals;
% end
% 
% % MLDS
% for interp = 1:length(mldsR2Vals)
%     if (mldsR2Vals(interp) > mldsR2Cutoff)
%         val = mldsCatVals(interp);
%         scatter(4.1, val, 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
%         filtered_mldsCatVals(end+1) = val;
%     end
% end
% 
% % CONES
% for interp = 1:length(coneR2Vals)
%     if (coneR2Vals(interp) > coneR2Cutoff)
%         val = coneCatVals(interp);
%         scatter(0.8, val, 36, 'filled', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
%         filtered_coneCatVals(end+1) = val;
%     end
% end
% 
% 
% % Categorization task
% for interp = 1:length(catTaskR2Vals)
%     if (catTaskR2Vals(interp) > catTaskR2Cutoff)
%         val = catTaskCatVals(interp);
%         scatter(4.1, val, 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
%         filtered_catTaskCatVals(end+1) = val;
%     end
% end
% 
% % Axis labels
% xticks(1:4), xticklabels({'Early', 'Middle', 'Late', 'Behavior'}), xlabel('ROI/Layer')
% ylabel('Categorical index')
% title('Categorical influence on RSM')
% 
% %legend
% h1 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Neural network');
% h2 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Brain');
% h3 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'MLDS');
% h4 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Categorization task');
% h5 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Cone mosaic');
% legend([h1, h2, h3, h4, h5]);
% 
% 
% save = 0;
% if save
%     drawPublishAxis('labelFontSize=8','figSize=[8, 7]','lineWidth=0.5');
%     savepdf(figure(2),'~/Desktop/catFigs/comps/bigFigure') ;
% end

