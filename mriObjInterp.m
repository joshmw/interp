function mriObjInterp(varargin)
% interp_mriAnal.m
%
%  Takes the outputs of 'run_glmDenoiseInterp.m' (results) and does analyses.
%
%  Usage: interp_mriAnal(varargin)
%  Authors: Josh wilson
%  Date: 12/23/2024
%
%  Arguments:
%    reliabilityCutoff: split-half correlation of betas used to select voxels for analysis
%    r2cutoff: r2 cutoff (from glmsingle) used to select voxels for analysis
%    shuffleData: set to 1 to randomize condition labels in the very beginning (control measure)
%    zscorebetas: set to 1 to zscore the betas in each run
%    numBoots: number of bootstraps for split-half correlation analysis
%    nVoxelsNeeded: Number of voxels an ROI needs to have to be included in the analysis (# voxels that meet reliability cutoff)
%    mldsReps: number of mlds bootstraps. if set above 1, when averaging each condition, will average a random subset of trials instead of all
%    plotBig: plots things in their own graphs rather than as subplots
%    doROImlds: flag for if you want to do mlds in each ROI. takes a while to do this, so set to 0 to save time
%    truncateTrials: If you want to use less data. Should set as a fraction of m/n, where n is numScansInGLM and m is the number of scans you want to use for data. Use 1 for all.

%get args
getArgs(varargin, {'reliabilityCutoff=300', 'r2cutoff=-inf', 'shuffleData=0', 'zscorebetas=1', 'numBoots=250', 'nVoxelsNeeded=20', 'mldsReps=1', 'plotBig=0', 'doROImlds=1,'...
    'truncateTrials=(10/10)', 'clean=1'});


%% LOAD AND PROCESS THE DATA

% Task 1 is right visual field  so LEFT HEMISPHERE rois should be responsive
% Task 2 is LEFT visual field, so RIGHT HEMISPHERE rois shold be responsive

%list the subjects you want. right now, you have to use the big rois from combineGlasserRois -> createBigRoi if you want to combine. might add more options later but atm seems like a pain in the ass for little return.
subNumbers = {'s0605'};

%load each of the subjects data. You are getting a subject -> roi -> stimulus -> trial structure for each.
for sub = 1:length(subNumbers)
    %get paths for that subject
    dataPath = (strcat('~/data/interp/', subNumbers{sub}, '/betaFiles')); fileNames{1} = strcat(subNumbers{sub}, 'Task1BigROIs.mat'); fileNames{2} = strcat(subNumbers{sub}, 'Task2BigROIsasdfas.mat'); bigRois = 1;
    %get the data. The task and everything else should be the same.
    [task, data{sub}, roiNames, numBetasEachScan, numScansInGLM, numStimRepeats, numUsableVoxels] =... 
    processData(reliabilityCutoff, r2cutoff, shuffleData, zscorebetas, numBoots, nVoxelsNeeded, plotBig, truncateTrials, dataPath, fileNames);
end

%% COMBINE THE DATA
%if there is only 1 subject, you are combining it with itself. It needs to do this step to put the data in the right format.
allBetasCombinedFiltered = combineData(data);


%% MAKE DIFFERENT LARGE ROIS (EARLY, MIDDLE, VENTRAL) AND ALSO MAKE AVERAGED VERSIONS
stimNames = [1:24];
interpSets = {[1:6], [7:12], [13:18], [19:24]};
colors = cool(numBetasEachScan/2);

% early visual cortex ROI (v1 and v2)
earlyROIs = [1 3];
if bigRois, earlyROIs = 1; end
[allBetasEVCROI, allBetasEVCROIAveraged] = createUsableROIs(earlyROIs, allBetasCombinedFiltered, stimNames);


% mid-visual cortex ROIS (v3, v4, etc)
midROIs = [5 7 9 11 17 19];
if bigRois, midROIs = 3; end
[allBetasMVCROI, allBetasMVCROIAveraged] = createUsableROIs(midROIs, allBetasCombinedFiltered, stimNames);


% "ventral" visual ROIs (IT, FFA, LO, etc).
lateROIs = [13 15 21 23 25 27 29 31 33 39 41 45 47 49 51];
if bigRois, lateROIs = 5; end
[allBetasVVSROI, allBetasVVSROIAveraged] = createUsableROIs(lateROIs, allBetasCombinedFiltered, stimNames);


% "parietal" visual ROIs
parietalROIs = [63 65 69 71 73 81 83 85 87];
if bigRois, parietalROIs = 7; end
[allBetasParietalROI, allBetasParietalROIAveraged] = createUsableROIs(parietalROIs, allBetasCombinedFiltered, stimNames);


%big roi - all ROIs to combine.
[allBetasBigROI, allBetasBigROIAveraged] = createUsableROIs([earlyROIs midROIs lateROIs], allBetasCombinedFiltered, stimNames);



%% RELIABILITY OF EACH ROI PATTERN AS A NUMBER OF REPEATS
figure

%plot all voxels
subplot(2,2,1), hold on, title("All voxel reliability")
plotROIReliability(allBetasBigROI, numStimRepeats, stimNames)

%plot evc voxels
subplot(2,2,2), hold on, title("EVC voxel reliability")
plotROIReliability(allBetasEVCROI, numStimRepeats, stimNames)

%plot all voxels
subplot(2,2,3), hold on, title("MVC voxel reliability") 
plotROIReliability(allBetasMVCROI, numStimRepeats, stimNames)

%plot all voxels
subplot(2,2,4), hold on, title("VVS voxel reliability")
plotROIReliability(allBetasVVSROI, numStimRepeats, stimNames)



%% MAKE RSMS FOR ALL ROIS, EVC ROIS, MVC ROIS, AND VVS ROIS.

% RSM of the big ROI, combining all the small ROIS
figure, subplot(2,2,1)
[BigROIRSM, BigROIRSMstd] = calculateRSM(allBetasBigROI, stimNames, numStimRepeats, 0);
title('RSM: All ROIs combined. Correlation between averaged patterns of activity')

% RSM of the EVC ROI
subplot(2,2,2)
[EVCRSM, EVCRSMstd] = calculateRSM(allBetasEVCROI, stimNames, numStimRepeats, 0);
title('RSM: EVC voxels. Correlation between averaged patterns of activity')

% RSM of the mid ROI
subplot(2,2,3)
[MVCRSM, MVCRSMstd] = calculateRSM(allBetasMVCROI, stimNames, numStimRepeats, 0);
title('RSM: Mid-ventral voxels. Correlation between averaged patterns of activity')

% RSM of the VVC ROI
subplot(2,2,4),
[VVSRSM, VVSRSMstd] = calculateRSM(allBetasVVSROI, stimNames, numStimRepeats, 0);
title('RSM: Ventral voxels. Correlation between averaged patterns of activity')

% RSM of the Parietal ROI
figure
[ParietalRSM, ParietalRSMstd] = calculateRSM(allBetasParietalROI, stimNames, numStimRepeats, 0);
title('RSM: Parietal voxels. Correlation between averaged patterns of activity')
if clean, close, end



%% DO MLDS IN DIFFERENT REGIONS
if ~clean

%do EVC
figure
doMLDS(allBetasEVCROIAveraged, mldsReps, colors, task, interpSets, 0, 1)
sgtitle('EVC mlds')

%do MVC
figure
doMLDS(allBetasMVCROIAveraged, mldsReps, colors, task, interpSets, 0, 1)
sgtitle('Mid-level cortex Mlds')

%do ventral ROIs
figure
doMLDS(allBetasVVSROIAveraged, mldsReps, colors, task, interpSets, 0, 1)
sgtitle('VVS mlds')

%do parietal ROI
figure
doMLDS(allBetasParietalROIAveraged, mldsReps, colors, task, interpSets, 0, 1)
sgtitle('Parietal mlds')

%do big ROI
figure
doMLDS(allBetasBigROIAveraged, mldsReps, colors, task, interpSets, 0, 1)
sgtitle('All voxels mlds')

end
%% 2-WAY CLASSIFICATION BETWEEN INTERPOLATION SETS
if ~clean
%classify EVC
figure
doClassification(allBetasEVCROI, colors, task, numStimRepeats, interpSets)
sgtitle('Classification: EVC')

%classify MVC
figure
doClassification(allBetasMVCROI, colors, task, numStimRepeats, interpSets)
sgtitle('Classification: MVC')


%classify VVS
figure
doClassification(allBetasVVSROI, colors, task, numStimRepeats, interpSets)
sgtitle('Classification: VVC')

%classify Parietal
figure
doClassification(allBetasParietalROI, colors, task, numStimRepeats, interpSets)
sgtitle('Classification: Parietal')

end

%% AVERAGE THE RSMS OF THE INDIVIDUAL INTERPOLATIONS TOGETHER AND PLOT FOR EACH AREA

%first, average the RSMS
BigROIRSMAveraged = averageRSM(BigROIRSM, interpSets);
EVCRSMAveraged = averageRSM(EVCRSM, interpSets);
MVCRSMAveraged = averageRSM(MVCRSM, interpSets);
VVSRSMAveraged = averageRSM(VVSRSM, interpSets);
ParietalRSMAveraged = averageRSM(ParietalRSM, interpSets);


%plot them
figure

subplot(2,2,1), imagesc(BigROIRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: All voxels'), xlabel('Interpolation number'), ylabel('Interpolation number')

subplot(2,2,2), imagesc(EVCRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: EVC Voxels'), xlabel('Interpolation number'), ylabel('Interpolation number')

subplot(2,2,3), imagesc(MVCRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: Mid-level voxels'), xlabel('Object number'), ylabel('Interpolation number')

subplot(2,2,4), imagesc(VVSRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: VVS voxels'), xlabel('Object number'), ylabel('Interpolation number')

sgtitle('RSMs, averaged (post-normalization) over all interpolated stimulus sets')

%parietal
figure, imagesc(ParietalRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: Parietal voxels'), xlabel('Object number'), ylabel('Interpolation number')
if clean, close, end



%% DO MLDS ON THE AVERAGED RSMs AND PLOT IT:

figure, subplot(1,4,1), hold on
doMLDS(EVCRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('Early visual cortex')

subplot(1,4,2), hold on
doMLDS(MVCRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('Mid-level visual cortex')

subplot(1,4,3), hold on
doMLDS(VVSRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('Ventral visual cortex')

subplot(1,4,4), hold on
doMLDS(BigROIRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('Entire cortex')

sgtitle('MLDS for averaged interpolations in different areas')

%do parietal
figure, hold on
doMLDS(ParietalRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('Parietal RSM)')
if clean, close, end




%% LOAD IN AND PROCESS NEURAL NETWORK DATA
remapNSDfmri = 0;

if ~remapNSDfmri
    NNdata = load('NNcorrMatrices.mat');
    NNEVCRSM = (NNdata.layer_0 + NNdata.layer_1)/2; %V1 and V2 in the model
    NNMVCRSM = NNdata.layer_2; % V4 in the model
    NNVVSRSM = NNdata.layer_3; % IT in the model
    NNChoiceRSM = NNdata.layer_4; % 1000 way classification vector
elseif remapNSDfmri
    NNdata = load('NNcorrMatricesMapped.mat');
    NNEVCRSM = NNdata.layer_0; %V1 and V2 in the model
    NNMVCRSM = NNdata.layer_1; % V4 in the model
    NNVVSRSM = NNdata.layer_2; % IT in the model
    NNChoiceRSM = NNdata.layer_2; % 1000 way classification vector
end


NNEVCRSMAveraged = averageRSM(NNEVCRSM, interpSets);
NNMVCRSMAveraged = averageRSM(NNMVCRSM, interpSets);
NNVVSRSMAveraged = averageRSM(NNVVSRSM, interpSets);
NNChoiceRSMAveraged = averageRSM(NNChoiceRSM, interpSets);

%plot the averaged
figure

subplot(2,2,1), imagesc(NNChoiceRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: Classification layer'), xlabel('Interpolation number'), ylabel('Interpolation number')

subplot(2,2,2), imagesc(NNEVCRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: V1/V2 layers'), xlabel('Interpolation number'), ylabel('Interpolation number')

subplot(2,2,3), imagesc(NNMVCRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: V4 layer'), xlabel('Interpolation number'), ylabel('Interpolation number')

subplot(2,2,4), imagesc(NNVVSRSMAveraged), colormap(hot), colorbar, caxis([0 1])
title('RSM: IT layer'), xlabel('Interpolation number'), ylabel('Interpolation number')

sgtitle('NEURAL NETWORKS: RSMs, averaged (post-normalization) over all interpolated stimulus sets')


%% Do MLDS on the NN representations
figure, subplot(1,4,1), hold on
doMLDS(NNEVCRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('V1/V2 layers')

subplot(1,4,2), hold on
doMLDS(NNMVCRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('V4 layer')

subplot(1,4,3), hold on
doMLDS(NNVVSRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('IT layer')

subplot(1,4,4), hold on
doMLDS(NNChoiceRSMAveraged, mldsReps, colors, task, {interpSets{1}}, 1, 0)
title('Choice layer')

sgtitle('NEURAL NETOWORK MLDS for averaged interpolations in different areas')

if clean, close, end


%% PLOT EVIDENCE FOR CATEGORICAL VS LINEAR RSMS
figure(100), hold on

%plot human data
inputRSMs = {EVCRSMAveraged MVCRSMAveraged VVSRSMAveraged};
compareCatRSM(inputRSMs, [0 0 1], 72, 1, 0, 100)

% %plot unaveraged human trials
% for interpSet = 1:length(interpSets)
%     inputRSMs = {EVCRSM(interpSets{interpSet}, interpSets{interpSet}) MVCRSM(interpSets{interpSet}, interpSets{interpSet}) VVSRSM(interpSets{interpSet}, interpSets{interpSet})};
%     compareCatRSM(inputRSMs, [0 0 1], 36, .5, 1, 100)
% end

% %plot unaveraged CORnet trials
% for interpSet = 1:length(interpSets)
%     inputRSMs = {NNEVCRSM(interpSets{interpSet}, interpSets{interpSet}) NNMVCRSM(interpSets{interpSet}, interpSets{interpSet}) NNVVSRSM(interpSets{interpSet}, interpSets{interpSet}) NNChoiceRSM(interpSets{interpSet}, interpSets{interpSet})};
%     compareCatRSM(inputRSMs, [0 1 1], 36, .5, 1, 100)
% end
% 
% %plot averaged cornet
NNinputRSMs = {NNEVCRSMAveraged NNMVCRSMAveraged NNVVSRSMAveraged NNChoiceRSMAveraged};
compareCatRSM(NNinputRSMs, [0 1 1], 72, 1, 0, 100)

%plot controls
plotGaborWaveletRSMs(interpSets, 100);
plotNNearlyLayers(interpSets)



%% Do MDS, PCA on the averaged RSMs
figure, hold on
inputRSMs = {EVCRSMAveraged MVCRSMAveraged VVSRSMAveraged BigROIRSMAveraged};
doMDSPCA(inputRSMs,2)

figure, hold on
NNinputRSMs = {NNEVCRSMAveraged NNMVCRSMAveraged NNVVSRSMAveraged};
doMDSPCA(NNinputRSMs,2)
if clean, close, end


%% Test the summary metric describing how categorical a matrix is
%figure, hold on
%testCompareCatRSM(10, 'k', 36, .2)







keyboard
%% %%%%%%%%%%%% END OF SCRIPT %%%%%%%%%%%%%%%%%




function testCompareCatRSM(numBoots, color, size, alpha)

% create the 2 null hypothesis matrices
categoricalRSMCovar = [ones(3) zeros(3); zeros(3) ones(3)];
linearRSMCovar = max(0, 1 - 0.2 * abs((1:6)' - (1:6)));

%set parameters - boots, noise levels, categorical influence levels,
noiseLevels = [.02:.02:.2];
catLevels = [0 : 0.1 : 1];
colors = jet(10);

%simulate
for noiseLevel = 1:length(noiseLevels)
    subplot(2,5,noiseLevel); hold on
    plot([0 1], [0 1], 'k')
    ylim([-.2 1.2])
    title(sprintf('Noise level: %.2f', noiseLevels(noiseLevel)))
    xlabel('Ground truth categorical structure')
    ylabel('Recovered categorical structure')


    for catLevel = catLevels;
        for boot = 1:numBoots
            %get the RSM and compute the idealized matrices based on measured variance
            noiseMatrix = normrnd(0, noiseLevels(noiseLevel), 6, 6);
            inputRSM = catLevel * categoricalRSMCovar + (1-catLevel) * linearRSMCovar + (noiseMatrix + noiseMatrix')/2;
           
            linearRSM = linearRSMCovar;
            categoricalRSM = categoricalRSMCovar;
            
            %find betas that describe input of linear/categorical matrices to observed matrix
            [categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM);
            %scatter(inputNum, categoricalBeta - linearBeta, color,'filled'),
            scatter(catLevel, categoricalBeta / (categoricalBeta + linearBeta), size, 'filled', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', alpha),
        end
    end
end





%%%%%%%%%%%%%%%%
%% plotGaborWaveletRSMs
%%%%%%%%%%%%%%%%
function plotGaborWaveletRSMs(interpSets, figNum)

gaborData = load("gaborPyramidRSM.mat");
gaborRSM = gaborData.allRSMs/6;

%average
gaborRSMAveraged = averageRSM(gaborRSM, interpSets);
inputRSM = gaborRSMAveraged;

%create null matrices
categoricalRSM = [ones(3) zeros(3); zeros(3) ones(3)];
linearRSM = max(0, 1 - 0.2 * abs((1:6)' - (1:6)));

%find betas that describe input of linear/categorical matrices to observed matrix
[categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM, 1);

%catDiff = categoricalBeta - linearBeta;
catDiff = categoricalBeta / (categoricalBeta + linearBeta)

figure(figNum),
plot([1 4], [catDiff catDiff], 'r', 'DisplayName','Gabor wavelets')





%%%%%%%%%%%%%%%%%%%%%%%%
%% compareCatRSM
%%%%%%%%%%%%%%%%%%%%%%
function compareCatRSM(inputRSMs, color, size, alpha, normalize, figNum)

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
   
    %plot the r2 of the fits
    figure(figNum+1); hold on
    scatter(inputNum, r2, size, 'filled', 'MarkerFaceColor', color, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', alpha);
    xticks(1:4), xticklabels({'EVC', 'MVC', 'VVS', 'Choice layer'}); ylabel('r-squared of RSM fit'); title('R-squared of RSM fit'), ylim([0 1])


    %plot the percentrage of structure that is categorical
    figure(figNum)
    scatter(inputNum, categoricalBeta / (categoricalBeta + linearBeta), size, 'filled', 'MarkerFaceColor', color, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', alpha),

end
%figure(figNum), hold on

legendEntries = repmat({''}, 1, length(inputRSMs)*2+1);
legendEntries([2, 6]) = {'Human', 'CORnet'}; % Assign specific entries
legend(legendEntries)

xlim([0.5 4.5])%, ylim([0 0.5])
xticks(1:4), xticklabels({'EVC', 'MVC', 'VVS', 'Choice layer'})
ylabel('Structure attributable to catgeorical representation:  Bcat / (Bcat+Blinear)')
title('Categorical influence on RSM')





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
betas = fminsearch(objective, [.5, .5]) % Initial guesses for Betas

%results
figure, 
subplot(2,3,1), imagesc(inputRSM), colormap(hot), colorbar, caxis([0 1]), title('input')
subplot(2,3,2), imagesc(categoricalRSM * betas(1) + linearRSM * betas(2)), colormap(hot), colorbar, caxis([0 1]), title('fit RSM')
subplot(2,3,4), imagesc(categoricalRSM), colormap(hot), colorbar, caxis([0 1]), title('Categorical')
subplot(2,3,5), imagesc(linearRSM), colormap(hot), colorbar, caxis([0 1]), title('Linear')

%r-squared of the fit matrix
fitRSM = categoricalRSM * betas(1) + linearRSM * betas(2);
r2 = corr(fitRSM(:), inputRSM(:))^2;
sprintf('r-squared between fit and input matrix: %1.3f', r2)
subplot(2,3,3), hold on, scatter(inputRSM(~eye(size(fitRSM))), fitRSM(~eye(size(inputRSM))), 'k', 'filled')
xlabel('Input RSM value'), ylabel('Fit RSM value'), title(sprintf('r-squared: %1.3f', corr(fitRSM(~eye(size(fitRSM))), inputRSM(~eye(size(inputRSM))))^2))
plot([0 1], [0 1], 'k');

subplot(2,3,6), scatter(inputNum, r2);


clean=1; if clean, close, end


categoricalBeta = betas(1)/sum(betas);
linearBeta = betas(2)/sum(betas);






%%%%%%%%%%%%%%%%%%%%%%% 
%% plot early NN layers %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotNNearlyLayers(interpSets)

manyNNdata = load('NNcorrMatricesManyModels.mat');
models = fieldnames(manyNNdata);
colors = interp1([1 length(models)], [0 0.5 0; 0.31 0.78 0.47], 1:length(models));

for model = 1:length(models)
    NNdata = manyNNdata.(models{model});
    NNEVCRSM = (NNdata.layer_1); %V1 and V2 in the model
    NNMVCRSM = NNdata.layer_3; % V4 in the model
    NNVVSRSM = NNdata.layer_4; % IT in the model
    NNChoiceRSM = NNdata.layer_5; % 1000 way classification vector
    
    %average
    NNEVCRSMAveraged = averageRSM(NNEVCRSM, interpSets);
    NNMVCRSMAveraged = averageRSM(NNMVCRSM, interpSets);
    NNVVSRSMAveraged = averageRSM(NNVVSRSM, interpSets);
    NNChoiceRSMAveraged = averageRSM(NNChoiceRSM, interpSets);

    %plot RSMs
    figure(110)
    
    subplot(4,length(models),model + length(models) * 3), imagesc(NNChoiceRSMAveraged), colormap(hot), colorbar, caxis([0 1])
    title('Classification'),
    
    subplot(4,length(models),model), imagesc(NNEVCRSMAveraged), colormap(hot), colorbar, caxis([0 1])
    title(models(model)), 
    
    subplot(4,length(models),model + length(models)), imagesc(NNMVCRSMAveraged), colormap(hot), colorbar, caxis([0 1])
    title('Middle'), 
    
    subplot(4,length(models),model + length(models) * 2), imagesc(NNVVSRSMAveraged), colormap(hot), colorbar, caxis([0 1])
    title('Late'), 


    %plot summary statistics
    NNinputRSMs = {NNEVCRSMAveraged NNMVCRSMAveraged NNVVSRSMAveraged NNChoiceRSMAveraged};
    compareCatRSM(NNinputRSMs, colors(model,:), 36, 0.4, 0, 100)
end





%%%%%%%%%%%%%%%%%%%%%%%%%%
%% do MDS and PCA 
%%%%%%%%%%%%%%%%%%%%%%

function doMDSPCA(inputRSMs,numDimensions)

for inputNum = 1:length(inputRSMs)
    inputRSM = inputRSMs{inputNum}; num = inputNum-1;
    
    %make colors to plot
    colors = [repmat([1 0 0], 3, 1); %red
              repmat([0 0 0], 3, 1)]; %black
    
    %% do MDS first
    % create DSM and zero out diag (it's not 0 because you normalized and averaged earlier)
    dissimilarityMatrix = 1 - inputRSM;
    for i = 1:size(dissimilarityMatrix, 1), dissimilarityMatrix(i, i) = 0; end
    
    %set options
    MDSopts = statset('MaxIter', 100000, 'TolFun', 1e-6); % Increase iterations and tolerance
    
    % fit the MDS
    [Y, stress] = mdscale(dissimilarityMatrix, numDimensions, 'Options', MDSopts); stress
    
    % show it
    subplot(length(inputRSMs), 15, [1:6] + num*15), hold on, xlim([-1 1]), ylim([-1 1])
    scatter(Y(:,1), Y(:,2), 50, colors, 'filled')
    text(Y(:,1), Y(:,2), string(1:size(Y,1)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    title('MDS Embedding'), xlabel('Dimension 1'), ylabel('Dimension 2')
    
    
    %% now do PCA
    similarityMatrix = inputRSM;
    
    % perform PCA
    [coeff, score, latent] = pca(similarityMatrix);
    
    % plot the first principal component
    subplot(length(inputRSMs), 15, [7:12] + num*15);
    scatter(score(:,1), score(:,2), 50, colors, 'filled'); % 1D PCA
    text(score(:,1), score(:,2), string(1:size(score,1)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    xlim([-1 1]);
    
    %label
    title('PCA embedding'); xlabel('First Principal Component');
    
    %plot the egeinvalues
    subplot(length(inputRSMs), 15, [14:15] + num*15)
    scatter(1:length(latent),latent, 'filled','k')
    title('Scree'), xlabel('Eigenvector'), ylabel('Eigenvalue')
    
end




%%%%%%%%%%%%%%%%%%
%% createUsableROIs
%%%%%%%%%%%%%%%%%%

function [allBetasROI allBetasROIAveraged] = createUsableROIs(ROIs, allBetasCombinedFiltered, stimNames);

%combine the listed ROIs
for stim = 1:length(stimNames)
    allBetasROI{stim} = [];
    for roi = ROIs
        allBetasROI{stim} = [allBetasROI{stim}; allBetasCombinedFiltered{roi}{stim}];
    end
end

%average roi
for interp = 1:length(stimNames)
    allBetasROIAveraged{interp} = mean(allBetasROI{interp}, 2);
end





%%%%%%%%%%%%%
%% Calculate RSMs %%
%%%%%%%%%%%%%%%%%%%%
function [RSM, RSMstd] = calculateRSM(voxels, stimNames, numStimRepeats, shuffle)
    %create empty RSM bootstraps
    NumRSMBoots = 1000;
    RSMBoots = zeros(length(stimNames),length(stimNames),NumRSMBoots);
    %shuffle indices if you want
    if shuffle
        sprintf('!!!!YOU ARE SHUFFLING THE DATA!!!!!!')
        voxels = voxels(randperm(length(voxels)));
    end
    %do the bootstrapping
    for boot = 1:NumRSMBoots
        %create empty matrices and shuffle order for this loop
        averagedReps = zeros(length(stimNames), size(voxels{1},1));
        averagedRepsHalf1 = averagedReps; averagedRepsHalf2 = averagedReps;
        indices = randperm(numStimRepeats);
        
        %go through and add each averaged stimulus response to the empty average matrix
        for stim = 1:length(stimNames);
            %get the average of each of the halves of the stimuli (even or odds)
            averagedRepsHalf1(stim,:) = mean(voxels{stim}(:,indices(1:end/2)),2);
            averagedRepsHalf2(stim,:) = mean(voxels{stim}(:,indices(end/2+1:end)),2);
            %averagedRepsHalf1(stim,:) = mean(voxels{stim}(:,indices),2);
            %averagedRepsHalf2(stim,:) = mean(voxels{stim}(:,indices),2);

        end
        
        %rename everything according to earlier convention
        allBetasAveragedHalf1 = averagedRepsHalf1; allBetasAveragedHalf2 = averagedRepsHalf2;
        
        %make the RSM using halves.
        half1 = corr(allBetasAveragedHalf1', allBetasAveragedHalf2');
        half2 = corr(allBetasAveragedHalf2', allBetasAveragedHalf1');
        RSMBoots(:,:,boot) = (half1+half2)/2;
    end
    
    %average the bootstraps
    RSM = mean(RSMBoots,3);
    RSMstd = std(RSMBoots, 1, 3);
    
    %plot it
    imagesc(RSM),
    colormap(hot), colorbar, caxis([-1 1])
    xlabel('Object number'), ylabel('Object number')
    xticklabels(stimNames), xticks([1:length(stimNames)]), yticklabels(stimNames), yticks([1:length(stimNames)])
    drawLines






%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot roi reliability
%%%%%%%%%%%%%%%%%%%%%%%%%
function plotROIReliability(roi, numStimRepeats, stimNames)
    r = [];
    for numVoxels = 1:(numStimRepeats/2)
        corrVals = [];
        for stimulus = 1:length(stimNames)
            corrVals1stim = [];
            for repeat = 1:500
                subsetVoxels = randsample(1:numStimRepeats, numVoxels*2);
                subset = roi{stimulus}(:,subsetVoxels);
                averageHalf1 = mean(subset(:,1:numVoxels),2);
                averageHalf2 = mean(subset(:,(1+numVoxels):end),2);
                corrVal = corr(averageHalf1,averageHalf2);
                corrVals = [corrVals corrVal];
                corrVals1stim = [corrVals1stim corrVal];
            end
            scatter(numVoxels, mean(corrVals1stim), 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeAlpha', 0.1);
        end
        scatter(numVoxels, mean(corrVals), 60, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'w')
        r = [r mean(corrVals)]; y = r; x = 1:(numStimRepeats/2);
    end
    
    xlabel(('Number of scans in averages'))
    ylabel('Split half reliability')
    
    opt = fminsearch(@(p) sum((y - (p(1)*sqrt(x) + p(2))).^2), [1, 0]);
    a = opt(1); b = opt(2);
    y_fit = a * sqrt(x) + b;
    R2 = 1 - sum((y - y_fit).^2) / sum((y - mean(y)).^2);
    plot(y_fit)
    ylim([0 .8])




%%%%%%%%%%%%%%
%% averageRSMs
%%%%%%%%%%%%%%%
%average the RSMs of the different interpolations
function averagedRSM = averageRSM(rsm, interpSets)
flip = 0;
averagedRSM = zeros(length(interpSets{1}), length(interpSets{1}));

%concat different interpolations
catRSM = cat(3, rsm(interpSets{1},interpSets{1}), rsm(interpSets{2},interpSets{2}), rsm(interpSets{3},interpSets{3}), rsm(interpSets{4},interpSets{4}));

%z score
%normalizedRSM = (catRSM - mean(catRSM, [1 2])) ./ std(catRSM, 0, [1 2]);

%min/max normalization
minVal = min(catRSM, [], [1 2]); % Minimum value across each matrix
maxVal = max(catRSM, [], [1 2]); % Maximum value across each matrix
normalizedRSM = (catRSM - minVal) ./ (maxVal - minVal); % 


%average
averagedRSM = mean(normalizedRSM, 3);

%flip it - this ablates the order (1:3 vs 4:6)
if flip
    averagedRSM = (rot90(rot90(averagedRSM)) + averagedRSM)/2;
    sprintf('Flipping (180 rotation) and averaging the averaged RSM. This flips the order of the interpolations and averages induces rotational symmetry ')
end




%%%%%%%%%%%%
%% drawLines
%%%%%%%%%%%%
function drawLines
    hold on
    plot([.5 24.5], [6.5 6.5], 'k'), plot([.5 24.5], [12.5 12.5], 'k'), plot([.5 24.5], [18.5 18.5], 'k')
    plot([6.5 6.5], [.5 24.5], 'k'), plot([12.5 12.5], [.5 24.5], 'k'), plot([18.5 18.5], [.5 24.5], 'k'),




%%%%%%%%%%%%%%
%% do classification
%%%%%%%%%%%%%%%
% classification - train an SVM on the endpoints and see how it predicts the interpolations
function doClassification(classificationVoxels, colors, task, numStimRepeats, interpSets)
    for set = 1:length(interpSets)
        %define endpoints
        end1 = min(cell2mat(interpSets(set))); end2 = max(cell2mat(interpSets(set)));
        %get data for SVM
        data = [classificationVoxels{end1}'; classificationVoxels{end2}'];
        labels = [repmat(0,1,numStimRepeats) repmat(1,1,numStimRepeats)];
        %fit it
        numFolds = 5;
        svm = fitcsvm(data, labels, 'CrossVal', 'on', 'KFold', numFolds);
        
        %plot the results of different interpolation classifications
        subplot(1,4,set), hold on
        numEndpoint2 = [];
        for interp = cell2mat(interpSets(set));
            percentCat1 = [];
            for fold = 1:numFolds
                percentCat1(fold) = mean(svm.Trained{fold}.predict(classificationVoxels{interp}'));
            end
            numEndpoint2 = [numEndpoint2 mean(percentCat1)];
        end
        scatter(cell2mat(interpSets(1)),numEndpoint2,'filled','markerFaceColor',colors(max(interpSets{set}),:));
        %equality line
        plot([1 6], [0 1],'k','lineStyle','--')
        gaussFit = fitCumulativeGaussian(1:6,        numEndpoint2);
        plot(gaussFit.fitX,gaussFit.fitY,'color',colors(max(interpSets{set}),:))
        scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
        xlabel('Interpolation value')
        ylabel('Percent classified as endpoint 2')
        title(strcat(task{1}.stimfile.stimulus.interpNames{set}{1}, ' --> ', task{1}.stimfile.stimulus.interpNames{set}{2}))
    end



%%%%%%%%%%%%%%%%%%%%%
%% doMLDS %%%%%
%%%%%%%%%%%%%%%%%%
% do maximum likelihood distance scaling on the averaged representation with voxels from all ROIs
function [allPsi allSigma] = doMLDS(mldsVoxels, mldsReps, colors, task, interpSets, preComputeCorr, constrainBounds)    
    disp('Doing mlds...')
    %iterate through different interps
    for repititions = 1:mldsReps
    
        %average a different set of trials if bootstrapping
        if mldsReps > 1
            keyboard
            %if not bootstrapping, use the average of all presentations
        else
            allBetasBigROIAveragedMlds = mldsVoxels;
        end
    
        %do the mlds
        for set = 1:length(interpSets)
        
            %get the correlation matrix
            if ~preComputeCorr
                averagedInterps = cat(2, allBetasBigROIAveragedMlds{interpSets{set}});
                corMatrix = corr(averagedInterps); %cosine distance
            else
                corMatrix = mldsVoxels;
            end


            %simulate n draws of 4 images
            numSamples = 1000;
            ims = randi(6,4,numSamples);

            %take out repeat stimuli (e.g. 2,2 vs 1,4 // 2,4 vs 5,5) if you want
            for j  = numSamples:-1:1, if ims(1,j) == ims(2,j) | ims(3,j) == ims(4,j), ims(:,j) = []; end, end
            sprintf('Taking out mlds repeats - empirically didnt make a difference, but maybe check later if you are having trouble fitting...')

            responses = [];
            for trial = 1:size(ims,2)
                responses(trial) = corMatrix(ims(1,trial), ims(2,trial)) < corMatrix(ims(3,trial), ims(4,trial));
                j = ims(1,trial); k = ims(2,trial); l = ims(3,trial); m = ims(4,trial);
                if j == k | l == m | isequal(sort([j k]), sort([l m]));
                    responses(trial) = 2;
                end
            end
    
           % Set initial parameters
            psi = [0.2 0.4 0.6 0.8];
            sigma = 0.2;
            initialParams = [psi, sigma];
            
            % define bounds - there is a flag to constrain psis to 0-1 with the endpoints preset as 0 1
            if constrainBounds,
                lb = [0, 0, 0, 0, 0]; % Lower bounds for psi and sigma
                ub = [1, 1, 1, 1, 2]; % Upper bounds for psi and sigma
            else
                lb = [-inf, -inf, -inf, -inf, 0]; % Lower bounds for psi and sigma
                ub = [inf, inf, inf, inf, inf]; % Upper bounds for psi and sigma
            end

            % Options for fmincon
            options = optimoptions('fmincon', 'MaxIterations', 20000);
            
            % Define objective function
            objective = @(params) computeLoss(params, ims, responses);
            
            % Optimize using fmincon
            optimalParams = fmincon(objective, initialParams, [], [], [], [], lb, ub, [], options);
            
            % Normalize psi
            psi = [0, optimalParams(1:4), 1];
            psi = (psi - min(psi));
            psi = psi / max(psi);
                        
            %plot
            if length(interpSets) > 1, subplot(1,4,set), hold on, end
            scatter(1:6,psi, 'filled', 'MarkerFaceColor', colors(max(interpSets{set}),:))
            gaussFit = fitCumulativeGaussian(1:6, psi);
            PSE = gaussFit.mean;
            PSEs{set}(repititions) = PSE;
            plot(gaussFit.fitX,gaussFit.fitY,'color',colors(max(interpSets{set}),:))
            scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
            plot([1 6], [0 1],'k','lineStyle','--')
        
            %limits and label
            ylim([-0.05, 1.05]);
            xlim([1 6]);
            xlabel('Synthesized interpolation value')
            ylabel('Neural distance value')
        
            title(strcat(task{1}.stimfile.stimulus.interpNames{set}{1}, ' --> ', task{1}.stimfile.stimulus.interpNames{set}{2}))

            allPsi{set}{repititions} = psi;
            allSigmas{set}{repititions} = psi(5);
        end
    end





%%%%%%%%%%%%%%%%%%%%%%
%% shuffle data and labels %%
%%%%%%%%%%%%%%%%%%%%%%%
function [data, labels] = shuffleDataLabelOrder(data,labels)
idx = randperm(size(data, 1));
data = data(idx, :);
labels = labels(idx);




%%%%%%%%%%%%%%%%%%%
%% compute loss %%
%%%%%%%%%%%%%%%%%%%
function totalProb = computeLoss(params, ims, responses)
    % set interp values as psi and get differences between top and bottom pair
    psi = [0 params(1:4) 1];
    sigma = params(5);
    for interpVal = 1:6
        ims(ims == (interpVal)) = psi(interpVal);
    end
    diffs = abs(ims(1,:)-ims(2,:)) - abs(ims(3,:)-ims(4,:));
    % count up probability
    totalProb = 1000;
    for responseNum = 1:length(diffs)
        if responses(responseNum) == 1
            probResponse = -log(normcdf(diffs(responseNum),0,sigma));
            totalProb = totalProb + probResponse;
        elseif responses(responseNum) == 0
            probResponse = -log(1-normcdf(diffs(responseNum),0,sigma));
            totalProb = totalProb + probResponse;
        end
    end
    %disp(totalProb)




%%%%%%%%%%%%%%%%%%
%% processData
%%%%%%%%%%%%%%%%%%
function [task, allBetasCombinedFiltered, roiNames, numBetasEachScan, numScansInGLM, numStimRepeats, numUsableVoxelsByROI] = processData(...
    reliabilityCutoff, r2cutoff, shuffleData, zscorebetas, numBoots, nVoxelsNeeded, plotBig, ...
    truncateTrials, dataPath, fileNames);


%load the data
cd(dataPath);
task{1} = load(fileNames{1});

% little trick here - if you did foveal stimuli, duplicate the task. Then, later we will combine left and right hemi rois for the SAME task - so you
% are getting all the voxels you want anyway. This will make contra and ipso duplicates of eachother - which is ok, because we toss ipso later.
% if periphery (both sides), just load the other task (other side).
if ~isfield(task{1}, 'foveal'), task{1}.foveal = 0, end

if ~task{1}.foveal;
    task{2} = load(fileNames{2});
else
    task{2} = task{1};
end

% get parameters you need about the glm
numBetasEachScan = task{1}.numBetasEachScan;
numScansInGLM = task{1}.numScansInGLM;
numStimRepeats = task{1}.numStimRepeats;


%fix the stim names thing - remove the duplicate of the blanks (in this case, 1 8 15 22
for taskNum = 1:2,
    blanks = [1 8 15 22]; for blank = blanks, task{taskNum}.stimNames(blank) = []; end
end

%swap the roiNums for task 2 so that we can do both hemispheres together
odds = mod(task{2}.whichROI,2) == 1;
evens = ~odds;
task{2}.whichROI(evens) = task{2}.whichROI(evens) - 1;
task{2}.whichROI(odds) = task{2}.whichROI(odds) + 1;

%rename rois to contra and ipso so we can combine
roiNames = task{1}.roiNames;
for roi = 1:length(roiNames)
    if roiNames{roi}(1) == 'l'
        string = strcat('contra ', roiNames{roi}(2:end));
        roiNames{roi} = strrep(string,'_',' ');
    else
        string = strcat('ipsi ', roiNames{roi}(2:end));
        roiNames{roi} = strrep(string,'_',' ');
    end
end


%% if you want to truncate the data (fewer repeats), do so here by setting truncateTrials <1
%note that we calculated reliability before truncating. Might want to switch.
numStimRepeats = numStimRepeats * truncateTrials;
numScansInGLM = numScansInGLM * truncateTrials;
for taskNum = 1:2,
    task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd = task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd(:,:,:,1:(numBetasEachScan*numScansInGLM));
    task{taskNum}.trial_conditions = task{taskNum}.trial_conditions(1:(numBetasEachScan*numScansInGLM));
    %you can make it backwards if you uncomment these. the proportion included will flip (.7 truncate -> include last .3 of data). DO NOT UNCOMMENT if not truncating.
    %task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd = task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd(:,:,:,((numBetasEachScan*numScansInGLM)+1):end);
    %task{taskNum}.trial_conditions = task{taskNum}.trial_conditions((numBetasEachScan*numScansInGLM+1):end);
end


%% ZSCORE THE DATA AND GET AVERAGE AMPLITUDE VALUE
for taskNum = 1:2
    %get betas
    task{taskNum}.amplitudes = squeeze(task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd);

    %zscore if you want
    if zscorebetas
        for scan = 1:numScansInGLM
            %zscore by scan
            task{taskNum}.amplitudes(:,(1 + ((scan-1)*numBetasEachScan)):(scan*numBetasEachScan)) = zscore(task{taskNum}.amplitudes(:,(1 + ((scan-1)*numBetasEachScan)):(scan*numBetasEachScan)), 0, 2);
        end
    end

    %average
    for stim = 1:length(task{taskNum}.stimNames)
        task{taskNum}.averaged_amplitudes(:,stim) = nanmean(task{taskNum}.amplitudes(:,(task{taskNum}.trial_conditions == stim)),2);
    end
end


%% CALCULATE THE RELIABILITY OF INDIVIDUAL VOXELS (CONSISTENCY OF BETA VALUES ACROSS PRESENTATIONS)
for taskNum = 1:2;
    task{taskNum}.betas = task{taskNum}.amplitudes;
    wb = waitbar(0, 'Starting reliability bootstraps');
    for boot = 1:numBoots
        condition_amps = {};
        split1_amps = {};
        split2_amps = {};
        %
        for cond = 1:length(task{taskNum}.stimNames);
        %for cond = [1 6 7 12 13 18 19 24]; % if you want, could only include voxels that are stable for just the endpoint objects...
            condition_amps{cond} = task{taskNum}.betas(:,task{taskNum}.trial_conditions==cond);
            nTrials = size(condition_amps{cond},2);
            
            trialorder = randperm(nTrials);
            split1_amps{end+1} = condition_amps{cond}(:,trialorder(1:floor(nTrials/2)));
            split2_amps{end+1} = condition_amps{cond}(:,trialorder((floor(nTrials/2)+1):end));
        end
        
        s1a = cellfun(@(x) mean(x,2), split1_amps, 'un', 0);
        s2a = cellfun(@(x) mean(x,2), split2_amps, 'un', 0);
        
        s1a = cat(2, s1a{:});
        s2a = cat(2, s2a{:});
        reliability(:,boot) = diag(corr(s1a', s2a'));
        waitbar(boot/numBoots, wb, sprintf('Reliability bootstraps (half: %i): %d %%', taskNum, floor(boot/numBoots*100)))
    end
    close(wb)
    task{taskNum}.allBootReliability = reliability;
    task{taskNum}.reliability = mean(reliability,2);
end


%% PARTITION BY ROI AND FILTER BY VOXEL RELIABILITY
for taskNum = 1:2
    for roi = 1:length(task{taskNum}.roiNames);
        %this is just for plotting the betas. just set to 0. They are z-scored anyway, so this chart will always be mean 0.
        if reliabilityCutoff > 1;
            newReliabilityCutoff = 0;
        else
            newReliabilityCutoff = reliabilityCutoff;
        end
        %face responses
        task{taskNum}.averagedBetas{roi} = task{taskNum}.averaged_amplitudes((task{taskNum}.whichROI == roi)' & (task{taskNum}.reliability > newReliabilityCutoff) & (task{taskNum}.glmR2_FIT_HRF_GLMdenoise_RR > r2cutoff),:);
    end
end

% combine hemispheres in averaged betas
for roi = 1:length(roiNames)
    averagedBetas{roi} = [task{1}.averagedBetas{roi}; task{2}.averagedBetas{roi}];
end



%% PLOT SPLIT HALF RELIABILITY, R2, AND BETA AMPLITUDES BY ROI
%plot all the reliabilities with the average
figure, 
if ~plotBig, subplot(3,1,1), hold on, else, figure, hold on, end
for roi = 1:length(roiNames)
    %add the two halves
    reliability = [task{1}.reliability(task{1}.whichROI == roi); task{2}.reliability(task{2}.whichROI == roi)];
    %plot all then average
    scatter(repmat(roi,length(reliability),1), reliability,'k','filled','MarkerFaceAlpha',.05)
    scatter(roi, median(reliability),72,'r','filled','MarkerEdgeColor','w')
end

%label
hline(0,':k')
xticks(1:length(roiNames))
xticklabels(roiNames)
xlabel('ROI name'), ylabel('Voxel reliability')
ylim([-.5 .5])
title('Correlation between first half and second half Betas')

%plot the R2 with the average
if ~plotBig, subplot(3,1,2), hold on, else, figure, hold on, end
for roi = 1:length(roiNames)
    r2 = [task{1}.models.FIT_HRF_GLMdenoise_RR.R2(task{1}.whichROI == roi); task{2}.models.FIT_HRF_GLMdenoise_RR.R2(task{2}.whichROI == roi)];
    scatter(repmat(roi,length(r2),1), r2,'k','filled','MarkerFaceAlpha',.05)
    scatter(roi, median(r2),72,'r','filled','MarkerEdgeColor','w')
end

xticks(1:length(roiNames))
xticklabels(roiNames)
xlabel('ROI name'), ylabel('Voxel R-squared')
ylim([-40 40])
hline(0,':k')
title('R-Squared of voxels')

%plot the betas for each ROI
if ~plotBig, subplot(3,1,3), hold on, else, figure, hold on, end
for roi = 1:length(roiNames)
    roiBetas = [task{1}.averagedBetas{roi}(:); task{2}.averagedBetas{roi}(:)];
    scatter(repmat(roi,length(roiBetas),1)', roiBetas,'k','filled','MarkerFaceAlpha',.01);
    scatter(roi, median(roiBetas),72,'r','filled','MarkerEdgeColor','w');
end

xticks(1:length(roiNames))
xticklabels(roiNames)
xlabel('ROI name'), ylabel('Beta Amplitude')
ylim([-2 2])
hline(0,':k')
title('Beta weights')


%% shuffle the data if you want to - this is a flag you can set for control analyses. randomizes the category labels of all the stimuli
if shuffleData
    task{1}.trial_conditions = task{1}.trial_conditions(randperm(length(task{1}.trial_conditions)));
    task{2}.trial_conditions = task{2}.trial_conditions(randperm(length(task{2}.trial_conditions)));
end

%% PROCESS DATA - FILTER BY RELIABILITY, COMBINE HEMIS
%first, get all of the betas for individual stim types for rois
for taskNum = 1:2
    for roi = 1:length(roiNames);
        % if the reliability cutoff is >1, then pick that number of voxels for each roi.
        if reliabilityCutoff > 1
            sortedReliability = sort(task{taskNum}.reliability((task{taskNum}.whichROI == roi)'), 'descend');
            newReliabilityCutoff = sortedReliability(reliabilityCutoff+1);
        else
            newReliabilityCutoff = reliabilityCutoff
        end
        %now sort by reliability, roi, etc
        for condition = 1:length(task{1}.stimNames);
        %get betas for individual trials, filtering by reliability
            allBetas{taskNum}{roi}{condition} = task{taskNum}.betas((task{taskNum}.whichROI == roi)' & (task{taskNum}.reliability > newReliabilityCutoff) & (task{taskNum}.glmR2_FIT_HRF_GLMdenoise_RR > r2cutoff),task{taskNum}.trial_conditions==condition);
        end
    end
end

%combine all of the betas from each hemisphere
for roi = 1:length(roiNames)
    for stimName = 1:length(task{1}.stimNames)
        allBetasCombinedFiltered{roi}{stimName} = [allBetas{1}{roi}{stimName}; allBetas{2}{roi}{stimName}];
    end
end


%% COUNT NUMBER OF VOXELS FOR SUB = 1:2:LENGTH(ROINAMES);
for sub = 1:2:length(roiNames);
    singleTrials = cat(2, allBetasCombinedFiltered{sub}{1:length(task{1}.stimNames)});
    if size(singleTrials,1) > 2
        %save the number of voxels you have
        numUsableVoxelsByROI(sub) = size(singleTrials,1);
    end
end

%make roisToCombine
roisToCombine = 1:length(task{1}.roiNames); roisToCombine = roisToCombine(numUsableVoxelsByROI > nVoxelsNeeded); roisToCombine = roisToCombine(mod(roisToCombine,2)==1);
singleTrialCorrs = {};

%go through each ROI, plot the average RSM for INDIVIDUAL presentations of the same stimuli
%this should be read as how consistent the ROI is
figure

for roi = roisToCombine;
    singleTrialCorrelations = zeros(numStimRepeats);
    for stim = 1:length(task{1}.stimNames);
        singleTrialCorrelations = singleTrialCorrelations + corr(allBetasCombinedFiltered{roi}{stim})/length(task{1}.stimNames);
        singleTrialCorrs{roi}{stim} = corr(allBetasCombinedFiltered{roi}{stim});
    end

    subplot(4,ceil(length(roisToCombine)/4),find(roisToCombine == roi)), hold on
    imagesc(singleTrialCorrelations),
    colorbar, caxis([-.2 .2])
    title(roiNames(roi))
end
 
sgtitle('Reliability by area (individual trial correlations, averaged over all stimuli)')
close


% plot reliability of different patterns of stimuli
%stimNames = task{1}.stimfile.stimulus.objNames;

figure, hold on
colors = cool(numBetasEachScan/2);
for stim = 1:length(task{1}.stimNames)
    y = [];
    %subplot(4,ceil(length(stimNames)/4),stim), hold on

    for roi = 1:length(roisToCombine);
        stimSingleTrialCorrs = singleTrialCorrs{roisToCombine(roi)}{stim};
        %scatter(roi, median(stimSingleTrialCorrs(stimSingleTrialCorrs<1)));
        y = [y median(stimSingleTrialCorrs(stimSingleTrialCorrs<1))];
    end

    scatter(1:length(roisToCombine), y, 100, 'MarkerFaceColor', colors(stim,:), 'MarkerEdgeColor', 'w');


end

hline(0)
xticks(1:length(roisToCombine)); xticklabels(roiNames(roisToCombine));
title('Median single-trial correlation by area (by stimulus)');
%legend(stimNames);




%%%%%%%%%%%%%%%%%
%% combineData %%
%%%%%%%%%%%%%%%%%
function allBetasCombinedFiltered = combineData(data);

%get first subjects all betas combined filtered
allBetasCombinedFiltered = data{1};

%go through addition subjects, if any
for subject = 2:length(data)
    for roi = 1:length(data{subject})
        for stimulus = 1:length(data{subject}{roi})
            allBetasCombinedFiltered{roi}{stimulus} = [allBetasCombinedFiltered{roi}{stimulus}; data{subject}{roi}{stimulus}];
        end
    end
end









% %% look at the RSMs in individual ROIs
% figure
% for roi = roisToCombine;
%     averagedReps = zeros(length(stimNames), numUsableVoxelsByROI(roi));
%     for stim = 1:length(task{1}.stimNames);
%         averageStim =  mean(allBetasCombinedFiltered{roi}{stim},2);
%         averagedReps(stim,:) = averageStim;
%     end
% 
%     subplot(4,ceil(length(roisToCombine)/4),find(roisToCombine == roi)), hold on
%     RSM = corr(averagedReps');
%     imagesc(RSM),
%     colorbar, caxis([-1 1])
%     title(roiNames(roi))
% end
% 
% sgtitle('RSMs for all simuli in different areas')


% %% do mlds on UNAVERAGED representations
% figure
% 
% disp('Doing mlds - takes a minute or so.')
% %iterate through different interps
% for repititions = 1:mldsReps
% 
%     %get single trial correlations
%     for set = 1:length(interpSets)      
%         corMatrix = [];
%         for interp = interpSets{set}
%             for interp2 = interpSets{set}
%                 allCors = corr(allBetasBigROI{interp}, allBetasBigROI{interp2});
%                 corMatrix(interp,interp2,:) = allCors(tril(allCors,-1) ~= 0);
%             end
%         end
%         %simulate n draws of 4 images
%         numSamples = 5000;
%         ims = randi(6,4,numSamples);
%         %calculate which pair has a higher correlation
%         responses = [];
%         for trial = 1:numSamples
%             responses(trial) = corMatrix(ims(1,trial), ims(2,trial), randi(size(corMatrix,3))) < corMatrix(ims(3,trial), ims(4,trial), randi(size(corMatrix,3)));
%             j = ims(1,trial); k = ims(2,trial); l = ims(3,trial); m = ims(4,trial);
%             if j == k | l == m | isequal(sort([j k]), sort([l m]));
%                 responses(trial) = 2;
%             end
%         end
% 
%        % Set initial parameters
%             psi = [0.2 0.4 0.6 0.8];
%             sigma = 0.2;
%             initialParams = [psi, sigma];
%             
%             % define bounds - there is a flag to constrain psis to 0-1 with the endpoints preset as 0 1
%             if constrainBounds,
%                 lb = [0, 0, 0, 0, 0]; % Lower bounds for psi and sigma
%                 ub = [1, 1, 1, 1, 2]; % Upper bounds for psi and sigma
%             else
%                 lb = [-inf, -inf, -inf, -inf, 0]; % Lower bounds for psi and sigma
%                 ub = [inf, inf, inf, inf, inf]; % Upper bounds for psi and sigma
%             end
% 
%             % Options for fmincon
%             options = optimoptions('fmincon', 'MaxIterations', 20000);
%             
%             % Define objective function
%             objective = @(params) computeLoss(params, ims, responses);
%             
%             % Optimize using fmincon
%             optimalParams = fmincon(objective, initialParams, [], [], [], [], lb, ub, [], options);
%         
%             % Normalize psi
%             psi = [0, optimalParams(1:4), 1];
%             psi = (psi - min(psi));
%             psi = psi / max(psi);
%             
%         %plot
%         subplot(1,4,set), hold on
%         scatter(1:6,psi, 'filled')
%         gaussFit = fitCumulativeGaussian(1:6, psi);
%         PSE = gaussFit.mean;
%         PSEs{set}(repititions) = PSE;
%         plot(gaussFit.fitX,gaussFit.fitY)
%         scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
%         plot([1 6], [0 1],'k','lineStyle','--')
%     
%         %limits and label
%         ylim([-0.05, 1.05]);
%         xlim([1 6]);
%         xlabel('Synthesized interpolation value')
%         ylabel('Neural interpolation value')
%         if set == 1; title('Grass to leaves mlds'); elseif set == 2, title('Lemons to bananas mlds'), end
%     
%     end
% end
% sgtitle(sprintf('MLDS, all %i voxels in all ROIs', sum(numUsableVoxelsByROI)))


