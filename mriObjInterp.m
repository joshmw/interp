function interp_mriAnal(varargin)
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
%    stdCutoff: remove voxels whos betas in any single condition have a standard deviation higher than this value
%    shuffleData: set to 1 to randomize condition labels in the very beginning (control measure)
%    zscorebetas: set to 1 to zscore the betas in each run
%    numBoots: number of bootstraps for split-half correlation analysis
%    nVoxelsNeeded: Number of voxels an ROI needs to have to be included in the analysis (# voxels that meet reliability cutoff)
%    showAvgMDS/showDistancesSingleTrial: flags to show analyses I don't think are relevant
%    mldsReps: number of mlds bootstraps. if set above 1, when averaging each condition, will average a random subset of trials instead of all
%    plotBig: plots things in their own graphs rather than as subplots
%    doROImlds: flag for if you want to do mlds in each ROI. takes a while to do this, so set to 0 to save time
%    numBetasEachScan: Used for zscoring. Should be the number of betas you get from each scan from glmSingle (e.g. 12 conds x 4 repeats = 48)
%    numScansInGLM: Used for zscoring. Needed to zscore over individual scans.
%    numStimRepeats: Number of repeats of each stimulus (how many times shown in scanner)
%    truncateTrials: If you want to use less data. Should set as a fraction of m/n, where n is numScansInGLM and m is the number of scans you want to use for data. Use 1 for all.



%% Load the data
%get args
getArgs(varargin, {'reliabilityCutoff=.4', 'r2cutoff=0', 'stdCutoff=100', 'shuffleData=0', 'zscorebetas=1', 'numBoots=1000', 'nVoxelsNeeded=20' 'showAvgMDS=50', 'showDistancesSingleTrials=0', 'mldsReps=1', 'plotBig=0', 'doROImlds=1,'...
    'numBetasEachScan=48', 'numScansInGLM=15', 'numStimRepeats=30','truncateTrials=(10/10)'});

% Task 1 is right visual field  so LEFT HEMISPHERE rois should be responsive
% Task 2 is LEFT visual field, so RIGHT HEMISPHERE rois shold be responsive
%load the data

cd('~/data/interp/s0603/')
task{1} = load('s0603Task1.mat');
task{2} = load('s0603Task2.mat');


%fix the stim names thing - remove the duplicate of the blanks (in this case, 1 8 15 22
for taskNum = 1:2,
    blanks = [1 8 15 22]; for blank = blanks, task{taskNum}.stimNames(blank) = []; end
end

%swap the roiNums for task 2 so that we can do both hemispheres together
odds = mod(task{2}.whichROI,2) == 1;
evens = ~odds;
task{2}.whichROI(evens) = task{2}.whichROI(evens) - 1;
task{2}.whichROI(odds) = task{2}.whichROI(odds) + 1;

%shuffle the data if you want to - this is a flag you can set for control analyses. randomizes the category labels of all the stimuli
if shuffleData
    task{1}.trial_conditions = task{1}.trial_conditions(randperm(length(task{1}.trial_conditions)));
    task{2}.trial_conditions = task{2}.trial_conditions(randperm(length(task{2}.trial_conditions)));
end

%rename rois to contra and ipso so we can combine
roiNames = task{1}.roiNames;
for roi = 1:length(roiNames)
    if roiNames{roi}(1) == 'l'
        string = strcat('contra ', roiNames{roi}(2:end));
        roiNames{roi} = strrep(string,'_',' ');
    else
        string = strcat('ipso ', roiNames{roi}(2:end));
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


%% zscore and average some
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


%% calculate the reliability of individual voxels (consistancy of beta values across presentations)
for taskNum = 1:2;
    task{taskNum}.betas = task{taskNum}.amplitudes;
    wb = waitbar(0, 'Starting reliability bootstraps');
    for boot = 1:numBoots
        condition_amps = {};
        split1_amps = {};
        split2_amps = {};
        %
        for cond = 1:length(task{taskNum}.stimNames);
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




%% partition by roi and filter by voxel reliability
for taskNum = 1:2
    for roi = 1:length(task{taskNum}.roiNames);
        %face responses
        task{taskNum}.averagedBetas{roi} = task{taskNum}.averaged_amplitudes((task{taskNum}.whichROI == roi)' & (task{taskNum}.reliability > reliabilityCutoff) & (task{taskNum}.glmR2_FIT_HRF_GLMdenoise_RR > r2cutoff),:);
    end
end

% combine hemispheres in averaged betas
for roi = 1:length(roiNames)
    averagedBetas{roi} = [task{1}.averagedBetas{roi}; task{2}.averagedBetas{roi}];
end



%% plot split half reliability, R2, and beta amplitudes by ROI
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


%% process data - filter by reliability, combined hemis
%first, get all of the betas for individual stim types for rois
for taskNum = 1:2
    for roi = 1:length(roiNames);
        for condition = 1:length(task{1}.stimNames);
        %get betas for individual trials, filtering by reliability
            allBetas{taskNum}{roi}{condition} = task{taskNum}.betas((task{taskNum}.whichROI == roi)' & (task{taskNum}.reliability > reliabilityCutoff) & (task{taskNum}.glmR2_FIT_HRF_GLMdenoise_RR > r2cutoff),task{taskNum}.trial_conditions==condition);
        end
    end
end

%combined all of the betas from each hemisphere
for roi = 1:length(roiNames)
    for stimName = 1:length(task{1}.stimNames)
        allBetasCombined{roi}{stimName} = [allBetas{1}{roi}{stimName}; allBetas{2}{roi}{stimName}];
    end
end

%determine the std of each voxel's response to repeated presentations, averaged over all stimuli types
for roi = 1:length(roiNames)
    avgStd = 0;
    for stim = 1:length(task{1}.stimNames)
        avgStd = avgStd + std(allBetasCombined{roi}{stim}');
    end
    voxelStd{roi} = avgStd/length(task{1}.stimNames);
end

%filter out voxels that are really noisy (high std in betas for repeats)
%YOU DON'T NEED TO DO THIS IS Z-SCORING! which you should be doing. just set stdCutoff to high (>5) and it wont do anything.
for roi = 1:length(roiNames)
    for stim = 1:length(task{1}.stimNames)
        allBetasCombinedFiltered{roi}{stim} = allBetasCombined{roi}{stim}(voxelStd{roi} < stdCutoff,:);
    end
end



%% count number of voxelsfor sub = 1:2:length(roiNames);
for sub = 1:2:length(roiNames);
    singleTrials = cat(2, allBetasCombinedFiltered{sub}{1:length(task{1}.stimNames)});
    if size(singleTrials,1) > 2
        %save the number of voxels you have
        numUsableVoxelsByROI(sub) = size(singleTrials,1);
    end
end



%% look at the reliability of the patterns for single trials in different areas
figure,

%make roisToCombine
roisToCombine = 1:length(task{1}.roiNames); roisToCombine = roisToCombine(numUsableVoxelsByROI > nVoxelsNeeded); roisToCombine = roisToCombine(mod(roisToCombine,2)==1);
singleTrialCorrs = {};

%go through each ROI, plot the average RSM for INDIVIDUAL presentations of the same stimuli
%this should be read as how consistent the ROI is
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



%% look at reliability of patterns for different stimuli
%stimNames = task{1}.stimfile.stimulus.objNames;
stimNames = [1:24];

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



%% look at the RSMs in individual ROIs
figure
for roi = roisToCombine;
    averagedReps = zeros(length(stimNames), numUsableVoxelsByROI(roi));
    for stim = 1:length(task{1}.stimNames);
        averageStim =  mean(allBetasCombinedFiltered{roi}{stim},2);
        averagedReps(stim,:) = averageStim;
    end

    subplot(4,ceil(length(roisToCombine)/4),find(roisToCombine == roi)), hold on
    RSM = corr(averagedReps');
    imagesc(RSM),
    colorbar, caxis([-1 1])
    title(roiNames(roi))
end

sgtitle('RSMs for all simuli in different areas')


%% classification in individual ROIs - train an SVM on the n-way classification task (number of stimuli types)
figure
for roi = roisToCombine
    %create data and labels
    data = cell2mat(allBetasCombinedFiltered{roi})';
    labels = repelem(1:length(stimNames), numStimRepeats);
    [data, labels] = shuffleDataLabelOrder(data, labels);
    %fit sv 
    numFolds = 5;
    svm = fitcecoc(data, labels, 'CrossVal', 'on', 'KFold', numFolds);
    % eval
    cvLoss = kfoldLoss(svm); % Cross-validated classification error
    disp(['Cross-validated loss: ', num2str(cvLoss)]);
    predictions = kfoldPredict(svm);
    %plot confusion
    subplot(4,ceil(length(roisToCombine)/4),find(roisToCombine == roi)),
    confusionchart(labels, predictions, 'Normalization', 'column-normalized')
    title(roiNames(roi))
end





%% Make different averaged ROIs. 
% early visual cortex ROI (v1 and v2)
earlyROIs = [1 3];
for stim = 1:length(task{1}.stimNames);
    allBetasEVCROI{stim} = [];
    for roi = earlyROIs
        allBetasEVCROI{stim} = [allBetasEVCROI{stim}; allBetasCombinedFiltered{roi}{stim}];
    end
end

%average EVC roi
for interp = 1:length(task{1}.stimNames)
    allBetasEVCROIAveraged{interp} = mean(allBetasEVCROI{interp}, 2);
end

% mid-visual cortex ROIS (v3, v4, etc)
midROIs = [5 7 9 11 17 19];
for stim = 1:length(task{1}.stimNames);
    allBetasMVCROI{stim} = [];
    for roi = midROIs
        allBetasMVCROI{stim} = [allBetasMVCROI{stim}; allBetasCombinedFiltered{roi}{stim}];
    end
end

%average MVC roi
for interp = 1:length(task{1}.stimNames)
    allBetasMVCROIAveraged{interp} = mean(allBetasMVCROI{interp}, 2);
end

% "late" visual ROIs (IT, FFA, LO, etc).
lateROIs = [13 15 21 23 25 27 29 31 33 39 41 45 47 49 51];
for stim = 1:length(task{1}.stimNames);
    allBetasVVSROI{stim} = [];
    for roi = lateROIs
        allBetasVVSROI{stim} = [allBetasVVSROI{stim}; allBetasCombinedFiltered{roi}{stim}];
    end
end

%average VVS roi
for interp = 1:length(task{1}.stimNames)
    allBetasVVSROIAveraged{interp} = mean(allBetasVVSROI{interp}, 2);
end

%big roi - all ROIs to combine.
for stim = 1:length(task{1}.stimNames);
    allBetasBigROI{stim} = [];
    for roi = [earlyROIs midROIs lateROIs];
        allBetasBigROI{stim} = [allBetasBigROI{stim}; allBetasCombinedFiltered{roi}{stim}];
    end
end

%average big roi
for interp = 1:length(task{1}.stimNames)
    allBetasBigROIAveraged{interp} = mean(allBetasBigROI{interp}, 2);
end



%% classification on the different ROIs (all, early, middle, late)

% classification on big roi
    figure, subplot(2,2,1)
    %create data and labels
    data = cell2mat(allBetasBigROI)';
    labels = repelem(1:length(stimNames), numStimRepeats);
    [data, labels] = shuffleDataLabelOrder(data, labels);
    %fit svm
    numFolds = 5;
    svm = fitcecoc(data, labels, 'CrossVal', 'on', 'KFold', numFolds);
    % eval
    cvLoss = kfoldLoss(svm); % Cross-validated classification error
    disp(['Cross-validated loss: ', num2str(cvLoss)]);
    predictions = kfoldPredict(svm);
    allROIsConfusionChart = confusionchart(labels, predictions, 'Normalization', 'column-normalized');
    title(sprintf('All ROIs confusion chart: %0.2f accuracy', 1-cvLoss))
    
    RSMconfusionCorr = corr(RSM(RSM<1),allROIsConfusionChart.NormalizedValues(RSM<1));
    sprintf('Correlation between confusion chart and RSM: %0.3f', RSMconfusionCorr)


% classification on EVC roi
    subplot(2,2,2)
    %create data and labels
    data = cell2mat(allBetasEVCROI)';
    labels = repelem(1:length(stimNames), numStimRepeats);
    [data, labels] = shuffleDataLabelOrder(data, labels);
    %fit svm
    svm = fitcecoc(data, labels, 'CrossVal', 'on', 'KFold', numFolds);
    % eval
    cvLoss = kfoldLoss(svm); % Cross-validated classification error
    disp(['Cross-validated loss: ', num2str(cvLoss)]);
    predictions = kfoldPredict(svm);
    confusionchart(labels, predictions, 'Normalization', 'column-normalized')
    title(sprintf('Early visual confusion chart: %0.2f accuracy', 1-cvLoss))


% classification on MVC roi
    subplot(2,2,3)
    %create data and labels
    data = cell2mat(allBetasMVCROI)';
    labels = repelem(1:length(stimNames), numStimRepeats);
    [data, labels] = shuffleDataLabelOrder(data, labels);
    %fit svm
    svm = fitcecoc(data, labels, 'CrossVal', 'on', 'KFold', numFolds);
    % eval
    cvLoss = kfoldLoss(svm); % Cross-validated classification error
    disp(['Cross-validated loss: ', num2str(cvLoss)]);
    predictions = kfoldPredict(svm);
    confusionchart(labels, predictions, 'Normalization', 'column-normalized')
    title(sprintf('Mid-level visual confusion chart: %0.2f accuracy', 1-cvLoss))


% classification on VVS roi
    subplot(2,2,4)
    %create data and labels
    data = cell2mat(allBetasVVSROI)';
    labels = repelem(1:length(stimNames), numStimRepeats);
    [data, labels] = shuffleDataLabelOrder(data, labels);
    %fit svm
    svm = fitcecoc(data, labels, 'CrossVal', 'on', 'KFold', numFolds);
    % eval
    cvLoss = kfoldLoss(svm); % Cross-validated classification error
    disp(['Cross-validated loss: ', num2str(cvLoss)]);
    predictions = kfoldPredict(svm);
    confusionchart(labels, predictions, 'Normalization', 'column-normalized')
    title(sprintf('VVS confusion chart: %0.2f accuracy', 1-cvLoss))




%% reliability as a function of the number of repeats
figure, hold on

r = [];
for numVoxels = 1:(numStimRepeats/2)
    corrVals = [];
    for stimulus = 1:length(stimNames)
        corrVals1stim = [];
        for repeat = 1:1000
            subsetVoxels = randsample(1:numStimRepeats, numVoxels*2);
            subset = allBetasBigROI{stimulus}(:,subsetVoxels);
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
ylabel('Correlation between responses to the same stimuli')

opt = fminsearch(@(p) sum((y - (p(1)*sqrt(x) + p(2))).^2), [1, 0]);
a = opt(1); b = opt(2);
y_fit = a * sqrt(x) + b;
R2 = 1 - sum((y - y_fit).^2) / sum((y - mean(y)).^2);
plot(y_fit)


keyboard

%% Make RSMS for all rois, EVC rois, MVC rois, and VVS rois.
% RSM of the big ROI, combining all the small ROIS
    %create empty matrices
    averagedReps = zeros(length(stimNames), size(allBetasBigROI{1},1));
    averagedRepsHalf1 = averagedReps; averagedRepsHalf2 = averagedReps;
    
    %go through and add each averaged stimulus response to the empty average matrix
    for stim = 1:length(task{1}.stimNames);
        %get the average of each of the halves of the stimuli (even or odds)
        averagedRepsHalf1(stim,:) = mean(allBetasBigROI{stim}(:,1:2:end),2);
        averagedRepsHalf2(stim,:) = mean(allBetasBigROI{stim}(:,2:2:end),2);
    end
    
    %rename everything according to earlier convention
    allBetasBigROIAveragedHalf1 = averagedRepsHalf1; allBetasBigROIAveragedHalf2 = averagedRepsHalf2;
    
    %make the RSM using halves.
    half1 = corr(allBetasBigROIAveragedHalf1', allBetasBigROIAveragedHalf2');
    half2 = corr(allBetasBigROIAveragedHalf2', allBetasBigROIAveragedHalf1');
    RSM = (half1+half2)/2;
    
    %plot it
    figure, subplot(2,2,1), imagesc(RSM),
    colormap(hot), colorbar, caxis([-1 1])
    title('RSM: All ROIs combined. Correlation between averaged patterns of activity')
    xlabel('Object number'), ylabel('Object number')
    xticklabels(stimNames), xticks([1:length(stimNames)]), yticklabels(stimNames), yticks([1:length(stimNames)])
    drawLines



% RSM of the EVC ROI
    %create empty matrices
    averagedReps = zeros(length(stimNames), size(allBetasEVCROI{1},1));
    averagedRepsHalf1 = averagedReps; averagedRepsHalf2 = averagedReps;
    
    %go through and add each averaged stimulus response to the empty average matrix
    for stim = 1:length(task{1}.stimNames);
        %get the average of each of the halves of the stimuli (even or odds)
        averagedRepsHalf1(stim,:) = mean(allBetasEVCROI{stim}(:,1:2:end),2);
        averagedRepsHalf2(stim,:) = mean(allBetasEVCROI{stim}(:,2:2:end),2);
    end
    
    %rename everything according to earlier convention
    allBetasEVCROIAveragedHalf1 = averagedRepsHalf1; allBetasEVCROIAveragedHalf2 = averagedRepsHalf2;
    
    %make the RSM using halves.
    half1 = corr(allBetasEVCROIAveragedHalf1', allBetasEVCROIAveragedHalf2');
    half2 = corr(allBetasEVCROIAveragedHalf2', allBetasEVCROIAveragedHalf1');
    EVCRSM = (half1+half2)/2;
    
    %plot it
    subplot(2,2,2), imagesc(EVCRSM),
    colormap(hot), colorbar, caxis([-1 1])
    title('RSM: EVC. Correlation between averaged patterns of activity')
    xlabel('Object number'), ylabel('Object number')
    xticklabels(stimNames), xticks([1:length(stimNames)]), yticklabels(stimNames), yticks([1:length(stimNames)])
    drawLines



% RSM of the mid ROI
    %create empty matrices
    averagedReps = zeros(length(stimNames), size(allBetasMVCROI{1},1));
    averagedRepsHalf1 = averagedReps; averagedRepsHalf2 = averagedReps;
    
    %go through and add each averaged stimulus response to the empty average matrix
    for stim = 1:length(task{1}.stimNames);
        %get the average of each of the halves of the stimuli (even or odds)
        averagedRepsHalf1(stim,:) = mean(allBetasMVCROI{stim}(:,1:2:end),2);
        averagedRepsHalf2(stim,:) = mean(allBetasMVCROI{stim}(:,2:2:end),2);
    end
    
    %rename everything according to earlier convention
    allBetasMVCROIAveragedHalf1 = averagedRepsHalf1; allBetasMVCROIAveragedHalf2 = averagedRepsHalf2;
    
    %make the RSM using halves.
    half1 = corr(allBetasMVCROIAveragedHalf1', allBetasMVCROIAveragedHalf2');
    half2 = corr(allBetasMVCROIAveragedHalf2', allBetasMVCROIAveragedHalf1');
    MVCRSM = (half1+half2)/2;
    
    %plot it
    subplot(2,2,3), imagesc(MVCRSM),
    colormap(hot), colorbar, caxis([-1 1])
    title('RSM: MVC. Correlation between averaged patterns of activity')
    xlabel('Object number'), ylabel('Object number')
    xticklabels(stimNames), xticks([1:length(stimNames)]), yticklabels(stimNames), yticks([1:length(stimNames)])
    drawLines


% RSM of the VVC ROI
    %create empty matrices
    averagedReps = zeros(length(stimNames), size(allBetasVVSROI{1},1));
    averagedRepsHalf1 = averagedReps; averagedRepsHalf2 = averagedReps;
    
    %go through and add each averaged stimulus response to the empty average matrix
    for stim = 1:length(task{1}.stimNames);
        %get the average of each of the halves of the stimuli (even or odds)
        averagedRepsHalf1(stim,:) = mean(allBetasVVSROI{stim}(:,1:2:end),2);
        averagedRepsHalf2(stim,:) = mean(allBetasVVSROI{stim}(:,2:2:end),2);
    end
    
    %rename everything according to earlier convention
    allBetasVVSROIAveragedHalf1 = averagedRepsHalf1; allBetasVVSROIAveragedHalf2 = averagedRepsHalf2;
    
    %make the RSM using halves.
    half1 = corr(allBetasVVSROIAveragedHalf1', allBetasVVSROIAveragedHalf2');
    half2 = corr(allBetasVVSROIAveragedHalf2', allBetasVVSROIAveragedHalf1');
    VVSRSM = (half1+half2)/2;
    
    %plot it
    subplot(2,2,4), imagesc(VVSRSM),
    colormap(hot), colorbar, caxis([-1 1])
    title('RSM: VVS. Correlation between averaged patterns of activity')
    xlabel('Object number'), ylabel('Object number')
    xticklabels(stimNames), xticks([1:length(stimNames)]), yticklabels(stimNames), yticks([1:length(stimNames)])
    drawLines




%% do mlds in different regions
%do EVC
figure
doMLDS(allBetasEVCROIAveraged, mldsReps, colors, task)
sgtitle('EVC mlds')

%do MVC
figure
doMLDS(allBetasMVCROIAveraged, mldsReps, colors, task)
sgtitle('Mid-level cortex Mlds')

%do ventral ROIs
figure
doMLDS(allBetasVVSROIAveraged, mldsReps, colors, task)
sgtitle('VVS mlds')





%% classification - train an SVM on the endpoints and see how it predicts the interpolations
figure
interpSets = {[1:6], [7:12], [13:18], [19:24]};
for set = 1:length(interpSets)
    %define endpoints
    end1 = min(cell2mat(interpSets(set))); end2 = max(cell2mat(interpSets(set)));
    %get data for SVM
    data = [allBetasBigROI{end1}'; allBetasBigROI{end2}'];
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
            percentCat1(fold) = mean(svm.Trained{fold}.predict(allBetasBigROI{interp}'));
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
    if set == 1; title('Grass to leaves'); elseif set == 2, title('Lemons to bananas'); elseif set == 3, title('Petals to buttercream'); elseif set == 4, title('Acorns to redwood'),end
end
sgtitle(sprintf('Classification using all %i voxels', sum(numUsableVoxelsByROI(roisToCombine))))




%%
interpSet = interpSets{1};

%make evc
evcResponses = cell2mat(allBetasEVCROIAveraged);
evcResponses = evcResponses(:,interpSet)';
[coeff, scores, latent] = pca(evcResponses);
EVC = []; EVC.coeff = coeff; EVC.scores = scores; EVC.latent = latent;

%make vmvc
mvcResponses = cell2mat(allBetasMVCROIAveraged);
mvcResponses = mvcResponses(:,interpSet)';
[coeff, scores, latent] = pca(mvcResponses);
MVC = []; MVC.coeff = coeff; MVC.scores = scores; MVC.latent = latent;

%make vvs
vvsResponses = cell2mat(allBetasVVSROIAveraged);
vvsResponses = vvsResponses(:,interpSet)';
[coeff, scores, latent] = pca(vvsResponses);
VVS = []; VVS.coeff = coeff; VVS.scores = scores; VVS.latent = latent;






%%

keyboard









%%%%%%%
%% drawLines
%%%%%%%%%
function drawLines
    hold on
    plot([.5 24.5], [6.5 6.5], 'k'), plot([.5 24.5], [12.5 12.5], 'k'), plot([.5 24.5], [18.8 18.5], 'k')
    plot([6.5 6.5], [.5 24.5], 'k'), plot([12.5 12.5], [.5 24.5], 'k'), plot([18.5 18.5], [.5 24.5], 'k'),




%% do maximum likelihood distance scaling on the averaged representation with voxels from all ROIs
function [allPsi allSigma] = doMLDS(mldsVoxels, mldsReps, colors, task)
   %interpSets = {[1:6], [7:12]};
    interpSets = {[1:6], [7:12], [13:18], [19:24]};
    
    disp('Doing mlds - takes a minute or so.')
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
        
            %simulate n draws of 4 images
            numSamples = 1000;
            averagedInterps = cat(2, allBetasBigROIAveragedMlds{interpSets{set}});
            %can use either correlations or euclidean distance
            corMatrix = corr(averagedInterps); %cosine distance
            %corMatrix = -pdist2(averagedInterps',averagedInterps'); %euclidean distance implementation
            %calculate which pair has a higher correlation
            ims = randi(6,4,numSamples);
            responses = [];
            for trial = 1:numSamples
                responses(trial) = corMatrix(ims(1,trial), ims(2,trial)) < corMatrix(ims(3,trial), ims(4,trial));
                j = ims(1,trial); k = ims(2,trial); l = ims(3,trial); m = ims(4,trial);
                if j == k | l == m | isequal(sort([j k]), sort([l m]));
                    responses(trial) = 2;
                end
            end
    
            % set up initial params
            %psi = [0.5 0.5 0.5 0.5];
            psi = [.2 .4 .6 .8];
            sigma = .2;
            initialParams = [psi, sigma];
            
            %options
            options = optimset('fminsearch'); options.MaxFunEvals = 10000; options.MaxIter = 5000; options.Display = 'off';
            %options.TolFun = .0001;
            
            %search for params
            optimalParams = fminsearch(@(params) computeLoss(params, ims, responses), initialParams, options);
            psi = [0 optimalParams(1:4) 1];
            psi = (psi-min(psi));
            psi = psi/max(psi);
            
            %plot
            subplot(1,4,set), hold on
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
            if set == 1; title('Grass to leaves mlds'); elseif set == 2, title('Lemons to bananas mlds'); elseif set == 3, title('Petals to buttercream mlds'); elseif set == 4, title('Acorns to redwood mlds'),end
        
            title(strcat(task{1}.stimfile.stimulus.interpNames{set}{1}, ' --> ', task{1}.stimfile.stimulus.interpNames{set}{2}))

            allPsi{set}{repititions} = psi;
            allSigmas{set}{repititions} = psi(5);
        end
    end



%%




























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


%%





