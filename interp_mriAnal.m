function interp_mriAnal(varargin)
% interp_mriAnal.m
%
%  Takes the outputs of 'run_glmDenoiseInterp.m' (results) and does analyses.
%
%  Usage: interp_mriAnal(varargin)
%  Authors: Josh wilson
%  Date: 04/16/2024
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
getArgs(varargin, {'reliabilityCutoff=.25', 'r2cutoff=0', 'stdCutoff=5', 'shuffleData=0', 'zscorebetas=2', 'numBoots=250', 'nVoxelsNeeded=30' 'showAvgMDS=50', 'showDistancesSingleTrials=0', 'mldsReps=1', 'plotBig=0', 'doROImlds=1,'...
    'numBetasEachScan=48', 'numScansInGLM=20', 'numStimRepeats=40','truncateTrials=(10/10)'});

% Task 1 is right visual field  so LEFT HEMISPHERE roi's should be responsive
% Task 2 is LEFT visual field, so RIGHT HEMISPHERE roi's shold be responsive
%load the data

%cd('~/data/interp/s0625/')
%task{1} = load('s0625Task1ManyManyROIs.mat');
%task{2} = load('s0625Task2ManyManyROIs.mat');

%cd('~/data/interp/s0626/')
%task{1} = load('s0626Task2.mat');
%task{2} = load('s0626Task1.mat');

cd('~/data/interp/s0627/')
task{1} = load('s0627Task2.mat');
task{2} = load('s0627Task1.mat');


%fix the stim names thing - remove the duplicate of the blanks
for taskNum = 1:2,
    if length(task{taskNum}.stimNames) == 14; task{taskNum}.stimNames(8) = [];task{taskNum}.stimNames(1) = []; end
    if length(task{taskNum}.stimNames) == 28; task{taskNum}.stimNames(22) = [];task{taskNum}.stimNames(15) = [];task{taskNum}.stimNames(8) = [];task{taskNum}.stimNames(1) = [] ;end
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
        roiNames{roi} = strcat('contra ', roiNames{roi}(2:end));
    else
        roiNames{roi} = strcat('ipso ', roiNames{roi}(2:end));
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



%% calculate the reliability of individual voxels (consistancy of beta values across presentations)
for taskNum = 1:2;
    task{taskNum}.betas = squeeze(squeeze(task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd));
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
    
    end
    task{taskNum}.allBootReliability = reliability;
    task{taskNum}.reliability = mean(reliability,2);
end



%% Average together the same stimulus presentations, partition by ROI and get rid of voxels under reliability cutoff
%average together trials that were of the same stim type
for taskNum = 1:2
    %get betas
    amplitudes = squeeze(task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd);

    %zscore if you want
    if zscorebetas
        for scan = 1:numScansInGLM
            %zscore by scan
            amplitudes(:,(1 + ((scan-1)*numBetasEachScan)):(scan*numBetasEachScan)) = zscore(amplitudes(:,(1 + ((scan-1)*numBetasEachScan)):(scan*numBetasEachScan)), 0, 2);
        end
    end

    %average
    for stim = 1:length(task{taskNum}.stimNames)
        task{taskNum}.averaged_amplitudes(:,stim) = nanmean(amplitudes(:,(task{taskNum}.trial_conditions == stim)),2);
    end
end

% partition by roi and filter by voxel reliability
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



%% process data for mds
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
for roi = 1:length(roiNames)
    for stim = 1:length(task{1}.stimNames)
        allBetasCombinedFiltered{roi}{stim} = allBetasCombined{roi}{stim}(voxelStd{roi} < stdCutoff,:);
    end
end



%% plot embeddings of indidivual trials in individual ROIs
figure
if length(task{1}.stimNames) == 12; endpointNums = [1 6 7 12];
elseif length(task{1}.stimNames) == 24; endpointNums = [1 6 7 12 13 18 19 24];end

for sub = 1:2:length(roiNames);
    % get the individual trials and reduce dimensions
    singleTrials = cat(2, allBetasCombinedFiltered{sub}{endpointNums});
    if size(singleTrials,1) > 2
        %[y stress] = mdscale(pdist(singleTrials'),2);
        [y, stress] = tsne(singleTrials');
    
        % plot the individual trials in different colors
        subplot(6,8,ceil(sub/2)), hold on
        scatter(y(1:numStimRepeats,1),y(1:numStimRepeats,2),'g','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        scatter(y((numStimRepeats+1):(numStimRepeats*2),1),y((numStimRepeats+1):(numStimRepeats*2),2),'b','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        scatter(y((numStimRepeats*2+1):(numStimRepeats*3),1),y((numStimRepeats*2+1):(numStimRepeats*3),2),'r','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        scatter(y((numStimRepeats*3+1):(numStimRepeats*4),1),y((numStimRepeats*3+1):(numStimRepeats*4),2),'m','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        title(roiNames(sub))

        %save the number of voxels you have
        numUsableVoxelsByROI(sub) = size(singleTrials,1);
    end
    
    %label
    xlabel('Dimension 1')
    ylabel('Dimension 2')
    legend('Grass','Leaves','Lemons','Bananas')

end



%% plot embeddings of individual trials in combined ROIS

%pick the ROIs you want to concatenate together - check they have >5 voxels and are odd (contra)
roisToCombine = 1:length(task{1}.roiNames); roisToCombine = roisToCombine(numUsableVoxelsByROI > nVoxelsNeeded); roisToCombine = roisToCombine(mod(roisToCombine,2)==1);

%combine into a big ROI
for stim = 1:length(task{1}.stimNames);
    allBetasBigROI{stim} = [];
    for roi = roisToCombine
        allBetasBigROI{stim} = [allBetasBigROI{stim}; allBetasCombinedFiltered{roi}{stim}];
    end
end

singleTrialsBigROI = cat(2, allBetasBigROI{endpointNums});
%[y stress] = mdscale(pdist(singleTrialsBigROI'),2);
[y, stress] = tsne(singleTrialsBigROI');

% plot the individual trials in different colors
figure, hold on
conf = 0.50;
% grass
scatter(y(1:numStimRepeats,1),y(1:numStimRepeats,2),'g','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
er1 = error_ellipse(cov(y(1:numStimRepeats,:)),[mean(y(1:numStimRepeats,1)) mean(y(1:numStimRepeats,2))],conf); er1.Color = 'g';
%leaves
scatter(y((numStimRepeats+1):(numStimRepeats*2),1),y((numStimRepeats+1):(numStimRepeats*2),2),'b','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
er2 = error_ellipse(cov(y((numStimRepeats+1):(numStimRepeats*2),:)),[mean(y((numStimRepeats+1):(numStimRepeats*2),1)) mean(y((numStimRepeats+1):(numStimRepeats*2),2))],conf); er2.Color = 'b';
%lemons
scatter(y((numStimRepeats*2+1):(numStimRepeats*3),1),y((numStimRepeats*2+1):(numStimRepeats*3),2),'r','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
er3 = error_ellipse(cov(y((numStimRepeats*2+1):(numStimRepeats*3),:)),[mean(y((numStimRepeats*2+1):(numStimRepeats*3),1)) mean(y((numStimRepeats*2+1):(numStimRepeats*3),2))],conf); er3.Color = 'r';
%bananas
scatter(y((numStimRepeats*3+1):(numStimRepeats*4),1),y((numStimRepeats*3+1):(numStimRepeats*4),2),'m','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
er4 = error_ellipse(cov(y((numStimRepeats*3+1):(numStimRepeats*4),:)),[mean(y((numStimRepeats*3+1):(numStimRepeats*4),1)) mean(y((numStimRepeats*3+1):(numStimRepeats*4),2)),conf]); er4.Color = 'm';
%acorns
%scatter(y((numStimRepeats*4+1):(numStimRepeats*5),1),y((numStimRepeats*4+1):(numStimRepeats*5),2),'k','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
%er4 = error_ellipse(cov(y((numStimRepeats*4+1):(numStimRepeats*5),:)),[mean(y((numStimRepeats*4+1):(numStimRepeats*5),1)) mean(y((numStimRepeats*4+1):(numStimRepeats*5),2)),conf]); er4.Color = 'k';


%label
title('Combined Ventral ROI')
xlabel('Dimension 1')
ylabel('Dimension 2')
legend('Grass','','Leaves','','Lemons','','Bananas','Petals','Buttercream','Acorns','Redwood')



%% plot distances between conditions
endpointIndices = [1 6 7 12];
colors = {'g','g','g','g','g','g',[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],...
    'm','m','m','m','m','m', [0.6350 0.0780 0.1840], [0.6350 0.0780 0.1840], [0.6350 0.0780 0.1840], [0.6350 0.0780 0.1840], [0.6350 0.0780 0.1840], [0.6350 0.0780 0.1840]};
endpointNames = {'Grass', 'Leaves', 'Lemons', 'Bananas','Petals','Buttercream','Acorns','Redwood'};

%plot the correlations OF ALL REPEATS with the other repeats from the endpoint conditions
if showDistancesSingleTrials
    figure
    for endpoint = 1:length(endpointNums)
        subplot(2,4,endpoint), hold on
        for interp = 1:length(task{1}.stimNames)
            corSimilarity = corr(allBetasBigROI{endpointNums(endpoint)}, allBetasBigROI{interp});
            bar(interp, mean(mean(corSimilarity(corSimilarity ~= 1))),'FaceColor',colors{interp})
        end
        xlabel('Stimulus')
        xticks(1:12)
        xticklabels({'Grass','','','','','Leaves','Lemons','','','','','Bananas'})
        ylabel('Correlation')
        ylim([.3 .6])
        title(sprintf('Correlation to %s', endpointNames{endpoint}))
    end
end
      
%plot the correlations of THE AVERAGE OF THE 40 REPEATS with the average of the 40 repeats from the endpoint conditions
figure
for endpoint = 1:length(endpointNums)
    subplot(2,4,endpoint), hold on
    for interp = 1:length(task{1}.stimNames)
        %plot the correlation between the averaged interp stim and the endpoint you are on
        corSimilarity = corr(mean(allBetasBigROI{endpointNums(endpoint)},2), mean(allBetasBigROI{interp},2));
        %bar(interp, mean(mean(corSimilarity)),colors(interp))
        %calculate bootstraps sampling different presentations for the mean
        corBoots = [];
        for boot = 1:numBoots
            corBoots = [corBoots corr(mean(allBetasBigROI{endpointNums(endpoint)}(:,randi(numStimRepeats,1,numStimRepeats)),2), mean(allBetasBigROI{interp}(:,randi(numStimRepeats,1,numStimRepeats)),2))];
        end
        errorbar(interp, mean(corBoots), mean(corBoots)-prctile(corBoots,5), prctile(corBoots,95)-mean(corBoots), 'k')
        scatter(interp, mean(corBoots), 'filled','markerFaceColor',colors{interp})
    end
    %label stuff
    xlabel('Stimulus')
    xticks(1:12)
    xticklabels({'Grass','','','','','Leaves','Lemons','','','','','Bananas'})
    ylabel('Correlation')
    ylim([.5 1]);
    title(sprintf('Correlation to %s', endpointNames{endpoint}))
    vline(6.5,':k');
end



%% Try doing to multi-dimensional scaling on the interpolations...
%average the trials together
for interp = 1:length(task{1}.stimNames)
    allBetasBigROIAveraged{interp} = mean(allBetasBigROI{interp}, 2);
end
showAvgMDS=1;
if showAvgMDS
    %do tsne the grass interpolations
    averagedInterps = cat(2, allBetasBigROIAveraged{[1:6]});
    [y, stress] = tsne(averagedInterps');
    %plot
    figure, subplot(1,2,1)
    plot(y(1:6,1),y(1:6,2),'g')
    
    xlabel('Dimension 1'), ylabel('dimension 2'), title('Grass -> leaves')
    
    averagedInterps = cat(2, allBetasBigROIAveraged{[7:12]});
    [y, stress] = tsne(averagedInterps');
    %plot
    subplot(1,2,2)
    plot(y(1:6,1),y(1:6,2),'Color', [0.9290 0.6940 0.1250])
    
    xlabel('Dimension 1'), ylabel('dimension 2'), title('Lemons -> Bananas')
end



%% do maximum likelihood distance scaling on the averaged representation with voxels from all ROIs
%interpSets = {[1:6], [7:12]};
interpSets = {[1:6], [7:12], [13:18], [19:24]};
figure

disp('Doing mlds - takes a minute or so.')
%iterate through different interps
for repititions = 1:mldsReps

    %average a different set of trials if bootstrapping
    if mldsReps > 1
        for interp = 1:length(task{1}.stimNames)
            allBetasBigROIAveragedMlds{interp} = mean(allBetasBigROI{interp}(:,randi(numStimRepeats,1,numStimRepeats)), 2);
        end
        %if not bootstrapping, use the average of all presentations
    else
            allBetasBigROIAveragedMlds = allBetasBigROIAveraged;
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
        scatter(1:6,psi, 'filled', 'MarkerFaceColor', colors{max(interpSets{set})})
        gaussFit = fitCumulativeGaussian(1:6, psi);
        PSE = gaussFit.mean;
        PSEs{set}(repititions) = PSE;
        plot(gaussFit.fitX,gaussFit.fitY,'color',colors{max(interpSets{set})})
        scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
        plot([1 6], [0 1],'k','lineStyle','--')
    
        %limits and label
        ylim([-0.05, 1.05]);
        xlim([1 6]);
        xlabel('Synthesized interpolation value')
        ylabel('Neural interpolation value')
        if set == 1; title('Grass to leaves mlds'); elseif set == 2, title('Lemons to bananas mlds'); elseif set == 3, title('Petals to buttercream mlds'); elseif set == 4, title('Acorns to redwood mlds'),end
    
        allPsi{set}{repititions} = psi;
        allSigmas{set}{repititions} = psi(5);
    end
end
sgtitle(sprintf('MLDS, all %i voxels in all ROIs', sum(numUsableVoxelsByROI(roisToCombine))))


%% do mlds on the averaged representation across different ROIs
if doROImlds

%%this is across individual ROIs.
figure, sub = 1; numSubs = length(roisToCombine); allSubs = 1:length(numUsableVoxelsByROI);

disp('Doing mlds on the individual ROIs - takes a bit.')
%iterate through different ROIs
for roi = roisToCombine;

    %iterate through different interps
    for repititions = 1:mldsReps
    
        %average a different set of trials if bootstrapping
        if mldsReps > 1
            for interp = 1:length(task{1}.stimNames)
                allBetasSingleROIAveragedMlds{interp} = mean(allBetasCombinedFiltered{roi}{interp}(:,randi(numStimRepeats,1,numStimRepeats)), 2);
            end
            %if not bootstrapping, use the average of all presentations
        else
            for interp = 1:length(task{1}.stimNames)
                allBetasSingleROIAveragedMlds{interp} = mean(allBetasCombinedFiltered{roi}{interp}, 2);
            end
        end
    
        %do the mlds
        for set = 1:length(interpSets)
        
            %simulate n draws of 4 images
            numSamples = 1000;
            averagedInterps = cat(2, allBetasSingleROIAveragedMlds{interpSets{set}});
            corMatrix = corr(averagedInterps);
            ims = randi(6,4,numSamples);
        
            %calculate which pair has a higher correlation
            responses = [];
            for trial = 1:numSamples
                responses(trial) = corMatrix(ims(1,trial), ims(2,trial)) < corMatrix(ims(3,trial), ims(4,trial));
                j = ims(1,trial); k = ims(2,trial); l = ims(3,trial); m = ims(4,trial);
                if j == k | l == m | isequal(sort([j k]), sort([l m]));
                responses(trial) = 2;
            end
            end
    
            % set up initial params
            psi = [0.2 0.4 0.6 0.8];
            sigma = .2;
            initialParams = [psi, sigma];
            
            %search for params
            optimalParams = fminsearch(@(params) computeLoss(params, ims, responses), initialParams, options);
            psi = [0 optimalParams(1:4) 1];
            psi = (psi-min(psi));
            psi = psi/max(psi);
            

            %plot
            subplot(4,numSubs,(set-1)*length(roisToCombine)+sub), hold on
            scatter(1:6,psi, 'filled', 'MarkerFaceColor', colors{max(interpSets{set})})
            gaussFit = fitCumulativeGaussian(1:6, psi);
            PSE = gaussFit.mean;
            PSEs{set}(repititions) = PSE;
            plot(gaussFit.fitX,gaussFit.fitY,'color',colors{max(interpSets{set})})
            scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
            plot([1 6], [0 1],'k','lineStyle','--')
        
            %limits and label
            ylim([-0.05, 1.05]); xlim([1 6])
            %xlabel('Synthesized interpolation value')
            %ylabel('Neural interpolation value')
            title(roiNames{roi},sprintf('%i voxels',numUsableVoxelsByROI(roi)))
        
            allPsiROIs{set}{roi}{repititions} = psi;
            allSigmasROIs{set}{roi}{repititions} = psi(5);
        end
    end
    sub = sub+1;
end
sgtitle('MLDS, individual ROIs')




%% this is using all ROIs EXCEPT the one you are indexing.
figure, sub = 1;

%iterate through different ROIs
for roi = roisToCombine;

    %iterate through different interps
    for repititions = 1:mldsReps

        %make leaveout
        for stim = 1:length(task{1}.stimNames);
            allBetasBigROILeaveout{stim} = [];
            for roiLeave = roisToCombine(~(roisToCombine==roi)) %EXCEPT THE SINGLE ROI
                allBetasBigROILeaveout{stim} = [allBetasBigROILeaveout{stim}; allBetasCombinedFiltered{roiLeave}{stim}];
            end
        end
        
        %get the leaveout average
        for interp = 1:length(task{1}.stimNames)
            allBetasBigROIAveragedLeaveout{interp} = mean(allBetasBigROILeaveout{interp}, 2);
        end


        %do the mlds
        for set = 1:length(interpSets)
        
            %simulate n draws of 4 images
            numSamples = 1000;
            averagedInterps = cat(2, allBetasBigROIAveragedLeaveout{interpSets{set}});
            corMatrix = corr(averagedInterps);
            ims = randi(6,4,numSamples);
        
            %calculate which pair has a higher correlation
            responses = [];
            for trial = 1:numSamples
                responses(trial) = corMatrix(ims(1,trial), ims(2,trial)) < corMatrix(ims(3,trial), ims(4,trial));
                j = ims(1,trial); k = ims(2,trial); l = ims(3,trial); m = ims(4,trial);
                if j == k | l == m | isequal(sort([j k]), sort([l m]));
                responses(trial) = 2;
            end
            end
    
            % set up initial params
            psi = [0.2 0.4 0.6 0.8];
            %psi = [.2 .4 .6 .8];
            sigma = .5;
            initialParams = [psi, sigma];
            
            %search for params
            optimalParams = fminsearch(@(params) computeLoss(params, ims, responses), initialParams, options);
            psi = [0 optimalParams(1:4) 1];
            psi = (psi-min(psi));
            psi = psi/max(psi);
            

            %plot
            subplot(4,numSubs,(set-1)*length(roisToCombine)+sub), hold on
            scatter(1:6,psi, 'filled', 'MarkerFaceColor', colors{max(interpSets{set})})
            gaussFit = fitCumulativeGaussian(1:6, psi);
            PSE = gaussFit.mean;
            PSEs{set}(repititions) = PSE;
            plot(gaussFit.fitX,gaussFit.fitY,'color',colors{max(interpSets{set})})
            scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
            plot([1 6], [0 1],'k','lineStyle','--')
        
            %limits and label
            ylim([-0.05, 1.05]); xlim([1 6])
            %xlabel('Synthesized interpolation value')
            %ylabel('Neural interpolation value')
            title(roiNames{roi},sprintf('%i voxels',length(allBetasBigROILeaveout{1})))
        
        end
    end
    sub = sub+1;

end
sgtitle('MLDS, LEAVING OUT individual ROIs')


end
keyboard

%% compare the neuro MLDS values (all voxels combined) to the behavioral ones
figure, hold on
behavioralSubject = 2;

%load in the psychophysics group data
PPdata = load('~/data/texMlds/texMldsGroupData.mat');

%make sure the psychophysics data are in order of the neuroimaging data
PPtexNums = [2 3 5 1];
disp('!!! The stimuli might be in different order in the behavior and neuroimaging data. Check that they are the same, and reorganize if you need to !!!')
for i = 1:length(interpSets)
    disp(strcat('Neuroimaging data: ', task{1}.stimfile.stimulus.interpNames{i}{1}, '_', task{1}.stimfile.stimulus.interpNames{i}{2}, ', behavioral data: ', PPdata.data.allTexNames{PPtexNums(i)}))
end

%first, draw the comparisons between the voxel psi values and the behavioral psi values
for set = 1:length(interpSets)
    scatter(allPsi{set}{1}, PPdata.data.psi{behavioralSubject}{PPtexNums(set)}(1:2:end), 'filled', 'markerFaceColor', colors{max(interpSets{set})})
end

plot([0 1],[0 1],'k')
xlim([-.05, 1.05]); ylim([-0.05, 1.05]);
xlabel('Neuroimaging Psi value (all voxels)')
ylabel('Behavioral Psi value')


%% compare the neuro MLDS values (INDIVIDUAL ROIS) to the behavioral ones
if doROImlds
    figure, hold on
    
    %first, draw the comparisons between the voxel psi values and the behavioral psi values
    for set = 1:length(interpSets)
        for roi = 1:length(roisToCombine)
            subplot(3, ceil(length(roisToCombine)/3), roi), hold on
            scatter(allPsiROIs{set}{roisToCombine(roi)}{1}, PPdata.data.psi{behavioralSubject}{PPtexNums(set)}(1:2:end), 'filled', 'markerFaceColor', colors{max(interpSets{set})})
            plot([0 1],[0 1],'k')
            title(roiNames{roisToCombine(roi)})
            xlim([-.05, 1.05]); ylim([-0.05, 1.05]);
        end
    
    end
    
    xlabel('Neuroimaging Psi value (all voxels)')
    ylabel('Behavioral Psi value')
end



%% classification - train an SVM on the endpoints and see how it predicts the interpolations
figure

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
    scatter(cell2mat(interpSets(1)),numEndpoint2,'filled','markerFaceColor',colors{max(interpSets{set})});
    %equality line
    plot([1 6], [0 1],'k','lineStyle','--')
    gaussFit = fitCumulativeGaussian(1:6,        numEndpoint2);
    plot(gaussFit.fitX,gaussFit.fitY,'color',colors{max(interpSets{set})})
    scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
    xlabel('Interpolation value')
    ylabel('Percent classified as endpoint 2')
    if set == 1; title('Grass to leaves'); elseif set == 2, title('Lemons to bananas'); elseif set == 3, title('Petals to buttercream'); elseif set == 4, title('Acorns to redwood'),end
end
sgtitle(sprintf('Classification using all %i voxels', sum(numUsableVoxelsByROI(roisToCombine))))


%% classification - individual ROIs
figure, sub = 1; numSubs = length(roisToCombine); allSubs = 1:length(numUsableVoxelsByROI);

for roi = roisToCombine;
    for set = 1:length(interpSets)
        %define endpoints
        end1 = min(cell2mat(interpSets(set))); end2 = max(cell2mat(interpSets(set)));
        %get data for SVM
        data = [allBetasCombinedFiltered{roi}{end1}'; allBetasCombinedFiltered{roi}{end2}'];
        labels = [repmat(0,1,numStimRepeats) repmat(1,1,numStimRepeats)];
        %fit it
        svm = fitcsvm(data, labels, 'CrossVal', 'on', 'KFold', numFolds);
        
        %plot the results of different interpolation classifications
        numEndpoint2 = [];
        for interp = cell2mat(interpSets(set));
            percentCat1 = [];
            for fold = 1:numFolds
                percentCat1(fold) = mean(svm.Trained{fold}.predict(allBetasCombinedFiltered{roi}{interp}'));
            end
            numEndpoint2 = [numEndpoint2 mean(percentCat1)];
        end
        %plot
        subplot(4,numSubs,(set-1)*length(roisToCombine) + sub), hold on
        scatter(cell2mat(interpSets(1)),numEndpoint2,'filled','markerFaceColor',colors{max(interpSets{set})});
        %equality line and PSE
        plot([1 6], [0 1],'k','lineStyle','--')
        gaussFit = fitCumulativeGaussian(1:6, numEndpoint2);
        plot(gaussFit.fitX,gaussFit.fitY,'color',colors{max(interpSets{set})});
        scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
        %label
        xlabel('Interpolation value')
        ylabel('Percent classified as endpoint 2')
        title(roiNames{roi},sprintf('%i voxels',numUsableVoxelsByROI(roi)))
        xlim([1 6])
    end
    sub = sub+1;
end
sgtitle('Classification in individual ROIs')
%%
keyboard




%%%%%%%%%%%%%%%%%%%
%% END OF SCRIPT %%
%%%%%%%%%%%%%%%%%%%










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







%%compute the gsn noise ceiling and compare to Kendrick's GSN

%%
% for cond = 1:12
%     gsnInTask1(:,cond,:) = task{1}.betas(:,task{1}.trial_conditions==cond);
%     gsnInTask2(:,cond,:) = task{2}.betas(:,task{2}.trial_conditions==cond); 
% end
% 
% res1 = performgsn(gsnInTask1(1:5000,:,:),struct('wantshrinkage',1));   
% res2 = performgsn(gsnInTask2(1:5000,:,:),struct('wantshrinkage',1));
% 
% 
% figure
% scatter(task{1}.reliability(mod(task{1}.whichROI(1:5000),2) == 1), res1.ncsnr(mod(task{1}.whichROI(1:5000),2) == 1), 'k', 'filled','markerFaceAlpha',.05), hold on
% scatter(task{2}.reliability(mod(task{2}.whichROI(1:5000),2) == 1), res2.ncsnr(mod(task{2}.whichROI(1:5000),2) == 1), 'k', 'filled','markerFaceAlpha',.05)
% 
% xlabel('Correlation between betas in first/second half of stimulus presentations')
% ylabel('Noise ceiling SNR (kendrick GSN)') 
% %%
% 
% 
% keyboard



% this doesn't work. well, it does for grass/leaves, but it's awful for lemons/bananas. single representations aren't good enough I guess. maybe
% should try with different voxel cutoffs but for now it's shit

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
%         numSamples = 50000;
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
%         % set up initial params
%         psi = [0.5 0.5 0.5 0.5];
%         sigma = .3;
%         initialParams = [psi, sigma];
%         
%         %options
%         options = optimset('fminsearch'); options.MaxFunEvals = 10000; options.MinFunEvals = 0; options.MaxIter = 5000;
%         %options.TolFun = .0001;
%         
%         %search for params
%         optimalParams = fminsearch(@(params) computeLoss(params, ims, responses), initialParams, options);
%         psi = [0 optimalParams(1:4) 1];
%         psi = (psi-min(psi));
%         psi = psi/max(psi);
%         
%         %plot
%         subplot(1,2,set), hold on
%         scatter(1:6,psi, 'filled', 'MarkerFaceColor', colors{max(interpSets{set})})
%         gaussFit = fitCumulativeGaussian(1:6, psi);
%         PSE = gaussFit.mean;
%         PSEs{set}(repititions) = PSE;
%         plot(gaussFit.fitX,gaussFit.fitY,'color',colors{max(interpSets{set})})
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
% 
% 








