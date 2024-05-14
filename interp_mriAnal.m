function interp_mriAnal(varargin)
% category_texture_analysis.m
%
%  Takes the outputs of 'run_glmDenoise.m' (results) and does analyses.
%
%  Usage: category_texture_analysis(varargin)
%  Authors: Josh+Francesca+Akshay
%  Date: 04/16/2024
%

%get args
getArgs(varargin, {'reliability_cutoff=.6', 'r2cutoff=0', 'stdCutoff=5', 'shuffleData=0', 'zscorebetas=0', 'numBoots=100'});
%set inclusion criteria from floc %% TO DO



%% Load the data

% Task 1 is right visual field  so LEFT HEMISPHERE roi's should be responsive
% Task 2 is LEFT visual field, so RIGHT HEMISPHERE roi's shold be responsive

%load the data
cd('~/data/NEPR207/s625/s62520240429/')
task{1} = load('s0625Task1ManyROIs.mat');
task{2} = load('s0625Task2ManyROIs.mat');

%fix the stim names thing - remove the duplicate of the blanks
for taskNum = 1:2,
    if length(task{taskNum}.stimNames) == 14; task{taskNum}.stimNames(8) = [];task{taskNum}.stimNames(1) = []; end
end

%swap the roiNums for task 2
odds = mod(task{2}.whichROI,2) == 1;
evens = ~odds;
task{2}.whichROI(evens) = task{2}.whichROI(evens) - 1;
task{2}.whichROI(odds) = task{2}.whichROI(odds) + 1;

%shuffle the data if you want to.
if shuffleData
    task{1}.trial_conditions = task{1}.trial_conditions(randperm(length(task{1}.trial_conditions)));
    task{2}.trial_conditions = task{2}.trial_conditions(randperm(length(task{2}.trial_conditions)));
end

%rename rois to contra and ipso so we can combine
roiNames = task{1}.roiNames;
for roi = 1:length(roiNames),
    if roiNames{roi}(1) == 'l';
        roiNames{roi} = strcat('contra ', roiNames{roi}(2:end));
    else
        roiNames{roi} = strcat('ipso ', roiNames{roi}(2:end));
    end
end


%% calculate the reliability
for taskNum = 1:2;
    task{taskNum}.betas = squeeze(squeeze(task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd));
    for boot = 1:numBoots
        condition_amps = {};
        split1_amps = {};
        split2_amps = {};
        %
        for cond = 1:12;%1:length(stimNames);
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
    
    task{taskNum}.reliability = mean(reliability,2);
end



%% Average together the same stimulus presentations, partition by ROI and get rid of voxels under reliability cutoff
%average together trials that were of the same stim type
for taskNum = 1:2
    %get betas
    amplitudes = squeeze(task{taskNum}.models.FIT_HRF_GLMdenoise_RR.modelmd);
    %zscore if you want
    if zscorebetas
        for scan = 1:10
            amplitudes(:,(1 + ((scan-1)*40)):(scan*40)) = zscore(amplitudes(:,(1 + ((scan-1)*40)):(scan*40)), 0, 2);
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
        task{taskNum}.averagedBetas{roi} = task{taskNum}.averaged_amplitudes((task{taskNum}.whichROI == roi)' & (task{taskNum}.reliability > reliability_cutoff) & (task{taskNum}.glmR2_FIT_HRF_GLMdenoise_RR > r2cutoff),:);
    end
end

% combine hemispheres in averaged betas
for roi = 1:length(roiNames)
    averagedBetas{roi} = [task{1}.averagedBetas{roi}; task{2}.averagedBetas{roi}];
end



%% plot split half reliability, R2, and beta amplitudes by ROI

%plot all the reliabilities with the average
figure, subplot(3,1,1), hold on
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
subplot(3,1,2), hold on
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
subplot(3,1,3), hold on
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
        for condition = 1:12;
        %get betas for individual trials, filtering by reliability
            allBetas{taskNum}{roi}{condition} = task{taskNum}.betas((task{taskNum}.whichROI == roi)' & (task{taskNum}.reliability > reliability_cutoff) & (task{taskNum}.glmR2_FIT_HRF_GLMdenoise_RR > r2cutoff),task{taskNum}.trial_conditions==condition);
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
for sub = 1:2:length(roiNames);
    % get the individual trials and reduce dimensions
    singleTrials = cat(2, allBetasCombinedFiltered{sub}{[1 6 7 12]});
    if min(size(singleTrials)) > 1
        %[y stress] = mdscale(pdist(singleTrials'),2);
        [y, stress] = tsne(singleTrials');
    
        % plot the individual trials in different colors
        subplot(4,5,ceil(sub/2)), hold on
        scatter(y(1:40,1),y(1:40,2),'g','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        scatter(y(41:80,1),y(41:80,2),'b','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        scatter(y(81:120,1),y(81:120,2),'r','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        scatter(y(121:160,1),y(121:160,2),'m','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
        title(roiNames(sub))
    end
    
    %label
    xlabel('Dimension 1')
    ylabel('Dimension 2')
    legend('Grass','Leaves','Lemons','Bananas')

end

%% plot embeddings of individual trials in combined ROIS

%pick the ROIs you want to concatenate together
roisToCombine = [1 3 5 7 9 15 23 27 29];

%combine into a big ROI
for stim = 1:length(task{1}.stimNames);
    allBetasBigROI{stim} = [];
    for roi = roisToCombine
        allBetasBigROI{stim} = [allBetasBigROI{stim}; allBetasCombinedFiltered{roi}{stim}];
    end
end

singleTrialsBigROI = cat(2, allBetasBigROI{[1 6 7 12]});
%[y stress] = mdscale(pdist(singleTrialsBigROI'),2);
[y, stress] = tsne(singleTrialsBigROI');

% plot the individual trials in different colors
figure, hold on
scatter(y(1:40,1),y(1:40,2),'g','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
scatter(y(41:80,1),y(41:80,2),'b','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
scatter(y(81:120,1),y(81:120,2),'r','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)
scatter(y(121:160,1),y(121:160,2),'m','filled','MarkerEdgeColor','w','MarkerFaceAlpha',.5)

%label
title('Combined Ventral ROI')
xlabel('Dimension 1')
ylabel('Dimension 2')
legend('Grass','Leaves','Lemons','Bananas')



%% plot distances between conditions
endpointIndices = [1 6 7 12];
colors = {'g','g','g','g','g','g',[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250],[0.9290 0.6940 0.1250]};
endpointNames = {'Grass', 'Leaves', 'Lemons', 'Bananas'};

%plot the correlations OF ALL 40 REPEATS with the other 40 repeats from the endpoint conditions
figure
for endpoint = 1:length(endpointIndices)
    subplot(2,2,endpoint), hold on
    for interp = 1:12
        corSimilarity = corr(allBetasBigROI{endpointIndices(endpoint)}, allBetasBigROI{interp});
        bar(interp, mean(mean(corSimilarity(corSimilarity ~= 1))),'FaceColor',colors{interp})
    end
    xlabel('Stimulus')
    xticks(1:12)
    xticklabels({'Grass','','','','','Leaves','Lemons','','','','','Bananas'})
    ylabel('Correlation')
    ylim([.3 .6])
    title(sprintf('Correlation to %s', endpointNames{endpoint}))
end
      
%plot the correlations of THE AVERAGE OF THE 40 REPEATS with the average of the 40 repeats from the endpoint conditions
figure
for endpoint = 1:length(endpointIndices)
    subplot(2,2,endpoint), hold on
    for interp = 1:12
        %plot the correlation between the averaged interp stim and the endpoint you are on
        corSimilarity = corr(mean(allBetasBigROI{endpointIndices(endpoint)},2), mean(allBetasBigROI{interp},2));
        %bar(interp, mean(mean(corSimilarity)),colors(interp))
        %calculate bootstraps sampling different presentations for the mean
        corBoots = [];
        for boot = 1:numBoots
            corBoots = [corBoots corr(mean(allBetasBigROI{endpointIndices(endpoint)}(:,randi(40,1,40)),2), mean(allBetasBigROI{interp}(:,randi(40,1,40)),2))];
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



%% Try doing to mds on the interpolations...
%average the trials together
for interp = 1:12
    allBetasBigROIAveraged{interp} = mean(allBetasBigROI{interp}, 2);
end

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



%% do maximum likelihood distance scaling on the averaged representation
interpSets = {[1:6], [7:12]};

%iterate through different interps
for set = 1:length(interpSets)

    %simulate n draws of 4 images
    numSamples = 10000;
    averagedInterps = cat(2, allBetasBigROIAveraged{interpSets{set}});
    corMatrix = corr(averagedInterps);
    ims = randi(6,4,numSamples);

    %calculate which pair has a higher correlation
    responses = [];
    for trial = 1:numSamples
        responses(trial) = abs(corMatrix(ims(1,trial)) - corMatrix(ims(2,trial))) > abs(corMatrix(ims(3,trial)) - corMatrix(ims(4,trial)));
    end
    
    % set up initial params
    psi = [1:6];
    sigma = 1;
    initialParams = [psi, sigma];
    
    %options
    options = optimset('fminsearch');
    options.MaxFunEvals = 10000;  
    options.MinFunEvals = 10000;  
    options.MaxIter = 5000;
    
    %search for params
    optimalParams = fminsearch(@(params) computeLoss(params, ims, responses), initialParams, options);
    psi = optimalParams(1:6);
    psi = (psi-min(psi));
    psi = psi/max(psi);
    
    %plot
    figure,scatter(1:6,psi, 'filled', 'MarkerFaceColor', colors{max(interpSets{set})}), hold on
    gaussFit = fitCumulativeGaussian(1:6, psi);
    PSE = gaussFit.mean;
    plot(gaussFit.fitX,gaussFit.fitY,'color',colors{max(interpSets{set})})
    scatter(gaussFit.mean,.5,50,'MarkerFaceColor','r','MarkerEdgeColor','w')
    plot([1 6], [1 6],'k','lineStyle','--')

    %limits and label
    ylim([-0.05, 1.05]);
    xlabel('Synthesized interpolation value')
    ylabel('Neural interpolation value')
    if set == 1; title('Grass to leaves mlds'); elseif set == 2, title('Lemons to bananas mlds'), end

end

%%














%%%%%%%%%%%%%%%%%%%
%% compute loss %%
%%%%%%%%%%%%%%%%%%%
function totalProb = computeLoss(params, ims, responses)
    % set interp values as psi and get differences between top and bottom pair
    psi = params(1:6);
    sigma = params(7);
    for interpVal = 1:6
        ims(ims == (interpVal)) = psi(interpVal);
    end
    diffs = abs(ims(1,:)-ims(2,:)) - abs(ims(3,:)-ims(4,:));
    keyboard
    % count up probability
    totalProb = 0;
    
    for responseNum = 1:length(diffs)
        if responses(responseNum) == 1
            probResponse = -log(normcdf(diffs(responseNum),0,sigma));
            totalProb = totalProb + probResponse;
        elseif responses(responseNum) == 0
            probResponse = -log(1-normcdf(diffs(responseNum),0,sigma));
            totalProb = totalProb + probResponse;
        end
    end



%%




































% 
% %%compute the gsn noise ceiling and compare to Kendrick's GSN
% 
% %%
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












