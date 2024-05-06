function category_texture_analysis(varargin)
% category_texture_analysis.m
%
%  Takes the outputs of 'run_glmDenoise.m' (results) and does analyses.
%
%  Usage: category_texture_analysis(varargin)
%  Authors: Josh+Francesca+Akshay
%  Date: 04/16/2024
%

%get args
getArgs(varargin, {'reliability_cutoff=0'});
%set inclusion criteria from floc %% TO DO


%% Load the data

% Task 1 is right visual field  so LEFT hemisphere roi's should be
% responsive
% Task 2 is LEFT visual field, so RIGHT hemisphere roi's shold be
% responsive

taskNum = 1;
sid = 's0811';

%read in data
if taskNum == 2;
    ffa = 1; pha =4;
elseif taskNum == 1;
    ffa = 5; pha = 8;
end
fileName = strcat('~/data/francesca/', sid, 'Task', num2str(taskNum), '.mat')
load(fileName);
    %TO DO - change to input variable


%% Average together the same stimulus presentations
nPresentations = length(trial_conditions);

for stim = 1:length(stimNames)
    
    %average together the trials that were of that stim type
    averaged_amplitudes(:,stim) = mean(amplitudes(:,(trial_conditions == stim)),2);
    
end

%save new trial conds now we've ordered and averaged trial conditions
averaged_trial_conditions = 1:72;


%% reliability check
for roiNum = 1:length(roiNames)
    reliability_by_roi{roiNum} = reliability_FIT_HRF_GLMdenoise_RR(whichROI== roiNum);
end

figure;
plot(1:length(roiNames), cellfun(@median, reliability_by_roi), 'o'); hold on;
%plot(1:length(roiNames), errorbar2(1:le))
errorbar2(1:length(roiNames), cellfun(@median, reliability_by_roi), cellfun(@(x) 1.96*std(x)/sqrt(length(x)),reliability_by_roi), 'y');
hline(0, ':k');
xticks(1:16)
xticklabels(roiNames)

ylabel('Split half reliability')
title('Split half reliability by ROI')


%% get the data into a workable format partitioned by ROI
for roi = 1:length(roiNames);
    %face responses
    faceResponses = averaged_amplitudes((whichROI == roi)' & reliability_FIT_HRF_GLMdenoise_RR > reliability_cutoff, stimValues(1,:)<=12 & stimValues(2,:) == -1);
    averageFaceResponse{roi} = nanmedian(faceResponses,"all");
    allFaceResponses{roi} = faceResponses;
    allFaceResponsesSTE{roi} = std(faceResponses(:))/sqrt(length(faceResponses(:)));
    %place responses
    houseResponses = averaged_amplitudes((whichROI == roi)' & reliability_FIT_HRF_GLMdenoise_RR > reliability_cutoff, stimValues(1,:)>12 & stimValues(2,:) == -1);
    averageHouseResponse{roi} = nanmedian(houseResponses,"all");
    allHouseResponses{roi} = houseResponses;
    allHouseResponsesSTE{roi} = std(houseResponses(:))/sqrt(length(houseResponses(:)));
    %scrambled faces
    faceScrambles = averaged_amplitudes((whichROI == roi)' & reliability_FIT_HRF_GLMdenoise_RR > reliability_cutoff, stimValues(1,:)<=12 & stimValues(2,:) ~= -1);
    averageFaceScramble{roi} = nanmedian(faceScrambles,"all");
    allFaceScrambles{roi} = faceScrambles;
    allFaceScramblesSTE{roi} = std(faceScrambles(:))/sqrt(length(faceScrambles(:)));
    %scrambled houses
    houseScrambles = averaged_amplitudes((whichROI == roi)' & reliability_FIT_HRF_GLMdenoise_RR > reliability_cutoff, stimValues(1,:)>12 & stimValues(2,:) ~= -1);
    averageHouseScramble{roi} = nanmedian(houseScrambles,"all");
    allHouseScrambles{roi} = houseScrambles;
    allHouseScramblesSTE{roi} = std(houseScrambles(:))/sqrt(length(houseScrambles(:)));
end


%% plot ROI-level differences in response as a sanity check
figure, hold on
plot(1:length(roiNames),cell2mat(averageFaceResponse), 'o', 'Color', 'k', 'MarkerFaceColor', 'k');
plot(1:length(roiNames),cell2mat(averageHouseResponse), 'o', 'Color', 'r',  'MarkerFaceColor', 'r');
plot(1:length(roiNames),cell2mat(averageFaceScramble),  'o', 'Color', 'k');
plot(1:length(roiNames),cell2mat(averageHouseScramble), 'o', 'Color', 'r');

%label
hline(0, ':k');
legend({'Face', 'House', 'Face Synth', 'House Synth'});
xticks(1:16)
xticklabels(roiNames)
ylabel('Average % BOLD response to category')
title('BOLD responses to faces and houses in different ROIs')

%%% Next steps for analyses:
% 1. Category "sensitivity" index: Calculate magnitude difference of each
% voxel in response to faces vs. houses
% 2. Configural "sensitivity" index: Calculate magnitude difference of each voxel in response to original vs synth.
% 3. Category "population selectivity": Calculate distance in pattern response (either cosine or pearson distance) in between category pairs vs. within category pairs.
% 4. Configural "population selectivity": original synth distance vs synth synth distance.
% 5. Configural sensititivty based on magnitude - do voxels in ffa with
% high face responses show more featural selectivity


%% Category sensitivity - response ratio between faces and houses for individual voxels
figure, hold on
for roi = 1:length(roiNames)
    voxelFacePref{roi} = (nanmean(allFaceResponses{roi},2) - nanmean(allHouseResponses{roi},2)) ./ (sqrt(0.5*(nanvar(allFaceResponses{roi},[],2) + nanvar(allHouseResponses{roi},[],2))));
    voxelHousePref{roi} = (nanmean(allHouseResponses{roi},2) - nanmean(allFaceResponses{roi},2)) ./ (sqrt(0.5*(nanvar(allHouseResponses{roi},[],2) + nanvar(allFaceResponses{roi},[],2))));

    %plot faces
    scatter(repmat(roi,1,length(voxelFacePref{roi})), voxelFacePref{roi},'filled','k','markerFaceAlpha',.1)
    scatter(roi,median(voxelFacePref{roi}),'filled','k','markerEdgeColor','w')

end

%labels etc
xticks(1:16)
xticklabels(roiNames)
ylabel('Face-house activation (d-prime)')
ylim([-4 4])
hline(0, ':k');
legend('','Face - house response','', 'House - face response')
title('Individual voxel difference in activation to faces and houses')


%% Configural sensivity index - response difference between synth and orig for individual voxels

figure
for roi = 1:length(roiNames)
    voxelOrigFacePref{roi} = nanmean(allFaceResponses{roi} - nanmean(reshape(allFaceScrambles{roi}, [], 12,2),3),2) ./ nanstd(allFaceResponses{roi} - nanmean(reshape(allFaceScrambles{roi}, [], 12,2),3),[],2); %./ (sqrt(0.5*(nanvar(allFaceResponses{roi},[],2) + nanvar(allFaceScrambles{roi},[],2))));

    voxelOrigHousePref{roi} = nanmean(allHouseResponses{roi} - nanmean(reshape(allHouseScrambles{roi}, [], 12,2),3),2) ./ nanstd(allHouseResponses{roi} - nanmean(reshape(allHouseScrambles{roi}, [], 12,2),3),[],2); %(nanmean(allHouseResponses{roi},2) - nanmean(allHouseScrambles{roi},2)) ./ (sqrt(0.5*(nanvar(allHouseResponses{roi},[],2) + nanvar(allHouseScrambles{roi},[],2))));

    %plot faces
    scatter(repmat(roi,1,length(voxelOrigFacePref{roi})), voxelOrigFacePref{roi},'filled','k','markerFaceAlpha',.1), hold on
    scatter(roi,median(voxelOrigFacePref{roi}),'filled','k','markerEdgeColor','w')

    %plot houses
    scatter(repmat(roi+.2,1,length(voxelOrigHousePref{roi})), voxelOrigHousePref{roi},'filled','r','markerFaceAlpha',.1)
    scatter(roi+.2,median(voxelOrigHousePref{roi}),'filled','r','markerEdgeColor','w')
end

xticks(1:16)
xticklabels(roiNames)
ylabel('Difference between original/synth activation (% BOLD)')
ylim([-4 4])
hline(0, ':k');
legend('','Faces','','Houses')
title('Individual voxel difference in activation to original vs synth')


%% Configural sensitivity based on category sensitivity

figure, hold on
scatter(voxelFacePref{1},voxelOrigFacePref{1},'k','filled')
scatter(voxelFacePref{1},voxelOrigHousePref{1},'r','filled')

legend('Faces', 'Houses')
xlabel('Difference in sensitivity to faces v houses (% BOLD difference)')
ylabel('Difference in sensitivity to original vs synth (% BOLD difference)')
title('Face v House sensitivity and configural sensitivity')
xlim([-1 2.5]), ylim([-1 2.5])


%% Category population selectivity - Distance in response pattern for within category vs between
%get correlations of representations with itself, then other category
%faces
faceFaceCorFFA = corr(allFaceResponses{ffa});
faceHouseCorFFA = corr(allFaceResponses{ffa}, allHouseResponses{ffa});
%houses
houseHouseCorPHA = corr(allHouseResponses{pha});
houseFaceCorPHA = corr(allHouseResponses{pha}, allFaceResponses{pha});

figure
%face cor w faces
subplot(2,3,1)
hist(faceFaceCorFFA(faceFaceCorFFA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of faces (FFA)')
%face cor w houses
subplot(2,3,2)
hist(faceHouseCorFFA(faceFaceCorFFA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of faces and houses (FFA)')
%difference
subplot(2,3,3)
hist(faceFaceCorFFA(faceFaceCorFFA ~= 1) - faceHouseCorFFA(faceFaceCorFFA ~= 1),20), xlim([-2 2]), hold on
vline(median(faceFaceCorFFA(faceFaceCorFFA ~= 1) - faceHouseCorFFA(faceFaceCorFFA ~= 1)), ':k')
xlabel('Difference between within/between category correlation')
%house cor w houses
subplot(2,3,4)
hist(houseHouseCorPHA(houseHouseCorPHA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of houses (PHA)')
%house cor w faces
subplot(2,3,5)
hist(houseFaceCorPHA(houseHouseCorPHA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of houses and faces (PHA)')
%difference
subplot(2,3,6)
hist(houseHouseCorPHA(houseHouseCorPHA ~= 1) - houseFaceCorPHA(houseHouseCorPHA ~= 1),20), xlim([-2 2]); hold on
vline(median(houseHouseCorPHA(houseHouseCorPHA ~= 1) - houseFaceCorPHA(houseHouseCorPHA ~= 1)), ':k')


sgtitle('Within vs between category representational  correlations')
%QUESTION - why are faces not really correlated with eachother? why are
%some house correlations extremely negative??



%% Configural population selectivity
faceSynthCorFFA = (corr(allFaceResponses{ffa} ,allFaceScrambles{ffa}(:,1:2:end)) + corr(allFaceResponses{ffa} ,allFaceScrambles{ffa}(:,2:2:end)))/2;
houseSynthCorPHA = (corr(allHouseResponses{pha}, allHouseScrambles{pha}(:,1:2:end)) + corr(allHouseResponses{pha}, allHouseScrambles{pha}(:,2:2:end)))/2;

figure
%face cor w faces
subplot(2,3,1)
hist(faceFaceCorFFA(faceFaceCorFFA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of faces (FFA)')
%face cor w houses
subplot(2,3,2)
hist(faceSynthCorFFA(faceFaceCorFFA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of faces and synths (FFA)')
%difference
subplot(2,3,3)
hist(faceFaceCorFFA(faceFaceCorFFA ~= 1) - faceSynthCorFFA(faceFaceCorFFA ~= 1),20), xlim([-2 2]), hold on
vline(median(faceFaceCorFFA(faceFaceCorFFA ~= 1) - faceSynthCorFFA(faceFaceCorFFA ~= 1)), ':k')
xlabel('Difference between within/between synth correlation')
%house cor w houses
subplot(2,3,4)
hist(houseHouseCorPHA(houseHouseCorPHA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of houses (PHA)')
%house cor w faces
subplot(2,3,5)
hist(houseSynthCorPHA(houseHouseCorPHA ~= 1),20), xlim([-1 1]);
xlabel('Correlation between all pairs of houses and synths (PHA)')
%difference
subplot(2,3,6)
hist(houseHouseCorPHA(houseHouseCorPHA ~= 1) - houseSynthCorPHA(houseHouseCorPHA ~= 1),20), xlim([-2 2]); hold on
vline(nanmedian(houseHouseCorPHA(houseHouseCorPHA ~= 1) - houseSynthCorPHA(houseHouseCorPHA ~= 1)), ':k')
xlabel('Difference between within/between synth correlation')

sgtitle('Difference between within/between synth correlations')

keyboard
%you are here



%% Population configural selectivity
% Get roi x stimulus class x synth type representations
for roiNum = 1:length(roiNames)

    %pull out the voxels that belong to that ROI
    rois_seperated{roiNum} = averaged_amplitudes([whichROI == roiNum]' & [reliability_FIT_HRF_GLMdenoise_RR > reliability_cutoff],:);

    
    %seperate into image types
    for imageClass = unique(stimValues(1,:))
        
        %seperate in synth types
        for synthType = unique(stimValues(2,:))
            
            %save into substructure: rois_seperated_conditioned{roi (n rois)}{image (24)}{synth (1-3)}(averaged activity)
            rois_seperated_conditioned{roiNum}{imageClass}{max(synthType,0) +1} = ...
                rois_seperated{roiNum}(:,(imageClass-1)*3 + max(synthType,0) +1);
        
            %end synth type loop
        end

    %end image class loop
    end

%end roi loop    
end

% Compute distace correlations between synths and originals
%loop over rois
for roiNum = 1:length(roiNames);

    %loop over images
    for imageClass = 1:length(imNames)
        
        %get the synth-synth and orig-synth correlations
        OSD1 = 1-corr(rois_seperated_conditioned{roiNum}{imageClass}{1}, rois_seperated_conditioned{roiNum}{imageClass}{2});
        OSD2 = 1-corr(rois_seperated_conditioned{roiNum}{imageClass}{1}, rois_seperated_conditioned{roiNum}{imageClass}{3});
        OSD = mean([OSD1 OSD2]);

        SSD = 1-corr(rois_seperated_conditioned{roiNum}{imageClass}{2}, rois_seperated_conditioned{roiNum}{imageClass}{3});
        
        SI = (OSD - SSD) / (OSD + SSD);

        selectivity_indexes{roiNum}{imageClass} = SI;

    %end image class loop
    end

%end roi loop
end

% Plot the selectivity index
figure, hold on
for roiNum = 1:length(roiNames)

    selectivity_vec = cell2mat(selectivity_indexes{roiNum});

    % plot the face data
    scatter(repmat(roiNum,1,12), selectivity_vec(1:12), 'r', 'filled', 'MarkerFaceAlpha', .1)
    scatter(roiNum, nanmedian(selectivity_vec(1:12)), 72, 'r', 'filled')

    % plot the house data
    scatter(repmat(roiNum,1,12), selectivity_vec(13:end), 'b', 'filled', 'MarkerFaceAlpha', .1)
    scatter(roiNum, nanmedian(selectivity_vec(13:end)), 72, 'b', 'filled')

end

%label
ylim([-1 1])
ylabel('selectivity index')
xlabel('ROI')
plot([1:length(roiNames)],repmat(0,1,length(roiNames)),'k')
xticks(1:16)
xticklabels(roiNames)



keyboard



























 %%old code just in case
 figure
for roi = 1:length(roiNames)
    voxelOrigFacePref{roi} = (nanmean(allFaceResponses{roi},2) - nanmean(allFaceScrambles{roi},2)) ./ (sqrt(0.5*(nanvar(allFaceResponses{roi},[],2) + nanvar(allFaceScrambles{roi},[],2))));

    voxelOrigHousePref{roi} = (nanmean(allHouseResponses{roi},2) - nanmean(allHouseScrambles{roi},2)) ./ (sqrt(0.5*(nanvar(allHouseResponses{roi},[],2) + nanvar(allHouseScrambles{roi},[],2))));

    %plot faces
    scatter(repmat(roi,1,length(voxelOrigFacePref{roi})), voxelOrigFacePref{roi},'filled','k','markerFaceAlpha',.1), hold on
    scatter(roi,median(voxelOrigFacePref{roi}),'filled','k','markerEdgeColor','w')

    %plot houses
    scatter(repmat(roi+.2,1,length(voxelOrigHousePref{roi})), voxelOrigHousePref{roi},'filled','r','markerFaceAlpha',.1)
    scatter(roi+.2,median(voxelOrigHousePref{roi}),'filled','r','markerEdgeColor','w')
end

xticks(1:16)
xticklabels(roiNames)
ylabel('Difference between original/synth activation (% BOLD)')
ylim([-4 4])
hline(0, ':k');
legend('','Faces','','Houses')
title('Individual voxel difference in activation to original vs synth')

keyboard

