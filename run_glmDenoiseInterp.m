function results = run_glmDenoise(varargin)

getArgs(varargin, {'subject=[]', 'TR=1.0', ...
                    'varName=imageClass_x_interpVal',...
                    'nVox=200', 'saveGLM=[]', 'stimdur=4.0',...
                    'roiNames', {'lV1', 'rV1', 'lV2', 'rV2', 'lV3', 'rV3', 'lhV4', 'rhV4','lV01','rV01','lV02','rV02','lLO1','rLO1','lLO2','rLO2'} ,...
                    'segNum=1', 'taskNum=2', 'scanNums=[1,2,3,4,5,6,7,8,9,10]'},...
                    'verbose=1');

%{'lV1', 'rV1', 'lV2', 'rV2', 'lV3', 'rV3', 'lhV4', 'rhV4','lV01','rV01','lV02','rV02','lLO1','rLO1','lLO2','rLO2'}

a = dir('MotionComp/TSeries/*.nii');
nScans = length(scanNums);
disp(sprintf('Running glmdenoise on %i scans in MotionComp group', nScans));

desiredModel = 4;

%% Create view and get stimfile.
v = newView;
v = viewSet(v, 'curGroup', 'MotionComp');
v = viewSet(v, 'curScan', scanNums(1));
% Get stimfile 
stimfile = viewGet(v, 'stimfile');
if iscell(stimfile)
  stimfile = stimfile{1};
end

%imNames = stimfile.stimulus.imageNames;
%layerNames = stimfile.stimulus.layerNames;
% Get scan dims
scanDims = viewGet(v, 'scanDims');
%% Get r2
%v2 = newView; v2 = viewSet(v2, 'curGroup', 'Concatenation');
%v2 = viewSet(v2, 'curScan', 1);
%v2 = loadAnalysis(v2, sprintf('erAnal/%s.mat', erName));

%overlays = viewGet(v2, 'overlays');
%r2 = overlays(1).data{1}; % Get r2 map
%ir2 = reshape(r2, prod(size(r2)), 1);

% Get full stimvols
[~,full_stimNames,~] = getStimvol(v,varName, sprintf('taskNum=%i', taskNum), sprintf('segmentNum=%i', segNum));

%% Extract timeseries and stimvols for each scan.
rois = {}; nTimePts = zeros(1,nScans);
design = {}; data = {}; nVox = {};
scanCoords = {}; roi_r2 = {};
i = 1;
for si = scanNums
  % Go through each scan one by one.
  v = viewSet(v, 'curScan', si);

  % Get the Timeseries for all the ROIs.
  rois{i} = loadROITSeries(v, roiNames, si, [], 'keepNAN', false, 'matchScanNum=1');
  
  tSeries = []; nVoxels = zeros(length(roiNames), 2);
  whichROIs = [];
  
  for ri = 1:length(roiNames)
    % Get all voxels in all ROIs
    if length(roiNames) > 1
      roi = rois{i}{ri};
    else
      roi = rois{i};
    end

    
    tSeries = cat(1, tSeries, roi.tSeries);
    
    % Keep track of which voxels belong to which ROIs
    if ri==1, nVoxels(ri,:) = [1, size(roi.tSeries,1)];
    else nVoxels(ri,:) = [nVoxels(ri-1,2)+1, nVoxels(ri-1,2)+size(roi.tSeries,1)]; end
    whichROIs(nVoxels(ri,1) : nVoxels(ri,2)) = ri;

    sc = roi.scanCoords;
    scanCoords{i}{ri} = roi.scanCoords;
    try
      slc = sub2ind(scanDims, sc(1,:), sc(2,:), sc(3,:));
      roi_r2{ri} = ir2(slc);
    end
  end

  data{i} = tSeries;
  
  [stimvol stimNames var] = getStimvol(v,varName, sprintf('taskNum=%i', taskNum),  sprintf('segmentNum=%i', segNum));
  % Make sure stimvol is the same size even if some conditions are missing
  % by adding zeros.
  if length(stimNames) < length(full_stimNames)
    [elem,idx] = setdiff(full_stimNames, stimNames);
    for ei = 1:length(idx)
      idxI = idx(ei);
      elemI = elem{ei};
      stimNames = {stimNames{1:idxI-1}, elemI, stimNames{idxI:end}};
      stimvol = {stimvol{1:idxI-1}, zeros(1,0), stimvol{idxI:end}};
    end
  end
  
  %get the design matrix
  nTimePts(i) = size(rois{i}{1}.tSeries,2);
  design{i} = getDesignMtx(stimvol, nTimePts(i));

  %modify design matrix to drop columns corresponding to blank
  design{i} = design{i}(:,~contains(stimNames, 'sample=NaN'));

  %% TO DO
  % once you add in the condition of which synth you are looking at, take
  % out the conditions where synths don't matter (e.g. synthnum doesn't
  % matter when showing an original)

  %get the params
  params{i} = getRunParameters(v);

  %jw - idk what this does.
  %[stimValues, condNames] = parseConditionName(stimNames);

  %iterate through scans
  i = i+1;

end

keyboard
% remove stim names that we took out of the design matrix
full_stimNamesNoBlanks = full_stimNames(~contains(full_stimNames, 'sample=NaN'));

% what does this do?
%[stimValues, condNames] = parseConditionName(full_stimNamesNoBlanks);

%i think you want to get rid of blanks here.
for scanNum = scanNums, design{scanNum} = design{scanNum}(:,[2:7 9:end]);end 

%% Run GLM Denoise - Single Trial Estimates
opt = struct('wantmemoryoutputs',[1 1 1 1]);

outputdir = '.';
%run the glm

%single trial
[glmdenoise_out, denoisedTSeries] = GLMestimatesingletrial(design,data,stimdur,TR,[outputdir '/glmdenoise_figures'], opt);

%not single trial
%[glmdenoise_out, denoisedTSeries] = GLMdenoisedata(design,data,stimdur,TR,[outputdir '/glmdenoise_figures'], opt);
amplitudes = squeeze(glmdenoise_out{desiredModel}.modelmd);

models.FIT_HRF = glmdenoise_out{2};
models.FIT_HRF_GLMdenoise = glmdenoise_out{3};
models.FIT_HRF_GLMdenoise_RR = glmdenoise_out{4};

    
glm_r2 = glmdenoise_out{desiredModel}.R2;

% Run baseline model
opt.wantlibrary = 0; % switch off HRF fitting
opt.wantglmdenoise = 0; % switch off GLMdenoise
opt.wantfracridge = 0; % switch off ridge regression
opt.wantfileoutputs = [0 1 0 0];
opt.wantmemoryoutputs = [0 1 0 0];
[ASSUME_HRF] = GLMestimatesingletrial(design,data,stimdur,TR,[outputdir '/GLMbaseline'],opt);
models.ASSUME_HRF = ASSUME_HRF{2};

clear glmdenoise_out ASSUME_HRF

%% Estimate reliability
designALL = cat(1,design{:});

% 1. Find which condition was presented on each trial in order.
trialcondition = [];
for p=1:size(designALL,1) % Loop through volumes
    if any(designALL(p,:)) % If this volume contains any event onset, add it to list.
        trialcondition = [trialcondition find(designALL(p,:))];
    end
end
% Now corder contains a list of which condition was shown on each trial
% corder size: 1 x nTrials

% 2. For each condition, get all trials it appears on.
NUMREPS = 40; % include only conditions with at least n repeats in reliability analysis.
repindices = [];
for ci=1:size(designALL,2)
    which_trials = find(trialcondition==ci);
    if length(which_trials) >= NUMREPS
        repindices = cat(2,repindices,which_trials(1:NUMREPS)');  % note that for images with >NUMREPS, we ignore extra.
    end
end

fprintf('There are %i repeated images in the experiment \n',length(repindices))

% We first arrange models from least to most sophisticated (for
% visualization purposes)
model_names = fieldnames(models);
model_names = model_names([4 1 2 3]);

% Create output variable for reliability values
vox_reliabilities = cell(1,length(models));

% For each GLM...
for m = 1 : length(model_names)

    % Get the GLM betas
    betas = models.(model_names{m}).modelmd(:,:,:,repindices(:));  % use indexing to pull out the trials we want
    betas_reshaped = reshape(betas,size(betas,1),size(betas,2),size(betas,3),NUMREPS,[]);  % reshape to X x Y x Z x NUMREPS x CONDITIONS

    % compute reliabilities using a vectorized utility function
    %vox_reliabilities{m} = calccorrelation(betas_reshaped(:,:,:,1,:),betas_reshaped(:,:,:,2,:),5);
    split1 = 1:(NUMREPS/2);
    split2 = floor(NUMREPS/2 + 1):NUMREPS;
    vox_reliabilities{m} = calccorrelation(mean(betas_reshaped(:,:,:,split1,:), 4),...
                                           mean(betas_reshaped(:,:,:,split2,:), 4), 5);
end


%%
% plot
figure;
%subplot(1,2,1);
cmap = [0.2314    0.6039    0.6980
    0.8615    0.7890    0.2457
    0.8824    0.6863         0
    0.9490    0.1020         0];

% For each GLM type we calculate median reliability for voxels within the
% visual ROI and plot it as a bar plot.
for m = 1 : 4
    bar(m,nanmedian(vox_reliabilities{m}),'FaceColor',cmap(m,:), 'FaceAlpha', 0.5, 'EdgeColor', 'None'); hold on
end
ylabel('Median reliability')
% legend(model_names,'Interpreter','None','Location','NorthEast')
% legend boxoff;
set(gca,'Fontsize',16)
set(gca,'TickLabelInterpreter','none')
xtickangle(45)
xticks(1:4)
xticklabels(model_names);
box off;
%ylim([0.1 0.2])
%set(gcf,'Position',[418   412   782   605])
title('Median voxel split-half reliability of GLM models')

%%


%% Save out results to a structure containing all inro.
results = {};
for mi = 1:length(model_names)
    results.(['betas_' model_names{mi}]) = models.(model_names{mi}).modelmd;
    results.(['reliability_' model_names{mi}]) = vox_reliabilities{mi};
    results.(['glmR2_' model_names{mi}]) = models.(model_names{mi}).R2;
end

results.models = models;
results.amplitudes = amplitudes;
results.trial_conditions = trialcondition;
%results.stimValues = stimValues;
%results.condNames = condNames;
results.stimNames = full_stimNames;
results.nVoxels = nVoxels;
results.whichROI = whichROIs;
results.roiNames = roiNames;
%results.imNames = imNames;
%results.er_R2 = r2;
%results.glm_R2 = glm_r2;
results.stimfile = stimfile;
% results.glmdenoise_out = glmdenoise_out;
results.scanCoords = scanCoords;
%results.roi_r2 = roi_r2;
%results.subject = subject;
%results.splithalf_reliability = reliability;
%results.split1_amps = split1_amps;
%results.split2_amps = split2_amps;
%results.split1.amplitudes = split1_amps;
%results.split2.amplitudes = split2_amps;

%%
keyboard
%save('s0850Task1','-struct', 'results')
if ~ieNotDefined('saveGLM')
  assert(ischar(saveGLM), 'saveGLM must be a string containing the full filepath where rois should be saved.'); 
  disp(sprintf('Saving ROIs to location: %s', saveGLM));
  save(saveGLM, '-struct', 'results');
end

return



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getRunParameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function params = getRunParameters(v)

stimfile = viewGet(v, 'stimfile');
if iscell(stimfile)
  stimfile = stimfile{1};
end
e = getTaskParameters(stimfile.myscreen, stimfile.task);
params = {};

% Go through parameters
param_names = fields(e{1}.parameter);
for pi = 1:length(param_names)
  params.(param_names{pi}) = e{1}.parameter.(param_names{pi});
end

% Go through randVars
param_names = fields(e{1}.randVars);
for pi = 1:length(param_names)
  params.(param_names{pi}) = e{1}.randVars.(param_names{pi});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getDesignMtx
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function design_mtx = getDesignMtx(stimvol, nTimePts)

design_mtx = [];
for si = 1:length(stimvol)
    dm = zeros(1, nTimePts);
    dm(stimvol{si}) = 1;
    design_mtx = [design_mtx; dm];
end
design_mtx = design_mtx';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  plotAmplitudes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotAmplitudes(amplitudes, stimValues, roiNames, nVoxels, results, nTrialsPerCond)
%%
r2_cutoff = 5;
layers = unique(stimValues(1,:));
texIdx = [1 2];
isTexture = arrayfun(@(x) any(x==texIdx), stimValues(3,:));

layer_amps = zeros(length(roiNames), length(layers)+1);
for ri = 1:length(roiNames)
  whichVox = nVoxels(ri,:);
  roi_amps = amplitudes(whichVox(1):whichVox(2), :);
  roi_r2 = results.glmdenoise_out.R2(whichVox(1):whichVox(2));
  
  roi_amps = roi_amps(roi_r2 > r2_cutoff,:);
  
  layer_amps(ri, 1) = mean(mean(roi_amps(:, ~isTexture & nTrialsPerCond' > 0),2),1);
  for li = 1:length(layers)
    layer_amps(ri, li+1) = mean(mean(roi_amps(:, stimValues(1,:)==layers(li) & isTexture & nTrialsPerCond' > 0), 2), 1);
  end
end
%%
figure;
bar(layer_amps);
colormap('Winter'); box off;
xlabel('ROIs');
ylabel('GLM Amplitude (% signal change)');
set(gca, 'XTick', 1:length(roiNames));
set(gca, 'XTickLabel', roiNames);
legend({'Phase-Scrambled', 'Portilla-Simoncelli', 'Pool1', 'Pool2', 'Pool4'});
title('GLM Amplitude');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  get_mod_index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function roi_mod_idx = get_mod_index(amplitudes,stimValues, roiNames, nVoxels, results, imNames, nTrialsPerCond)
%%
texIdx = [1 2];
isTexture = arrayfun(@(x) any(x==texIdx), stimValues(3,:));
layers = unique(stimValues(1,:)); 
texFams = unique(stimValues(2,:)); 

mod_idx = zeros(length(layers), length(texFams), size(amplitudes,1));
for li = 1:length(layers)
  for fi = 1:length(texFams)
    ti = mean(amplitudes(:, stimValues(1,:)==layers(li) & stimValues(2,:)==texFams(fi) & isTexture & nTrialsPerCond' > 0), 2);
    ni = amplitudes(:, stimValues(1,:)==layers(li) & stimValues(2,:)==texFams(fi) & ~isTexture & nTrialsPerCond'>0);
    
    if isempty(ni)
      % If we didn't present any trials of that type, then allow
      % phase-scrambles from other "layers"
      ni = mean(amplitudes(:, stimValues(2,:)==texFams(fi) & ~isTexture & nTrialsPerCond'>0),2);
    end
    mi = (ti-ni)./(ti+ni);
    mod_idx(li,fi,:) = mi;
  end
end

%% Go through each ROI
r2_cutoff = 5;
roi_mod_idx = [];
for ri = 1:length(roiNames)
  whichVox = nVoxels(ri,:);
  %scanLinearCoords = sub2ind(viewGet(v, 'scanDims'), rois{1}{ri}.scanCoords(1,:), rois{1}{ri}.scanCoords(2,:), rois{1}{ri}.scanCoords(3,:));
  %ir2 = reshape(r2, numel(r2), 1);
  %roi_r2 = ir2(scanLinearCoords);
  roi_mod = mod_idx(:,:,whichVox(1):whichVox(2));
  
  roi_r2 = results.glmdenoise_out.R2(whichVox(1):whichVox(2));
  roi_mod = roi_mod(:,:, roi_r2>r2_cutoff);
  
  roi_mod_idx = cat(3, roi_mod_idx, median(roi_mod, 3));
  disp(sprintf('Computing modulation index for %g voxels in %s', sum(roi_r2>r2_cutoff), roiNames{ri}));
end

mean_mod = squeeze(mean(roi_mod_idx,2));

figure; bar(mean_mod');
set(gca, 'XTick', 1:length(roiNames));
set(gca, 'XTickLabel', roiNames);
legend({'PS', 'Pool1', 'Pool2', 'Pool4'});
colormap('Winter'); box off;
xlabel('ROIs');
ylabel('Modulation Index');

%%
figure;
for i = 1:size(roi_mod_idx,2)
  imgMod = squeeze(roi_mod_idx(:,i,:));
  subplot(4,5,i);
  bar(imgMod');
  set(gca, 'XTick', 1:length(roiNames));
  set(gca, 'XTickLabel', roiNames);
  if i==1, legend({'PS', 'Pool1', 'Pool2', 'Pool4'}); end
  colormap('Winter'); box off;
  xlabel('ROIs');
  ylabel('Modulation Index');
  title(imNames{i});
end

roi_mod_idx = permute(roi_mod_idx, [1 3 2]);

