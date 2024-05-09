function [ myscreen ] = texobj( varargin )
%
% EVENT-RELATED TEXTURES
%  Experiment to map neural responses to various textures
%
%  Usage: texobj(varargin)
%  Authors: Akshay Jagadeesh
%  Date: 11/05/2018
%

global stimulus

stimulus = struct;

%% Initialize Variables

% add arguments later
getArgs(varargin,{'scan=0', 'testing=0'}, 'verbose=1');
stimulus.scan = scan;
stimulus.debug = testing;
clear scan testing;

%% Stimulus parameters 
%% Open Old Stimfile
stimulus.counter = 1;
%% Setup Screen
if stimulus.scan || true
  myscreen = initScreen('fMRIprojFlex');
else
  myscreen = initScreen('justinOffice')
end

% set background to grey
myscreen.background = 0.5;

%% Initialize Stimulus
myscreen = initStimulus('stimulus',myscreen);
 
% Set response keys
stimulus.responseKeys = [1 2 3 4]; 

% set colors
stimulus.colors.white = [1 1 1];
stimulus.colors.black = [0 0 0];
stimulus.colors.red = [1 0 0];
stimulus.colors.green = [0 1 0];
stimulus.colors.blue = [0 0 1];
stimulus.live.fixColor = stimulus.colors.blue;
stimulus.live.cueColor = stimulus.colors.black;


stimulus.curTrial(1) = 0;

% define the amount of time the stimulus should be on and off.
stimulus.tStimOn = 0.800;
stimulus.tStimOff = 0.200;

% Task important variables
stimulus.interpNames = {{'grass', 'leaves'}, {'lemons', 'bananas'}};
stimulus.layerNames = {'pool4'};
stimulus.poolSize = '10x10';

stimulus.nInterpFams = length(stimulus.interpNames);
stimulus.imSize = 12;
stimulus.stimXPos = 7;
stimulus.interpNums = [0.0 0.2 0.4 0.6 0.8 1.0];

%% Select the condition for this run
% Choose which image and which pooling layer to display on this run on each side
stimulus.stimDir = '~/Desktop/interp/out/';

%% Preload images
mask = imread('Flattop8.tif');
stimulus.images.synths = struct();
disppercent(-inf, 'Preloading images');

% load texture and noise samples
for i = 1:stimulus.nInterpFams
  imName = stimulus.interpNames{i};
  for li = 1:length(stimulus.layerNames)
    layerI = stimulus.layerNames{li};
    for j = stimulus.interpNums
       imageName = sprintf('%s_%1.1f_%s_%s_%s_smp1.png', imName{1}, j, imName{2}, stimulus.poolSize, layerI);
       sd = imread(sprintf('%s%s_%s/%s/', stimulus.stimDir, imName{1}, imName{2}, stimulus.poolSize, imageName));
       %rename image name so you can save in stimulus
       if imageName(end-3:end) == '.png'; imageName = imageName(1:end-4); end
       imageName(imageName=='.') = [];
       %add texture to stimulus variable
       stimulus.images.synths.(imageName) = genTexFromIm(sd, mask);
    end
  end
  
  disppercent(i / stimulus.nInterpFams);
end

disppercent(inf);
clear sd nd


%%%%%%%%%%%%% TASK %%%%%%%%%%%%%%%%%
task{1}{1} = struct;
task{1}{1}.waitForBacktick = 1;

% Stimulus Timing: FAST event related
task{1}{1}.segmin = [4.0]; % 4 sec trials
task{1}{1}.segmax = [4.0];

if stimulus.debug==1
  task{1}{1}.segmin = [.10]; % 4 sec trials
  task{1}{1}.segmax = [.10];
end

stimulus.blank = 1;

stimulus.seg = {};
stimulus.seg.stim = 1;
%stimulus.seg.ITI = 2;

% Trial parameters
task{1}{1}.synchToVol = zeros(size(task{1}{1}.segmin));
task{1}{1}.getResponse = zeros(size(task{1}{1}.segmin));
task{1}{1}.numTrials = 20;
task{1}{1}.random = 1;
task{1}{1}.seglenPrecomputeSettings.framePeriod=1.0;
if stimulus.scan
  task{1}{1}.synchToVol(end) = 1;
  % Shorten the last segment to account for synchtovol
  task{1}{1}.segmin(end) = max(0, task{1}{1}.segmin(end) - 0.200);
  task{1}{1}.segmax(end) = max(0, task{1}{1}.segmax(end) - 0.200);
end
% Initialize task parameters
task{1}{1}.parameter.imageClass = 1:length(stimulus.interpNames);
task{1}{1}.parameter.interpVal = [-1 stimulus.interpNums stimulus.interpNums]
task{1}{1}.parameter.stimXPos = stimulus.stimXPos;

task{1}{1}.randVars.calculated.blank = NaN;
task{1}{1}.randVars.calculated.layer = NaN;
task{1}{1}.randVars.calculated.tSegStart = {NaN};
task{1}{1}.randVars.calculated.sample = NaN;


%%% 
% Create a second task on the left side.
task{2}{1} = task{1}{1};
task{2}{1}.parameter.stimXPos = -stimulus.stimXPos;


for i = 1:length(task{1})
  [task{1}{i}, myscreen] = initTask(task{1}{i},myscreen,@startSegmentCallback,@screenUpdateCallback,@getResponseCallback,@startTrialCallback,[],[]);
end

for i = 1:length(task{2})
  [task{2}{i}, myscreen] = initTask(task{2}{i},myscreen,@startSegmentCallback,@screenUpdateCallback,@getResponseCallback,@startTrialCallback,[],[]);
end


%%%%
% set the third task to be the fixation staircase task
[task{3} myscreen] = fixStairInitTask(myscreen);



%% Main Task Loop
mglClearScreen(0.5); 
upFix(stimulus);
mglFlush;
mglClearScreen(0.5); 
upFix(stimulus);

phaseNum = 1;
task{1}{1}.blockTrialnum = 0;
% Again, only one phase.
while (phaseNum <= length(task{1})) && ~myscreen.userHitEsc && (task{1}{1}.blockTrialnum <= task{1}{1}.numTrials)
  mglClearScreen;
  % update the task
  [task{1}, myscreen, phaseNum] = updateTask(task{1},myscreen,1);
  [task{2}, myscreen, phaseNum] = updateTask(task{2},myscreen,1);

  % update fixation task
  [task{3}, myscreen, phaseNum] = updateTask(task{3}, myscreen, 1);
  % flip screen
  myscreen = tickScreen(myscreen,task);
end

% task ended
mglClearScreen(0.5);
mglTextSet([],32,stimulus.colors.white);
% get count
mglFlush
myscreen.flushMode = 1;

% if we got here, we are at the end of the experiment
myscreen = endTask(myscreen,task);






%%%%%%%%%%%%%%%%%%%%%%%%% EXPERIMENT OVER: HELPER FUNCTIONS FOLLOW %%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Runs at the start of each Trial %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task, myscreen] = startTrialCallback(task,myscreen)

global stimulus

stimulus.live.gotResponse = 0;
stimulus.curTrial(task.thistrial.thisphase) = stimulus.curTrial(task.thistrial.thisphase) + 1;

% Choose the 45 stimuli in this block by randomly sampling with replacement.
task.thistrial.tSegStart = [];

if task.thistrial.stimXPos < 0
  trial_str = sprintf('Trial %i (LEFT)', task.trialnum);
  side = 'left';
else
  trial_str = sprintf('Trial %i (RIGHT)', task.trialnum);
  side = 'right';
end
image = stimulus.interpNames{task.thistrial.imageClass};

%% Determine whether this is a blank trial
if task.thistrial.interpVal == -1
  task.thistrial.blank = 1;
  disp(sprintf('%s: Blank', trial_str));
else
  task.thistrial.blank = 0;
  trialImName = sprintf('%s_%02d_%s_%s_pool4_smp1', stimulus.interpNames{task.thistrial.imageClass}{1}, task.thistrial.interpVal*10, stimulus.interpNames{task.thistrial.imageClass}{2}, stimulus.poolSize);

  stimulus.live.(sprintf('%s_trialStim', side)) = stimulus.images.synths.(trialImName);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Runs at the start of each Segment %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task, myscreen] = startSegmentCallback(task, myscreen)

global stimulus

% Save segment start time;
task.thistrial.tSegStart(task.thistrial.thisseg) = mglGetSecs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Refreshes the Screen %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [task, myscreen] = screenUpdateCallback(task, myscreen)
%%
global stimulus

if task.thistrial.stimXPos < 0
  side = 'left';
else
  side = 'right';
end

% Select which stimulus to display as a function of time since seg start
timeSinceSegStart = mglGetSecs(task.thistrial.tSegStart(task.thistrial.thisseg));

% Flash stimuli on and off - 800ms on, 200ms off.
cycleLen = stimulus.tStimOn + stimulus.tStimOff;
stimOn = mod(timeSinceSegStart, cycleLen) < stimulus.tStimOn;

%stimIdx = ceil(timeSinceSegStart / stimulus.smpLen);
stimIdx = ceil(timeSinceSegStart / cycleLen);

% Draw the stimuli at the correct flicker rate.
if task.thistrial.thisseg== stimulus.seg.stim
  if stimOn && ~task.thistrial.blank
    mglBltTexture(stimulus.live.(sprintf('%s_trialStim', side)), [task.thistrial.stimXPos, 0, stimulus.imSize, stimulus.imSize]);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Called When a Response Occurs %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [task, myscreen] = getResponseCallback(task, myscreen)

global stimulus

if task.thistrial.dead, return; end

validResponse = any(task.thistrial.whichButton == stimulus.responseKeys);

if validResponse
  if stimulus.live.gotResponse==0
    task.thistrial.detected = 1;
    task.thistrial.response = task.thistrial.whichButton - 10;
    stimulus.live.fix = 0;
  else
    disp(sprintf('Subject responded multiple times: %i',stimulus.live.gotResponse));
  end
  stimulus.live.gotResponse=stimulus.live.gotResponse+1;
  task = jumpSegment(task);
else
  disp(sprintf('Invalid response key. Subject pressed %d', task.thistrial.whichButton));
  task.thistrial.response = -1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              HELPER FUNCTIONS                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Draws a cross at center of the screen of color fixColor
function upFix(stimulus, fixColor)
if ieNotDefined('fixColor')
  fixColor = stimulus.live.fixColor;
end
mglFixationCross(1,1,fixColor);

%%% 
% Draws a circular cue at location x,y
function drawCue(x,y, stimulus)
mglGluAnnulus(x,y, 0.75, 0.8, stimulus.live.cueColor, 64);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% totalTrials -- computes # of total trials
%
function [trials] = totalTrials()
%%
% Counts trials + estimates the threshold based on the last 500 trials
% get the files list
files = dir(fullfile(sprintf('~/data/texobj/%s/18*stim*.mat',mglGetSID)));
trials = 0;

for fi = 1:length(files)
    load(fullfile(sprintf('~/data/texobj/%s/%s',mglGetSID,files(fi).name)));
    e = getTaskParameters(myscreen,task);
    e = e{1}; % why?!
    trials = trials + e.nTrials;
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Turns image into a texture
function tex = genTexFromIm(im, mask)
r = flipud(im);

% Resize images to 256
if size(r,1) ~= 256;
  r = imresize(r, 256/size(r,1));
end

% Make sure they have 3 dimensions (even if grayscale)
if size(r,3) == 1
  r = cat(3, r, r, r);
end

% If a mask is passed in, apply as an alpha mask.
if ieNotDefined('mask')
  r(:,:,4) = 255;
else
  r(:,:,4) = mask(:,:,1);
end
% mgl requires the first dimension to be the RGBA dimension
rP = permute(r, [3 2 1]);
tex = mglCreateTexture(rP);  



