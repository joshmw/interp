%    texMlds.m
%
%       
%       usage: texDistance
%       by: josh wilson
%       date: January 2024
%       
%       PURPOSE: Get maximum likelihood distance scaling a la Maloney and Yang, 2003 (jov).
%       Needs inputs in the form of interpolated texture images. Display 2 pairs at once and
%       asks the observer which pair is more dissimilar (top or bottom).
%      



function myscreen = texMlds(varargin)

% check arguments - at the moment there are none, but you can define inputs here instead of manually below
getArgs(varargin);

% initilaize the screen
myscreen = initScreen('deskMonitor'); mglClearScreen; task{1}.waitForBacktick = 1;

% set task parameters
task{1}.seglen = [1 .5 inf]; task{1}.getResponse = [0 0 1]; %length of segments and response segment
task{1}.numTrials = 1000;
task{1}.random=1; 

% create the names of the textures
task = createTexNames(task);

task{1}.parameter.texName = [1:length(task{1}.private.texName1)];

% create the interpolation number parameters for ijkl
task{1}.parameter.interpI = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
task{1}.parameter.interpJ = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
task{1}.parameter.interpK = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
task{1}.parameter.interpL = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];

% stuff to save
task{1}.response = [];
task{1}.i = [];
task{1}.j = [];
task{1}.k = [];
task{1}.l = [];
task{1}.texGroup = [];

% stimuli names
task{1}.stimPath = '/Users/joshwilson/Desktop/interp/out/'
task{1}.stimPath2 = '_10x10_pool4_smp1.png'

% stimulus positions
task{1}.leftPos = -4;
task{1}.rightPos = 4;
task{1}.downPos = -10;
task{1}.upPos = 10;

% initialize the task
for phaseNum = 1:length(task)
    [task{phaseNum} myscreen] = initTask(task{phaseNum},myscreen,@startSegmentCallback,@screenUpdateCallback,@getResponseCallback);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main display loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phaseNum = 1;
while (phaseNum <= length(task)) && ~myscreen.userHitEsc
    % update the task
    [task myscreen phaseNum] = updateTask(task,myscreen,phaseNum);
    % flip screen
    myscreen = tickScreen(myscreen,task);
end

% if we got here, we are at the end of the experiment
myscreen = endTask(myscreen,task);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function that gets called at the start of each segment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task myscreen] = startSegmentCallback(task, myscreen)

task.thistrial.path1 = task.stimPath + string(task.private.texName1(task.thistrial.texName)) + string(task.private.texName2(task.thistrial.texName)) + '/' + '10x10/' + string(task.private.texName1(task.thistrial.texName));
task.thistrial.path2 = string('_' + string(task.private.texName2(task.thistrial.texName))) + task.stimPath2;

% if segment 1, load the images and turn into blt textures.
if task.thistrial.thisseg == 1 
    im1 = imread(strcat(task.thistrial.path1,sprintf('%1.1f',task.thistrial.interpI),task.thistrial.path2));
    im1 = addAlpha(im1);
    task.thistrial.im1Tex = mglCreateTexture(im1);
    im2 = imread(strcat(task.thistrial.path1,sprintf('%1.1f',task.thistrial.interpJ),task.thistrial.path2));
    im2 = addAlpha(im2);
    task.thistrial.im2Tex = mglCreateTexture(im2);
    im3 = imread(strcat(task.thistrial.path1,sprintf('%1.1f',task.thistrial.interpK),task.thistrial.path2));
    im3 = addAlpha(im3);
    task.thistrial.im3Tex = mglCreateTexture(im3);
    im4 = imread(strcat(task.thistrial.path1,sprintf('%1.1f',task.thistrial.interpL),task.thistrial.path2));
    im4 = addAlpha(im4);
    task.thistrial.im4Tex = mglCreateTexture(im4);

    %clear screen and put up fixation cross
    mglClearScreen(0);
    mglFixationCross(1);

elseif task.thistrial.thisseg == 2
     mglBltTexture([task.thistrial.im1Tex task.thistrial.im2Tex task.thistrial.im3Tex task.thistrial.im4Tex], ...
         [task.leftPos task.upPos; task.rightPos task.upPos; task.leftPos task.downPos; task.rightPos task.downPos],0,0,0);
      
     % fixation cross
     mglFixationCross(1)

elseif task.thistrial.thisseg == 3
    mglClearScreen(0);
end

myscreen.flushMode = 1;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%resp%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function that gets called to draw the stimulus each frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task myscreen] = screenUpdateCallback(task, myscreen)

%%%%%%%%%%%%%%%%%%%%%%%%%%
%    responseCallback    %
%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task myscreen] = getResponseCallback(task,myscreen)

task.i = [task.i task.thistrial.interpI];
task.j = [task.j task.thistrial.interpJ];
task.k = [task.k task.thistrial.interpK];
task.l = [task.l task.thistrial.interpL];
task.texGroup = [task.texGroup [string(task.private.texName1(task.thistrial.texName)) + string(task.private.texName2(task.thistrial.texName))]];

task.blockTrialnum

if task.thistrial.whichButton == 1
   task.response = [task.response 1];
   task = jumpSegment(task);
elseif task.thistrial.whichButton == 2
   task.response = [task.response 2];
   task = jumpSegment(task);;
end



%%%%%%%%%%%%%%%%%%%%%%%
%% addAlphaDimension %%
%%%%%%%%%%%%%%%%%%%%%%%
function im = addAlpha(im)
imSize = size(im);
alpha = ones(imSize(1),imSize(2));
im(:,:,4) = alpha*255;

%%%%%%%%%%%%%%%%%%%%
%% createTexNames %%
%%%%%%%%%%%%%%%%%%%%
function task = createTexNames(task)
%name1 = {'acorns_', 'b1_', 'canopy_', 'fern_', 'fern_', 'grass_', 'lemons_', 'moss2_', 'moss3_', 'mango_', 'obsidian_', 'ocean_', 'pebbles_', 'pine_', 'pine2_', 'redwood_', 'rubies_', 'blueberries_', 'autumn_', 'petals_', 'brick_', 'orangePeel_'}
%name2 = {'redwood', 'b6', 'moss', 'grass', 'leaves', 'leaves', 'bananas', 'grass', 'leaves', 'yellowGems', 'licorice', 'sky', 'granite', 'grass', 'moss', 'rockwall', 'cherries', 'beads_', 'fire_', 'buttercream_', 'redwood_', 'orangeFabric_'}
%name1 = {'grass_', 'pebbles2_', }
%name2 = {'leaves', 'granite'}

%small test setf
name1 = {'acorns_',  'grass_', 'lemons_', 'pebbles_',  'petals_',}
name2 = {'redwood',  'leaves', 'bananas', 'granite', 'buttercream',}

assert(length(name1) == length(name2), 'Texture name halves are different lengths');
for tex = 1:length(name1)
    task{1}.private.texName1(tex) = name1(tex);
    task{1}.private.texName2(tex) = name2(tex);
end







