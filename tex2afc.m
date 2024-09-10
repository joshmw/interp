%    tex2afc.m
%
%       
%       usage: tex2afc
%       by: josh wilson
%       date: January 2024
%       
%      




function myscreen = tex2afc(varargin)

% check arguments - at the moment there are none, but you can define inputs here instead of manually below
getArgs(varargin);

% initilaize the screen
myscreen = initScreen('deskMonitor'); mglClearScreen; task{1}.waitForBacktick = 1;

% set task parameters
task{1}.seglen = [1 .1 .5 inf]; task{1}.getResponse = [0 0 0 1]; %length of segments and response segment
task{1}.numTrials = 500;
task{1}.random=1; 
task{1}.prior = 0.5; % percentage of drawing from the first distribution (first texture)

% create the names of the textures
task = createTexNames(task);

task{1}.parameter.texName = [1:length(task{1}.private.texName1)];

% create the interpolation number parameters for ijkl
task{1}.parameter.interp = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
task{1}.parameter.interp1 = [0 0.1 0.2 0.3 0.4 0.5];
task{1}.parameter.interp2 = [0.5 0.6 0.7 0.8 0.9 1.0];
task{1}.parameter.trialPrior = [0:.01:1];


% stuff to save
task{1}.response = [];
task{1}.interp = [];
task{1}.texGroup = [];
task{1}.texNum = [];

% stimuli names
task{1}.stimPath = '/Users/joshwilson/Desktop/interp/out/'
task{1}.stimPath2 = '_10x10_pool4_smp1.png'

% stimulus positions
task{1}.leftPos = 0;
task{1}.rightPos = 0;
task{1}.downPos = 0;
task{1}.upPos = 0;

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

    % pick interp value using your prior
    if task.thistrial.trialPrior < task.prior
        task.thistrial.interpVal = task.thistrial.interp1;
    else 
        task.thistrial.interpVal = task.thistrial.interp2;
    end

    disp(task.thistrial.interpVal)

    % load the image
    im = imread(strcat(task.thistrial.path1,sprintf('%1.1f',task.thistrial.interpVal),task.thistrial.path2));
    im = addAlpha(im);
    imRotated = permute(im,[3 1 2]); %ugh
    task.thistrial.imTex = mglCreateTexture(imRotated);

    %load the 2 textures on both sides
    task.thistrial.im1 = imread(strcat(task.thistrial.path1,'0.0',task.thistrial.path2));
    task.thistrial.im2 = imread(strcat(task.thistrial.path1,'1.0',task.thistrial.path2));
    task.thistrial.imTex1 = mglCreateTexture(permute(addAlpha(task.thistrial.im1), [3 1 2]));
    task.thistrial.imTex2 = mglCreateTexture(permute(addAlpha(task.thistrial.im2), [3 1 2]));
    %clear screen and put up fixation cross
    mglClearScreen(0);
    mglFixationCross(1);

elseif task.thistrial.thisseg == 2
     mglBltTexture([task.thistrial.imTex], [task.leftPos task.upPos],0,0,0);
      
     % fixation cross
     %mglFixationCross(1)

elseif task.thistrial.thisseg == 3
    mglClearScreen(0);

elseif task.thistrial.thisseg == 4
    mglClearScreen(0);
    mglBltTexture([task.thistrial.imTex1], [task.leftPos-7.5 task.upPos],0,0,0);
    mglBltTexture([task.thistrial.imTex2], [task.leftPos+7.5 task.upPos],0,0,0);
    %mglTextDraw(string(task.private.texName1(task.thistrial.texName)),[-2 0]);
    %mglTextDraw(string(task.private.texName2(task.thistrial.texName)),[2 0]);




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

task.interp = [task.interp task.thistrial.interpVal];
task.texGroup = [task.texGroup [string(task.private.texName1(task.thistrial.texName)) + string(task.private.texName2(task.thistrial.texName))]];
task.texNum = [task.texNum task.thistrial.texName];

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
%alpha = ones(imSize(1),imSize(2));
mask = imread('~/Desktop/interp/Flattop8.tif');
im(:,:,4) = mask(:,:,1);




%%%%%%%%%%%%%%%%%%%%
%% createTexNames %%
%%%%%%%%%%%%%%%%%%%%
function task = createTexNames(task)
%name1 = {'acorns_', 'b1_', 'canopy_', 'fern_', 'fern_', 'grass_', 'lemons_', 'moss2_', 'moss3_', 'mango_', 'obsidian_', 'ocean_', 'pebbles2_', 'pine_', 'pine2_', 'redwood_', 'rubies_'}
%name2 = {'redwood', 'b6', 'moss', 'grass', 'leaves', 'leaves', 'bananas', 'grass', 'leaves', 'yellowGems', 'licorice', 'sky', 'granite', 'grass', 'moss', 'rockwall', 'cherries'}
name1 = {'acorns_',  'grass_', 'lemons_', 'pebbles_',  'petals_',}
name2 = {'redwood',  'leaves', 'bananas', 'granite', 'buttercream',}

assert(length(name1) == length(name2), 'Texture name halves are different lengths');
for tex = 1:length(name1)
    task{1}.private.texName1(tex) = name1(tex);
    task{1}.private.texName2(tex) = name2(tex);
end





