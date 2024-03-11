%    mldsAnal.m
%
%       
%       usage: mldsAnal
%       by: josh wilson
%       date: January 2024
%       
%       PURPOSE: Analysis for texDistance.m. Implementation of Larry Maloney's maximum likelihood distance scaling.
%      
function mldsAnal(varargin)

%set variables
SIDnums = {'s600','s0601','s0602'}
%SIDnums = {'s600'}
showImagesOnGraphs = 0;
bootstrap = 1;
numBootstraps = 20;

for SIDnum = 1:length(SIDnums)
    %get name of mat file and load it
    filename = strcat(SIDnums{SIDnum},'mlds.mat');
    load(filename);

    %change SUBnum for subplot reasons
    SUBnum = 1 + 2*(SIDnum-1);
    
    %get the names of the textures from the stim file
    task{1}.stimPath = 'Users/joshwilson/Desktop/interp/out/';
    texNames = [];
    for name = 1:length(task{1}.private.texName1)
        texNames = [texNames strcat(string(task{1}.private.texName1(name)), string(task{1}.private.texName2(name)))];
    end
    
    % iterate through the different texture names
    for texName = 1:length(texNames)

        [psi, PSE] = calcPlotGraphs(texName, texNames, task, showImagesOnGraphs, SUBnum, SIDnums, 0, 1);

        %if you are bootstrapping confidence intervals on psi...
        if bootstrap

            %init empty psi values
            psiBootstraps = []; PSEbootstraps = [];

            %iterate through bootstraps and save psi values
            for strap = 1:numBootstraps
                [psiBootstraps(strap,:), PSEbootstraps(strap)] = calcPlotGraphs(texName, texNames, task, showImagesOnGraphs, SUBnum, SIDnums, 1, 0);
            end

            %calc intervals and plot the bootstraps
            lowPercentilePsi = prctile(psiBootstraps(:,1:11),2.5);
            highPercentilePsi = prctile(psiBootstraps(:,1:11),97.5);
            subplot(length(SIDnums),2,SUBnum)
            errorbar(0:.1:1,psi,psi-lowPercentilePsi,highPercentilePsi-psi)

            lowPercetilePSE = prctile(PSEbootstraps,2.5);
            highPercetilePSE = prctile(PSEbootstraps,97.5);
            subplot(length(SIDnums),2,SUBnum+1);
            errorbar(PSE, .5, PSE - lowPercetilePSE, highPercetilePSE - PSE, 'horizontal');

        %end bootstrapping
        end
    %end textures
    end
%end subject loop
end

%end function
end




%%%%%%%%%%%%%
%% calcPlotGraphs %%
%%%%%%%%%%%%%%
function [psi, PSE] = calcPlotGraphs(texName, texNames, task, showImagesOnGraphs, SUBnum, SIDnums, bootstrap, plotFunctions)

% get all the interp values
allInterps = zeros(4,length(task{1}.i(task{1}.texGroup==texNames(texName))));
allInterps(1,:) = task{1}.i(task{1}.texGroup==texNames(texName));
allInterps(2,:) = task{1}.j(task{1}.texGroup==texNames(texName));
allInterps(3,:) = task{1}.k(task{1}.texGroup==texNames(texName));
allInterps(4,:) = task{1}.l(task{1}.texGroup==texNames(texName));
responses = task{1}.response(task{1}.texGroup==texNames(texName));

%bootstrapping
if bootstrap
    [allInterps(1,:), idx] = datasample(allInterps(1,:),length(allInterps(1,:)));    
    allInterps(2,:) = allInterps(2,idx);
    allInterps(3,:) = allInterps(3,idx);
    allInterps(4,:) = allInterps(4,idx);
    responses = responses(idx);
end

% set up initial params
psi = (0:10)/10;
sigma = .2;
initialParams = [psi, sigma];

%options
options = optimset('fminsearch');
options.MaxFunEvals = 10000;  
options.MinFunEvals = 10000;  

%loss func
lossFunction = @(params) computeLoss(params, allInterps, responses, task);

%search for params
optimalParams = fminsearch(@(params) computeLoss(params, allInterps, responses, task), initialParams, options);
psi = optimalParams(1:11);
psi = (psi-min(psi));
psi = psi/max(psi);

%fit a cumulative gaussian to it
gaussFit = fitCumulativeGaussian([0:.1:1], psi);
PSE = gaussFit.mean;

if plotFunctions
    %plot data
    figure(texName), subplot(length(SIDnums),2,SUBnum), hold on
    plot((0:.1:1), psi,'k')
    xlabel('Interp value')
    ylabel('Perceptual distance')
    sgtitle(strcat(texNames(texName), ' sigma =  ', num2str(optimalParams(12))))
    title('Behavioral Data')
    
    %equality line
    plot([0 1], [0 1],'r','lineStyle','--')
    
    %show the images
    if showImagesOnGraphs
        for interpValue = 1:11
            im = imread(strcat(task{1}.stimPath, string(task{1}.private.texName1(texName)), string(task{1}.private.texName2(texName)), '/', '10x10/', string(task{1}.private.texName1(texName)), sprintf('%1.1f',(interpValue-1)/10), string('_' + string(task{1}.private.texName2(texName))), task{1}.stimPath2));
            image('XData', [(interpValue-1)/10-.05, (interpValue-1)/10+.05], 'YData', [psi(interpValue)-.05, psi(interpValue)+.05], 'CData', im)
        end
    end
    
    %set graph limits
    xlim([-0.05, 1.05])
    ylim([-0.05, 1.05])
    
    %fit a polynomial and plot on a new subplot
    %polyfit = fit([0:.1:1]', psi', 'poly2');
    %plot(polyfit,'r')
    %subplot(length(SIDnums),2,SUBnum+1), hold on
    %plot([0 1], [0 1],'r','lineStyle','--')
    
    %fit a cumulative gaussian and plot on a new subplot
    subplot(length(SIDnums),2,SUBnum+1), hold on
    plot([0 1], [0 1],'r','lineStyle','--')
    plot(gaussFit.fitX,gaussFit.fitY,'b')
    scatter(gaussFit.mean,.5,'k','filled')  
    
    
    %equality line, label
    plot([0 1], [0 1],'r','lineStyle','--')
    title('Cumulative gaussian fit')
    xlabel('Interp value')
    ylabel('Perceptual distance')
    title('Cumulative gaussian fits')
    
    %limits
    xlim([-0.05, 1.05])
    ylim([-0.05, 1.05])
end

end



%%%%%%%%%%%%%%%%%%%
%% compute loss %%
%%%%%%%%%%%%%%%%%%%
function totalProb = computeLoss(params, allInterps, responses, task)
    % set interp values as psi and get differences between top and bottom pair
    psi = params(1:11);
    sigma = params(12);
    for interpVal = 1:11
        allInterps(allInterps == (interpVal-1)/10) = psi(interpVal);
    end
    diffs = abs(allInterps(1,:)-allInterps(2,:)) - abs(allInterps(3,:)-allInterps(4,:));
    
    % count up probability
    totalProb = 0;
    %keyboard
    for responseNum = 1:length(diffs)
        if responses(responseNum) == 1
            probResponse = -log(normcdf(diffs(responseNum),0,sigma));
            totalProb = totalProb + probResponse;
        elseif responses(responseNum) == 2
            probResponse = -log(1-normcdf(diffs(responseNum),0,sigma));
            totalProb = totalProb + probResponse;
        end
    end
end

