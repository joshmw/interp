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

%load mlds2samp.mat
load jwmlds2.mat
keyboard

texNames = [];
for name = 1:length(task{1}.private.texName1)
    texNames = [texNames strcat(string(task{1}.private.texName1(name)), string(task{1}.private.texName2(name)))];
end


for texName = 1:length(texNames)
    
    % get all the interp values
    allInterps = zeros(4,length(task{1}.i(task{1}.texGroup==texNames(texName))));
    allInterps(1,:) = task{1}.i(task{1}.texGroup==texNames(texName));
    allInterps(2,:) = task{1}.j(task{1}.texGroup==texNames(texName));
    allInterps(3,:) = task{1}.k(task{1}.texGroup==texNames(texName));
    allInterps(4,:) = task{1}.l(task{1}.texGroup==texNames(texName));
    responses = task{1}.response(task{1}.texGroup==texNames(texName));
    
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
    psi = psi/max(psi);

    %plot
    figure,scatter((0:.1:1), psi), hold on
    xlabel('interp value')
    ylabel('distance')
    title(strcat(texNames(texName), ' sigma =  ', num2str(optimalParams(12))))
    
    %show the images
    for interpValue = 1:11
        im = imread(strcat(task{1}.stimPath, string(task{1}.private.texName1(texName)), string(task{1}.private.texName2(texName)), '/', '10x10/', string(task{1}.private.texName1(texName)), sprintf('%1.1f',(interpValue-1)/10), string('_' + string(task{1}.private.texName2(texName))), task{1}.stimPath2));
        image('XData', [(interpValue-1)/10-.05, (interpValue-1)/10+.05], 'YData', [psi(interpValue)-.05, psi(interpValue)+.05], 'CData', im)
        xlim([-0.05, 1.05])
        ylim([-0.05, 1.05])
    end

    %equality line
    plot([0 1], [0 1])

end

keyboard
%end function
end

% loss function
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

