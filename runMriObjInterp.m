function runMriObjInterp

%% first, get all the data
subs = ['sub=s0605'; 'sub=s0606'; 'sub=s0607'; 'sub=s0608'; 'sub=s0609'; 'sub=s0610']
numSubs = size(subs,1)

for sub = 1:numSubs

    %get the values
    [unaveragedBrainCatVals, unaveragedBrainR2Vals, unaveragedCornetCatVals, unaveragedCornetR2Vals, unaveragedMldsCatVals, unaveragedMldsR2Vals, unaveragedBigRoiCatVals, unaveragedBigRoiR2Vals, unaveragedNNCatVals, unaveragedNNR2Vals, EVCRSM, MVCRSM, VVSRSM, VVSDotProduct, unaveragedConeCatVals, unaveragedConeR2Vals] = mriObjInterp(subs(sub,:));

    %save them
    brainCatVals{sub} = unaveragedBrainCatVals;
    brainR2Vals{sub} = unaveragedBrainR2Vals;
    cornetCatVals{sub} = unaveragedCornetCatVals;
    cornetR2Vals{sub} = unaveragedCornetR2Vals;
    mldsCatVals{sub} = unaveragedMldsCatVals;
    mldsR2Vals{sub} = unaveragedMldsR2Vals;
    bigRoiCatVals{sub} = unaveragedBigRoiCatVals;
    bigRoiR2Vals{sub} = unaveragedBigRoiR2Vals;
    EVCRSMs{sub} = EVCRSM;
    MVCRSMs{sub} = MVCRSM;
    VVSRSMs{sub} = VVSRSM;
    VVSDotProducts{sub} = VVSDotProduct;
    NNCatVals{sub} = unaveragedNNCatVals;
    NNR2Vals{sub} = unaveragedNNR2Vals;
    coneCatVals{sub} = unaveragedConeCatVals;
    coneR2Vals{sub} = unaveragedConeR2Vals;
end


keyboard


%% sort the data to be usable - get out of cells
r2_cutoff = 0.75;

brainCatVals = cell2mat(brainCatVals');
brainR2Vals = cell2mat(brainR2Vals');
cornetCatVals = cell2mat(cornetCatVals');
cornetR2Vals = cell2mat(cornetR2Vals');
mldsCatVals = cell2mat(mldsCatVals');
mldsR2Vals = cell2mat(mldsR2Vals');
bigRoiCatVals = cell2mat(bigRoiCatVals');
bigRoiR2Vals = cell2mat(bigRoiR2Vals');
NNCatVals = cell2mat(NNCatVals');
NNR2Vals = cell2mat(NNR2Vals');
coneR2Vals = cell2mat(coneR2Vals');
coneCatVals = cell2mat(coneCatVals');
catTaskCatVals = brainCatVals(:,4);
catTaskR2Vals = brainR2Vals(:,4);



%% set cutoffs
catTaskR2Cutoff = 0.92;
mldsR2Cutoff = 0.8;
brainR2Cutoff = 0.5; 
NNR2Cutoff = 0.9;
coneR2Cutoff = 0.8;




%% Do some analyses
%brain vs mlds
figure, hold on
for area = 1:3,
    %subplot(1,3,area)
    scatter(brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area) , mldsCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff)), 'filled')
    plot([0 1], [0 1], '--k')
end
xlabel('Brain categorical index'), ylabel('Mlds categorical index'), title('Brain and MLDS categorical indices')
legend({'Early', '', 'Middle', '', 'Late'})


%brain vs categorization
figure, hold on
for area = 1:3,
    %subplot(1,3,area)
    scatter(brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), area) , brainCatVals((brainR2Vals(:,area) > brainR2Cutoff) & (mldsR2Vals > mldsR2Cutoff), 4), 'filled')
    plot([0 1], [0 1], '--k')
end
xlabel('Brain categorical index'), ylabel('Cat task categorical index'), title('Brain and categorization categorical indices')
legend({'Early', '', 'Middle', '', 'Late'})

%mlds vs categorization
figure, hold on
scatter(mldsCatVals((mldsR2Vals > mldsR2Cutoff) & (catTaskR2Vals > brainR2Cutoff)), catTaskCatVals((mldsR2Vals > mldsR2Cutoff) & (catTaskR2Vals > brainR2Cutoff)), 'filled')
xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
xlabel('Mlds categorical index'), ylabel('Categorization task categorical index'), title('Mlds and Categorization task categorical indices')



%Whole brain vs mlds
figure, hold on
scatter(mldsCatVals((mldsR2Vals > mldsR2Cutoff) & (bigRoiR2Vals > brainR2Cutoff)), bigRoiCatVals((mldsR2Vals > mldsR2Cutoff) & (bigRoiR2Vals > brainR2Cutoff)), 'filled')
xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
xlabel('Mlds categorical index'), ylabel('Brain categorical index'), title('Whole brain categorical index vs mlds')



%% do "classification" with the brain data by comparing how close the endpoints are
figure
interpSets = {[1:6], [7:12], [13:18], [19:24]};

x = []; y = [];
for sub = 1:numSubs
    for set = 1:length(VVSRSMs{sub})/6
        low = min(interpSets{set}); high = max(interpSets{set});
        dists = VVSRSMs{sub}(low:high,high) - VVSRSMs{sub}(low:high,low); ylabelstr = 'Corr(interp,6) - Corr(interp, 1)';
        %dists = VVSDotProducts{sub}(low:high,high) - VVSDotProducts{sub}(low:high,low); ylabelstr = 'Dot(interp,6) - Dot(interp, 1)';

        subplot(1,2,1), hold on, plot([1 6], [0 0], '--k'), xlim([0.5, 6.5])
        scatter(1:6, dists, 'k', 'filled', 'MarkerFaceAlpha', 0.2, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w')
        ylabel(ylabelstr), xlabel('Interpolation point')
        title('Relative distance to endpoints')

        subplot(1,2,2), hold on, ylim([0.8 2.2]), xlim([0.5, 6.5])
        scatter(1:6, (dists>0)+1, 'k', 'filled', 'MarkerFaceAlpha', 0.1, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w')
        ylabel('Closest endpoint'), xlabel('Interpolation point')
        title('Classification of interpolations')
        x = [x 1:6]; y = [y dists>0];

    end
end

g = fitCumulativeGaussian(x,y(:)');
plot(g.fitX,g.fitY+1, 'k')

for interp = 1:6, choices(interp) = g.fitY(g.fitX == interp); end
brainCatTaskMatrix = 1- dist(choices);
[c r] = compareCatRSM(brainCatTaskMatrix, 1);




%% Make the big plot
figure, hold on, xlim([0.5 4.5]), ylim([-0.1 1.1])


% Initialize storage
filtered_NNCatVals = cell(size(NNR2Vals,3), 4);  % {model, area}
filtered_cornetCatVals = cell(1,4);
filtered_brainCatVals = cell(1,3);
filtered_mldsCatVals = [];
filtered_coneCatVals = [];
filtered_catTaskCatVals = [];

% Plot the models and store values
for model = 1:size(NNR2Vals,3)
    for area = 1:4
        temp_vals = [];
        for interp = 1:length(brainR2Vals)
            if (NNR2Vals(interp, area, model) > NNR2Cutoff)
                val = NNCatVals(interp, area, model);
                scatter(area-0.1, val, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
                temp_vals(end+1) = val;
            end
        end
        filtered_NNCatVals{model, area} = temp_vals;
    end
end

% Cornet model
for area = 1:4
    temp_vals = [];
    for interp = 1:length(cornetR2Vals)
        if (cornetR2Vals(interp, area) > NNR2Cutoff)
            val = cornetCatVals(interp, area);
            scatter(area-0.1, val, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
            temp_vals(end+1) = val;
        end
    end
    filtered_cornetCatVals{area} = temp_vals;
end

% Brain data
for area = 1:3
    temp_vals = [];
    for interp = 1:length(brainR2Vals)
        if (brainR2Vals(interp, area) > brainR2Cutoff) && ...
           (mldsR2Vals(interp) > mldsR2Cutoff) && ...
           (catTaskR2Vals(interp) > catTaskR2Cutoff)
            val = brainCatVals(interp, area);
            scatter(area+0.1, val, 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
            temp_vals(end+1) = val;
        end
    end
    filtered_brainCatVals{area} = temp_vals;
end

% MLDS
for interp = 1:length(mldsR2Vals)
    if (mldsR2Vals(interp) > mldsR2Cutoff)
        val = mldsCatVals(interp);
        scatter(4.1, val, 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
        filtered_mldsCatVals(end+1) = val;
    end
end

% CONES
for interp = 1:length(coneR2Vals)
    if (coneR2Vals(interp) > coneR2Cutoff)
        val = coneCatVals(interp);
        scatter(0.8, val, 36, 'filled', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
        filtered_coneCatVals(end+1) = val;
    end
end


% Categorization task
for interp = 1:length(catTaskR2Vals)
    if (catTaskR2Vals(interp) > catTaskR2Cutoff)
        val = catTaskCatVals(interp);
        scatter(4.1, val, 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
        filtered_catTaskCatVals(end+1) = val;
    end
end

% Axis labels
xticks(1:4), xticklabels({'Early', 'Middle', 'Late', 'Behavior'}), xlabel('ROI/Layer')
ylabel('Categorical index')
title('Categorical influence on RSM')

%legend
h1 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Neural network');
h2 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Brain');
h3 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'MLDS');
h4 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Categorization task');
h5 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Cone mosaic');
legend([h1, h2, h3, h4, h5]);




















%%
keyboard








%%
function [catVals, r2s] = compareCatRSM(inputRSM, normalize)

inputRSMs{1} = inputRSM;

%init empty array
catVals = [];
r2s = [];
% create the 2 null hypothesis matrices
categoricalRSMCovar = [ones(3) zeros(3); zeros(3) ones(3)];
linearRSMCovar = max(0, 1 - 0.2 * abs((1:6)' - (1:6)));

for inputNum = 1:length(inputRSMs)
    %get the RSM
    inputRSM = inputRSMs{inputNum};
    %if not already normalized, normalize (unaveraged RSMs aren't normalized)
    if normalize
        inputRSM = (inputRSM - min(inputRSM(:))) / (max(inputRSM(:)) - min(inputRSM(:)));
    end
    %compute the idealized matrices based on measured variance
    inputVar = sqrt(diag(inputRSM));
    varMatrix = inputVar*inputVar';
    linearRSM = varMatrix .* linearRSMCovar;
    categoricalRSM = varMatrix .* categoricalRSMCovar;
    
    %find betas that describe input of linear/categorical matrices to observed matrix
    [categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM, inputNum);
    catVals = [catVals categoricalBeta / (categoricalBeta + linearBeta)];
    r2s = [r2s r2];

end



%%%%%%%%%%%%%%%%%%
%% findCatLinearEvidence %%
%%%%%%%%%%%%%%%%%%%%%
function [categoricalBeta, linearBeta, r2] = findCatLinearEvidence(inputRSM, categoricalRSM, linearRSM, inputNum)

%mask to ignore diagonal - it's the same between both conditions but might cause a shitty fit in the neural networks
mask = ~eye(size(inputRSM));

%define the objective function to minimize absolute difference
objective = @(betas) sum(sum(abs(inputRSM(mask) - ...
    (betas(1) * categoricalRSM(mask) + betas(2) * linearRSM(mask))).^2));

% Optimize Betas using fminsearch
betas = fminsearch(objective, [.5, .5]); % Initial guesses for Betas


%r-squared of the fit matrix
fitRSM = categoricalRSM * betas(1) + linearRSM * betas(2);
r2 = corr(fitRSM(~eye(size(fitRSM))), inputRSM(~eye(size(inputRSM))))^2;

categoricalBeta = betas(1)/sum(betas);
linearBeta = betas(2)/sum(betas);








