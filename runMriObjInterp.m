function runMriObjInterp

%% first, get all the data
subs = ['sub=s0605'; 'sub=s0606'; 'sub=s0607']
numSubs = size(subs,1)

for sub = 1:numSubs

    %get the values
    [unaveragedBrainCatVals, unaveragedBrainR2Vals, unaveragedCornetCatVals, unaveragedCornetR2Vals, unaveragedMldsCatVals, unaveragedMldsR2Vals, unaveragedBigRoiCatVals, unaveragedBigRoiR2Vals, unaveragedNNCatVals, unaveragedNNR2Vals, VVSRSM, VVSDotProduct] = mriObjInterp(subs(sub,:));

    %save them
    brainCatVals{sub} = unaveragedBrainCatVals;
    brainR2Vals{sub} = unaveragedBrainR2Vals;
    cornetCatVals{sub} = unaveragedCornetCatVals;
    cornetR2Vals{sub} = unaveragedCornetR2Vals;
    mldsCatVals{sub} = unaveragedMldsCatVals;
    mldsR2Vals{sub} = unaveragedMldsR2Vals;
    bigRoiCatVals{sub} = unaveragedBigRoiCatVals;
    bigRoiR2Vals{sub} = unaveragedBigRoiR2Vals;
    VVSRSMs{sub} = VVSRSM;
    VVSDotProducts{sub} = VVSDotProduct;
    NNCatVals{sub} = unaveragedNNCatVals;
    NNR2Vals{sub} = unaveragedNNR2Vals;

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

catTaskCatVals = brainCatVals(:,4);
catTaskR2Vals = brainR2Vals(:,4);



%% Do some analyses
%brain vs mlds
figure, hold on
for area = 1:3,
    %subplot(1,3,area)
    scatter(brainCatVals((brainR2Vals(:,area) > r2_cutoff) & (mldsR2Vals > r2_cutoff), area) , mldsCatVals((brainR2Vals(:,area) > r2_cutoff) & (mldsR2Vals > r2_cutoff)), 'filled')
    plot([0 1], [0 1], '--k')
end
xlabel('Brain categorical index'), ylabel('Mlds categorical index'), title('Brain and MLDS categorical indices')
legend({'Early', '', 'Middle', '', 'Late'})


%brain vs categorization
figure, hold on
for area = 1:3,
    %subplot(1,3,area)
    scatter(brainCatVals((brainR2Vals(:,area) > r2_cutoff) & (mldsR2Vals > r2_cutoff), area) , brainCatVals((brainR2Vals(:,area) > r2_cutoff) & (mldsR2Vals > r2_cutoff), 4), 'filled')
    plot([0 1], [0 1], '--k')
end
xlabel('Brain categorical index'), ylabel('Mlds categorical index'), title('Brain and MLDS categorical indices')
legend({'Early', '', 'Middle', '', 'Late'})

%mlds vs categorization
figure, hold on
scatter(mldsCatVals(mldsR2Vals > r2_cutoff), brainCatVals((mldsR2Vals > r2_cutoff),4), 'filled')
xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
xlabel('Mlds categorical index'), ylabel('Categorization task categorical index'), title('Mlds and Categorization task categorical indices')



%Whole brain vs mlds
figure, hold on
scatter(mldsCatVals((mldsR2Vals > r2_cutoff) & (bigRoiR2Vals > r2_cutoff)), bigRoiCatVals((mldsR2Vals > r2_cutoff) & (bigRoiR2Vals > r2_cutoff)), 'filled')
xlim([-0.2 1]), ylim([-0.2 1]), plot([0 1], [0 1], '--k')
xlabel('Mlds categorical index'), ylabel('Brain categorical index'), title('Whole brain categorical index vs mlds')


%%
figure
interpSets = {[1:6], [13:18], [19:24]};

x = []; y = [];
for sub = 1:numSubs
    for set = 1:length(interpSets)
        low = min(interpSets{set}); high = max(interpSets{set});
        %dists = VVSRSMs{sub}(low:high,high) - VVSRSMs{sub}(low:high,low); ylabelstr = 'Corr(interp,6) - Corr(interp, 1)';
        dists = VVSDotProducts{sub}(low:high,high) - VVSDotProducts{sub}(low:high,low); ylabelstr = 'Dot(interp,6) - Dot(interp, 1)';

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


%% Make the big plot
figure, hold on, xlim([0.5 4.5]), ylim([-0.2 1.2])

%set cutoffs
catTaskR2Cutoff = 0.91;
mldsR2Cutoff = 0.8;
brainR2Cutoff = 0.7; 
NNR2Cutoff = 0.8;


%plot the models
for model = 1:size(NNR2Vals,3)
    for area = 1:4
        for interp = 1:length(brainR2Vals)
            if (NNR2Vals(interp, area, model) > NNR2Cutoff) 
                scatter(area-0.1, NNCatVals(interp,area,model), 18, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
            end
        end
    end
end


for area = 1:4
    for interp = 1:length(cornetR2Vals)
        if (cornetR2Vals(interp, area) > NNR2Cutoff) 
            scatter(area-0.1, cornetCatVals(interp,area), 18, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.2);
        end
    end
end

%plot brain data
for area = 1:3;
    for interp = 1:length(brainR2Vals)
        if (brainR2Vals(interp, area) > brainR2Cutoff) & (mldsR2Vals(interp) > mldsR2Cutoff) & (catTaskR2Vals(interp) > catTaskR2Cutoff)
            scatter(area+0.1, brainCatVals(interp,area), 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
        end
    end
end


xticks(1:4), xticklabels({'Early', 'Middle', 'Late', 'Behavior'}), xlabel('ROI/Layer')
ylabel('Categorical index')
title('Categorical influence on RSM')

%plot the  mlds data
for interp = 1:length(mldsR2Vals)
    if (mldsR2Vals(interp) > mldsR2Cutoff)
        scatter(4.1, mldsCatVals(interp), 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
    end
end


%plot the categorization data
for interp = 1:length(catTaskR2Vals)
    if (catTaskR2Vals(interp) > catTaskR2Cutoff)
        scatter(4.1, catTaskCatVals(interp), 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5);
    end
end


%legend
h1 = scatter(nan, nan, 18, 'filled', 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Neural network');
h2 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Brain');
h3 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [0 1 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'MLDS');
h4 = scatter(nan, nan, 36, 'filled', 'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Categorization task');
legend([h1, h2, h3, h4]);










%%
keyboard


