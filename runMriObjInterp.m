function runMriObjInterp

%% first, get all the data
subs = ['sub=s0605'; 'sub=s0606'; 'sub=s0607']
numSubs = size(subs,1)

for sub = 1:numSubs

    %get the values
    [unaveragedBrainCatVals, unaveragedBrainR2Vals, unaveragedCornetCatVals, unaveragedCornetR2Vals, unaveragedMldsCatVals, unaveragedMldsR2Vals, unaveragedBigRoiCatVals, unaveragedBigRoiR2Vals] = mriObjInterp(subs(sub,:));

    %save them
    brainCatVals{sub} = unaveragedBrainCatVals;
    brainR2Vals{sub} = unaveragedBrainR2Vals;
    cornetCatVals{sub} = unaveragedCornetCatVals;
    cornetR2Vals{sub} = unaveragedCornetR2Vals;
    mldsCatVals{sub} = unaveragedMldsCatVals;
    mldsR2Vals{sub} = unaveragedMldsR2Vals;
    bigRoiCatVals{sub} = unaveragedBigRoiCatVals;
    bigRoiR2Vals{sub} = unaveragedBigRoiR2Vals;

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
keyboard


