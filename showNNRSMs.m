function showNNRSMs

load('~/data/interp/allStimulicorrMatricesCandidateModelsSoftmaxedMasked.mat')

EVC = clip_largepatch.layer_1;
MVC = clip_largepatch.layer_3;
VVS = clip_largepatch.layer_4;
choice = clip_largepatch.layer_6;
% 
% EVC = vgg19.layer_1;
% MVC = vgg19.layer_3;
% VVS = vgg19.layer_4;
% choice = vgg19.layer_6;


interpSets = {[1:6], [7: 12], [13:18], [19:24], [25:30], [31:36], [37:42], [43:48]};



keyboard
%%



plotMatrices(choice/8,interpSets)
%%




function plotMatrices(rsm, interpSets)
figure
averageRSM = zeros(6);
for interpSet = 1:length(interpSets)
    subplot(3,3,interpSet)
    imagesc(rsm(interpSets{interpSet},interpSets{interpSet})), colormap('hot')
    averageRSM = averageRSM + rsm(interpSets{interpSet},interpSets{interpSet});
end

figure, imagesc(averageRSM), colormap('hot')