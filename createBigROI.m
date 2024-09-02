function v = createBigROI(v)
% Used to show you coverage of many different ROIs. Takes all the ROIs you put in and just combines them.
% You need to make an empty ROI to begin with called 'bigroi'.
% Specifcy the rois you want to combine in 'allrois'.

allrois = {'bigroi','lh_V1', 'rh_V1', 'lh_V2', 'rh_V2', 'lh_V3', 'rh_V3', 'lh_V3a', 'rh_V3a', 'lh_V3b', 'rh_V3b', 'lh_V3CD', 'rh_V3CD', 'lh_v6', 'rh_v6', 'lh_v6a', 'rh_v6a', ...
                    'lh_V4', 'rh_V4', 'lh_v4t', 'rh_v4t', 'lh_LO123', 'rh_LO123', ...
                    'lh_VVC', 'rh_VVC', 'lh_VMV123', 'rh_VMV123', 'lh_TF', 'rh_TF', 'lh_PIT', 'rh_PIT', 'lh_V8', 'rh_V8', 'lh_EC', 'rh_EC', 'lh_PeEc', 'rh_PeEc',...
                    'lh_TGv', 'rh_TGv', 'lh_6v', 'rh_6v', 'lh_FFC', 'rh_FFC', 'lh_v23ab', 'rh_v23ab', 'lh_PH', 'rh_PH',...
                    'lh_PHT', 'rh_PHT', 'lh_MST', 'rh_MST', 'lh_MT', 'rh_MT', 'lh_PreS', 'rh_PreS', 'lh_PHA123', 'rh_PHA123', 'lh_TE1amp', 'rh_TE1amp','lh_TE2ap', 'rh_TE2ap'}

for roi = 1:length(allrois)
    v = combineROIs(v,'bigroi',allrois{roi},'union')
end