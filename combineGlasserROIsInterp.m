function v = combineGlasserROIsInterp(v)
% creates combined ROIs from the glasser atlas to use. for example, combines pha1 pha2 and pha3 into pha123. 
% open up a view with all the glasser rois, and just run it (with view input). should save out the desired rois.


v = combineROIs(v,'lh_VMV1','lh_VMV2','union','lh_VMV12')
v = combineROIs(v,'lh_VMV12','lh_VMV3','union','lh_VMV123')

v = combineROIs(v,'lh_PHA1','lh_PHA2','union','lh_PHA12')
v = combineROIs(v,'lh_PHA12','lh_PHA3','union','lh_PHA123')

v = combineROIs(v,'lh_TE1a','lh_TE1m','union','lh_TE1am')
v = combineROIs(v,'lh_TE1am','lh_TE1p','union','lh_TE1amp')

v = combineROIs(v,'lh_TE2a','lh_TE2p','union','lh_TE2ap')

v = combineROIs(v,'lh_LO1','lh_LO2','union','lh_LO12')
v = combineROIs(v,'lh_LO12','lh_LO3','union','lh_LO123')


v = combineROIs(v,'rh_VMV1','rh_VMV2','union','rh_VMV12')
v = combineROIs(v,'rh_VMV12','rh_VMV3','union','rh_VMV123')

v = combineROIs(v,'rh_PHA1','rh_PHA2','union','rh_PHA12')
v = combineROIs(v,'rh_PHA12','rh_PHA3','union','rh_PHA123')

v = combineROIs(v,'rh_TE1a','rh_TE1m','union','rh_TE1am')
v = combineROIs(v,'rh_TE1am','rh_TE1p','union','rh_TE1amp')

v = combineROIs(v,'rh_TE2a','rh_TE2p','union','rh_TE2ap')

v = combineROIs(v,'rh_LO1','rh_LO2','union','rh_LO12')
v = combineROIs(v,'rh_LO12','rh_LO3','union','rh_LO123')


