for imNum = 1:17
    im1 = imread(strcat(name1{imNum}(1:end-1), '.jpg'));
    im2 = imread(strcat(name2{imNum}, '.jpg'));
    
    im3 = imhistmatch(im1,im2);
    im4 = imhistmatch(im2,im1);
    
    figure
    montage({im1, im2, im3, im4})
end



im1 = imhistmatch(im1,im2);
imwrite(im1, strcat(name1{imNum}(1:end-1), '.jpg'))

im2 = imhistmatch(im2,im1);
imwrite(im2, strcat(name2{imNum}, '.jpg'))




name1 = {'acorns_', 'b1_', 'canopy_', 'fern_', 'fern_', 'grass_', 'lemons_', 'moss2_', 'moss3_', 'mango_', 'obsidian_', 'ocean_', 'pebbles2_', 'pine_', 'pine2_', 'redwood_', 'rubies_'}
name2 = {'redwood', 'b6', 'moss', 'grass', 'leaves', 'leaves', 'bananas', 'grass', 'leaves', 'yellowGems', 'licorice', 'sky', 'granite', 'grass', 'moss', 'rockwall', 'cherries'}
