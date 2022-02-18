clc;
clear;
close all;
seed = 0;
rng(seed);

% import two images
I1 = rgb2gray(imread('.\images2\HG\1.jpg'));
I2 = rgb2gray(imread('.\images2\HG\4.jpg'));
% I1 = rgb2gray(imread('.\test\images3\1.jpg'));
% I2 = rgb2gray(imread('.\test\images3\2.jpg'));
I1 = fliplr(I1');
I2 = fliplr(I2');

% select points

% figure;
% imshow(I1);
% [x1,y1] = getpts();
% % Click "Enter" button to complete point (feature) selection
% matchedPoints1=[x1,y1];
% 
% %%
% figure;
% imshow(I2);
% [x2,y2] = getpts();
% matchedPoints2=[x2,y2];

status = false;
if status
    [mp, fp] = cpselect(I1, I2, 'wait', true);
else
    load("manual selection results.mat");
end
disp("Manual selection results:")

%% plot the correspondance
matchedPoints1 = mp;
matchedPoints2 = fp;

figure; ax = axes;
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
title(ax, 'Correspondences found manually');
legend(ax, 'Matched points 1','Matched points 2');

%%
% homography - maps inliers in I1 to inliers in I2 - X2 = H' * x1 - MSAC
% for outlier
[tform, inlierpoints1, inlierpoints2] = estimateGeometricTransform(matchedPoints1, matchedPoints2, 'projective');
% Homography matrix:
disp("Homography matrix")
H = tform.T

% plot matched correspondance points
figure; ax = axes;
showMatchedFeatures(I1,I2,inlierpoints1,inlierpoints2,'montage','Parent',ax);
title(ax, 'Inlier matches');
legend(ax, 'Inlier points 1','Inlier points 2');
disp("Number of correspondences: " + length(matchedPoints1));
disp("Number of inliers: " + length(inlierpoints1));

pointsImg1 = inlierpoints1;
pointsImg2 = inlierpoints2;

z_axis = ones(length(pointsImg2(:,1)), 1);
pn1 = [pointsImg1 z_axis];
pn2 = [pointsImg2 z_axis];

pn1_tr = pn1.';
H2 = H.';

%  project I1 to I2 - x2_1 = H' * x1
projectI1toI2 = zeros(length(z_axis), 3);
for c = 1:length(z_axis)
    projectI1toI2(c,1) = H2(1,:) * pn1_tr(:,c);
    projectI1toI2(c,2) = H2(2,:) * pn1_tr(:,c);
    projectI1toI2(c,3) = H2(3,:) * pn1_tr(:,c);
end

% MSE can also be computed by sum(sum((pn2-projectI1toI2).^2)) / (numel(pn2))
MSE = immse(double(pn2), projectI1toI2); % error from proj of 2 to 1
total_number_of_pixels = 65.535 * 65.535;
MSE_normalised = MSE ./ total_number_of_pixels;
fprintf("MSE: %.10f\n", MSE);
fprintf("MSE per pixel: %.10f\n", MSE_normalised);