clc;
clear;
close all;

%% Programed by yannan chu and shilei wang in 2022 Feb

I1 = rgb2gray(imread('.\images2\HG\1.jpg'));
I2 = rgb2gray(imread('.\images2\HG\4.jpg'));
% I1 = rgb2gray(imread('.\test\images3\1.jpg'));
% I2 = rgb2gray(imread('.\test\images3\2.jpg'));
I1 = fliplr(I1');
I2 = fliplr(I2');

% Detect festures (e.g., corners)
% Detect corners using Harris¨CStephens algorithm
points1 = detectHarrisFeatures(I1);
points2 = detectHarrisFeatures(I2);
disp("Harris method results:");
index = 1;

% Detect scale invariant feature transform (SIFT) features 
% points1 = detectSIFTFeatures(I1);
% points2 = detectSIFTFeatures(I2);
% disp("SIFT method results:");
% index = 2;

% Detect corners using minimum eigenvalue algorithm and return cornerPoints object
% points1 = detectMinEigenFeatures(I1);
% points2 = detectMinEigenFeatures(I2);
% disp("MinEigen method results:");
% index = 3;

% Use Speeded-Up Robust Features (SURF) algorithm to find blob features
% points1 = detectSURFFeatures(I1);
% points2 = detectSURFFeatures(I2);
% disp("SURF method results:");
% index = 4;

% uses the Features from Accelerated Segment Test (FAST) algorithm to find feature points.
% points1 = detectFASTFeatures(I1);
% points2 = detectFASTFeatures(I2);

% uses Maximally Stable Extremal Regions (MSER) algorithm to find regions.
% points1 = detectMSERFeatures(I1);
% points2 = detectMSERFeatures(I2);

% Extract features
[f1, vpts1] = extractFeatures(I1, points1);
[f2, vpts2] = extractFeatures(I2, points2);

% Match features
indexPairs = matchFeatures(f1, f2);

% Retrieve the locations of the corresponding points for each image.
matchedPoints1 = vpts1(indexPairs(:, 1), :);
matchedPoints2 = vpts2(indexPairs(:, 2), :);

%% plot matched correspondance points
figure; ax = axes;
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
legend(ax, 'Matched points 1','Matched points 2');
if index == 1
    title(ax, 'Candidate point matches (found by Harris¨CStephens algorithm)');
elseif index == 2
    title(ax, 'Candidate point matches (found by scale invariant feature transform algorithm)');
elseif index == 3
    title(ax, 'Candidate point matches (found by minimum eigenvalue algorithm)');
elseif index == 4
    title(ax, 'Candidate point matches (found by Speeded-Up Robust Features algorithm)');
end

%% homography - maps inliers in I1 to inliers in I2 - X2 = H' * x1 - MSAC
status = false;
if status
    [tform, inlierpoints1, inlierpoints2] = estimateGeometricTransform(matchedPoints1, matchedPoints2, 'projective');
end
% Homography matrix:
disp("Homography matrix")
H = tform.T

disp("Number of correspondences: " + length(matchedPoints1));
disp("Number of inliers: " + length(inlierpoints1));

% plot matched correspondance points
figure; ax = axes;
showMatchedFeatures(I1,I2,inlierpoints1,inlierpoints2,'montage','Parent',ax);
title(ax, 'Corresponding inlier points in two images');
legend(ax, 'Inlier points 1','Inlier points 2');

pointsImg1 = inlierpoints1.Location;
pointsImg2 = inlierpoints2.Location;

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

MSE = immse(double(pn2), projectI1toI2); % error from proj of 2 to 1
total_number_of_pixels = 65.535 * 65.535;
MSE_normalised = MSE ./ total_number_of_pixels;
fprintf("MSE: %.10f\n", MSE);
fprintf("MSE per pixel: %.10f\n", MSE_normalised);




