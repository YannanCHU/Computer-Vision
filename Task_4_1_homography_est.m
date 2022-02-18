clc;
clear;
close all;

% Programed by yannan chu and shilei wang in 2022 Feb

I1 = rgb2gray(imread('.\images2\HG\1.jpg'));
I2 = rgb2gray(imread('.\images2\HG\4.jpg'));
% I1 = rgb2gray(imread('.\test\images3\1.jpg'));
% I2 = rgb2gray(imread('.\test\images3\2.jpg'));
I1 = fliplr(I1');
I2 = fliplr(I2');

% Detect festures (e.g., corners)
% Detect corners using Harris¨CStephens algorithm
% points1 = detectHarrisFeatures(I1);
% points2 = detectHarrisFeatures(I2);

% Detect scale invariant feature transform (SIFT) features 
points1 = detectSIFTFeatures(I1);
points2 = detectSIFTFeatures(I2);

% Detect corners using minimum eigenvalue algorithm and return cornerPoints object
% points1 = detectMinEigenFeatures(I1);
% points2 = detectMinEigenFeatures(I2);

% Detect corners using minimum eigenvalue algorithm
% points1 = detectSURFFeatures(I1);
% points2 = detectSURFFeatures(I2);

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

figure; ax = axes;
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
title(ax, 'Candidate point matches');
legend(ax, 'Matched points 1','Matched points 2');

%%
[tform,inlierIdx] = estimateGeometricTransform2D(matchedPoints2, matchedPoints1, 'projective');
inlierPtsDistorted = matchedPoints2(inlierIdx,:);
inlierPtsOriginal  = matchedPoints1(inlierIdx,:);

figure 
showMatchedFeatures(I1, I2, inlierPtsOriginal, inlierPtsDistorted)
title('Matched SIFT Inlier Points')
disp("Homography Matrix estimated by estimateGeometricTransform2D() function: ");
disp(tform.T);

% [tform,inlierPts2,inlierPts1] = ...
%     estimateGeometricTransform2D(matchedPoints2,matchedPoints1,...
%     'projective');
% 
% figure();
% showMatchedFeatures(I1,I2,...
%     inlierPts1,inlierPts2);
% title('Matched SURF points,including outliers');
% display(tform.T);

%% Solve equations using SVD
n = size(matchedPoints1.Location, 1);
x = matchedPoints1.Location(:,1); y = matchedPoints1.Location(:,2); 
X = matchedPoints2.Location(:,1); Y = matchedPoints2.Location(:,2);
A = zeros(n*2, 9);
A(1:2:end, 1) = x;
A(2:2:end, 4) = x;
A(1:2:end, 2) = y;
A(2:2:end, 5) = y;
A(1:2:end, 3) = ones(n,1);
A(2:2:end, 6) = ones(n,1);
A(1:2:end, 7) = -x.*X;      A(2:2:end, 7) = -x.*Y;
A(1:2:end, 8) = -y.*X;      A(2:2:end, 8) = -y.*Y;
A(1:2:end, 9) = -X;         A(2:2:end, 9) = -Y;

[U,S,V] = svd(A'*A, 'econ');
f = V(:,9) / V(9,9);
F = reshape(f, 3, 3).';

disp("Homography Matrix estimated by SCD based method: ");
disp(F);