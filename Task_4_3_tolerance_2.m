clc;
clear;
close all;
seed = 10;
rng(seed);

% Programed by yannan chu and shilei wang in 2022 Feb

I1 = rgb2gray(imread('.\images3\HG\1.jpg'));
I2 = rgb2gray(imread('.\images3\HG\4.jpg'));
% I1 = rgb2gray(imread('.\test\images3\1.jpg'));
% I2 = rgb2gray(imread('.\test\images3\2.jpg'));
I1 = fliplr(I1');
I2 = fliplr(I2');

% figure(1);   
% subplot(121); imshow(I1);   title("First Original Image");
% subplot(122); imshow(I2);   title("Second Original Image");

% get all matched points including both inliers and outliers
% Detect scale invariant feature transform (SIFT) features 
points1 = detectSIFTFeatures(I1);
points2 = detectSIFTFeatures(I2);

% Extract features
[f1, vpts1] = extractFeatures(I1, points1);
[f2, vpts2] = extractFeatures(I2, points2);

% Match features
indexPairs = matchFeatures(f1, f2);

% Retrieve the locations of the corresponding points for each image.
matchedPoints1 = vpts1(indexPairs(:, 1), :);
matchedPoints2 = vpts2(indexPairs(:, 2), :);

% figure(2); ax = axes;
% showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
% title(ax, 'Candidate point matches');
% legend(ax, 'Matched points 1','Matched points 2');

%% Tolerance to the number of outliers
seed = 10;
rng(seed);
% Get the estimated Homography Matrix and corresponding estimated inliers
[tform,inlierIdx] = estimateGeometricTransform2D(matchedPoints1, matchedPoints2, 'projective', 'MaxDistance', 1.5);
inlierPoints1  = matchedPoints1(inlierIdx,:);
inlierPoints2 = matchedPoints2(inlierIdx,:);

seed = 10;
rng(seed);
[~,inlierIdx2] = estimateGeometricTransform2D(matchedPoints1, matchedPoints2, 'projective', 'MaxDistance', 50);
midIndex1 = xor(inlierIdx, inlierIdx2);
outlierPoints1_small = matchedPoints1(midIndex1, :);
outlierPoints2_small = matchedPoints2(midIndex1, :);

seed = 10;
rng(seed);
[~,inlierIdx3] = estimateGeometricTransform2D(matchedPoints1, matchedPoints2, 'projective', 'MaxDistance', 500);
midIndex2 = xor(inlierIdx2, inlierIdx3);
outlierPoints1_large = matchedPoints1(midIndex2, :);
outlierPoints2_large = matchedPoints2(midIndex2, :);
outlierPoints1 = [outlierPoints1_small(1:3); outlierPoints1_large];
outlierPoints2 = [outlierPoints2_small(1:3); outlierPoints2_large];

% set one hundred matched points with varying outlier percentages
outlierNum = 0:1:100;
inlierNum = 100:-1:0;
outlierPer = outlierNum ./ (outlierNum+inlierNum);
outlierMeanStrength = zeros(size(outlierPer));
InlierMeanStrength = zeros(size(outlierPer));
reprojectionError = zeros(size(outlierPer));
mses = zeros(size(outlierPer));
total_number_of_pixels = 65.535 * 65.535;

for i = 1:1:length(outlierNum)
% for i = 72:1:72
%     i = 50;
    % selected outliers
%     outlierIndexArray = randi(length(outlierPoints1),1,outlierNum(i));
%     inlierIndexArray = randi(length(inlierPoints1),1,inlierNum(i));
    outlierIndexArray = 1:outlierNum(i);
    inlierIndexArray = 1:inlierNum(i);

    outlierPtsSel1 = outlierPoints1(outlierIndexArray,:);
    outlierPtsSel2 = outlierPoints2(outlierIndexArray,:);
    inlierPtsSel1 = inlierPoints1(inlierIndexArray,:);
    inlierPtsSel2 = inlierPoints2(inlierIndexArray,:);
%     outlierMeanStrength = mean(outlierPtsSel1.metric);

    in_outliers1 = [outlierPtsSel1; inlierPtsSel1];
    in_outliers2 = [outlierPtsSel2; inlierPtsSel2];

    n = size(in_outliers1.Location, 1);
    x = in_outliers1.Location(:,1); y = in_outliers1.Location(:,2); 
    X = in_outliers2.Location(:,1); Y = in_outliers2.Location(:,2);
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
    H = reshape(f, 3, 3).';
%     disp("Homography Matrix estimated by SCD based method: ");
    disp(H);
    
%     [tformi,inlierIdxi] = estimateGeometricTransform2D(in_outliers1, in_outliers2, 'projective', 'MaxDistance', 0.5, 'MaxNumTrials', 1000);
% %     tformi.T
    % compute the reprojection error according to the inliers
    projectionFromI1 = H * ([inlierPoints1.Location, ones(length(inlierPoints1), 1)]).';
    projectionFromI1 = projectionFromI1.';
%     projectionFromI1(:,[1,2]) = projectionFromI1(:,[1,2]) ./ projectionFromI1(:,3);
%     reprojectionError(i) = mean(sqrt( sum((inlierPoints2.Location - projectionFromI1(:,[1,2])).^2,2) ));
    mses(i) = immse(double(inlierPoints2.Location), double(projectionFromI1(:,[1,2]))) / total_number_of_pixels;
end


% plot normalised MSE error against percentage of outliers
figure();
subplot(211);
semilogy(outlierNum ./ (outlierNum+inlierNum) * 100, mses, outlierNum, ones(length(outlierNum),1), '--', 'LineWidth', 2);
xlabel("Number of outliers");
ylabel("Normalised MSE error")
title("Normalised MSE Error against Number of outliers");


%% Tolerance to the max distance of outliers
maxDistanceArray = [0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 120, 150, 180, 300, 600, 1000];
% maxDistanceArray = [(0.1:0.1:0.9), (1:1:10), (20:10:30)];
% maxDistanceArray = [(1:1:9), (10:10:90), (100:100:300)];
mses2 = zeros(length(maxDistanceArray),1);

for i = 1:length(maxDistanceArray)
    seed = 10;
    rng(seed);
    
    [~, inlierpoints1, inlierpoints2] = estimateGeometricTransform(matchedPoints1, matchedPoints2, 'projective' ...
        , 'MaxDistance', maxDistanceArray(i), 'MaxNumTrials', 1000);
    % tform_i.T
    pointsImg1 = inlierPoints1.Location;
    pointsImg2 = inlierPoints2.Location;
    
    z_axis = ones(length(pointsImg2(:,1)), 1);
    pn1 = [pointsImg1 z_axis];
    pn2 = [pointsImg2 z_axis];
    
    pn1_tr = pn1.';
    
    % SVD based Homography matrix estimation
    n = size(inlierpoints1.Location, 1);
    x = inlierpoints1.Location(:,1); y = inlierpoints1.Location(:,2); 
    X = inlierpoints2.Location(:,1); Y = inlierpoints2.Location(:,2);
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
    H2 = reshape(f, 3, 3).';
%     H2 = tform_i.T.';
    


    projectI1toI2 = zeros(length(z_axis), 3);
    for c = 1:length(z_axis)
        projectI1toI2(c,1) = H2(1,:) * pn1_tr(:,c);
        projectI1toI2(c,2) = H2(2,:) * pn1_tr(:,c);
        projectI1toI2(c,3) = H2(3,:) * pn1_tr(:,c);
    end
    
    MSE = immse(double(pn2), projectI1toI2); % error from proj of 2 to 1
    total_number_of_pixels = 65.535 * 65.535;
    mses2(i) = MSE ./ total_number_of_pixels;
end
%%
subplot(212);
loglog(maxDistanceArray, mses2, maxDistanceArray, ones(size(mses2)), '--', 'LineWidth', 2);
title("Normalised MSE Error against Distance of outliers");
xlabel("Maximum distance from point to projection");
ylabel("Normalised MSE Error");
