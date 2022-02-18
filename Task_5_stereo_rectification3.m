clc;
clear;
close all;



% Programed by yannan chu and shilei wang in 2022 Feb
%Read the two images.
% rng(0);
% I1 = rgb2gray(imread('./images3/Task 5/left10.jpg'));
% I2 = rgb2gray(imread('./images3/Task 5/right10.jpg'));
% rng(8);
% I1 = rgb2gray(imread('./images3/Task 5/left13.jpg'));
% I2 = rgb2gray(imread('./images3/Task 5/right13.jpg'));
% rng(3);
% I1 = rgb2gray(imread('./images3/Task 5/left.jpg'));
% I2 = rgb2gray(imread('./images3/Task 5/right.jpg'));
rng(18);
I1 = rgb2gray(imread('./images2/FD_object/6.jpg'));
I2 = rgb2gray(imread('./images2/FD_object/2.jpg'));
I1 = fliplr(I1');
I2 = fliplr(I2');


figure();   subplot(121); imshow(I1);   
title("Original image captured by left camera");
subplot(122); imshow(I2);
title("Original image captured by right camera");

%Find the SIFT features.
points1 = detectSIFTFeatures(I1);
points2 = detectSIFTFeatures(I2);

%Extract the features.
[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

%Retrieve the locations of matched points.
indexPairs = matchFeatures(f1,f2);
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

%Display the matching points. The data still includes several outliers, 
%but you can see the effects of rotation and scaling on the display of matched features.
% figure(1); ax = axes;
% showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
% title(ax, 'Candidate point matches');
% legend(ax, 'Matched points 1','Matched points 2');

% Find the fundamental matrix and inliers
[F,inliers] = estimateFundamentalMatrix(matchedPoints1(:),matchedPoints2(:),'Method', 'RANSAC', ...
    'DistanceThreshold', 1);
disp('The Fundamental Matrix is:');
disp(F);

figure(2);
inlier_points1 = matchedPoints1(inliers,:);
inlier_points2 = matchedPoints2(inliers,:);
showMatchedFeatures(I1, I2, inlier_points1, inlier_points2, ...
    'montage','PlotOptions',{'ro','go','y--'});
title('Point matches after outliers were removed');

%Show the inliers in the first image.
figure(3); 
subplot(121);
imshow(I1); 
title('Inliers and Epipolar Lines in First Image'); hold on;
plot(matchedPoints1.Location(inliers,1),matchedPoints1.Location(inliers,2),'go')
%Compute the epipolar lines in the first image.
epiLines = epipolarLine(F',matchedPoints2.Location(inliers,:));
%Compute the intersection pointsfdsafdas of the lines and the image border.
points = lineToBorderPoints(epiLines,size(I1));
%Show the epipolar lines in the first image
line(points(:,[1,3])',points(:,[2,4])');

subplot(122); 
imshow(I2);
title('Inliers and Epipolar Lines in Second Image'); hold on;
plot(matchedPoints2.Location(inliers,1),matchedPoints2.Location(inliers,2),'go')
epiLines = epipolarLine(F,matchedPoints1.Location(inliers,:));
points = lineToBorderPoints(epiLines,size(I2));
line(points(:,[1,3])',points(:,[2,4])');

[isIn1,epipole1] = isEpipoleInImage(F,size(I1));
[isIn2,epipole2] = isEpipoleInImage(F',size(I2));

%% Task 5.1 - stereo rectified pair of your images with epipolar lines
% f = estimateFundamentalMatrix(inlier_points1,inlier_points2,...
%     'Method','Norm8Point');
% Compute the rectification transformations.
[t1, t2] = estimateUncalibratedRectification(F,inlier_points1,...
    inlier_points2,size(I2));
% Rectify the stereo images using projective transformations t1 and t2.
[I1Rect,I2Rect] = rectifyStereoImages(I1,I2,t1,t2);
% Display the stereo anaglyph, which can also be viewed with 3-D glasses.
figure(4);
imshow(stereoAnaglyph(I1Rect,I2Rect));
title("Red-cyan anaglyph from stereo pair of images");

figure(5);
stackedImage = cat(2, I1Rect, I2Rect);
imshow(stackedImage);
title("Rectified images");

%% re-detect the feature
%Find the SIFT features.
points1 = detectSIFTFeatures(I1Rect);
points2 = detectSIFTFeatures(I2Rect);

%Extract the features.
[f1,vpts1] = extractFeatures(I1Rect,points1);
[f2,vpts2] = extractFeatures(I2Rect,points2);

%Retrieve the locations of matched points.
indexPairs = matchFeatures(f1,f2);
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

[F,inliers] = estimateFundamentalMatrix(matchedPoints1(:),matchedPoints2(:), ...
    'Method', 'RANSAC', 'DistanceThreshold', 0.01);

%%
figure(6);
inlier_points1 = matchedPoints1(inliers,:);
inlier_points2 = matchedPoints2(inliers,:);
showMatchedFeatures(I1Rect, I2Rect, inlier_points1, inlier_points2, ...
    'montage','PlotOptions',{'ro','go','y--'});
title('Rectified pair of images');

%% Task 5.2 plot the depth map
% % estimate the range of disparity
% imtool(stereoAnaglyph(I1Rect,I2Rect))
disparityRange = [0 128];
disparityMap = disparitySGM(I1Rect,I2Rect,'DisparityRange',disparityRange,'UniquenessThreshold',1);


%%
baselineLen = 30;
focalLen = 6;
depthmap = (baselineLen * focalLen) ./ disparityMap;

figure(7);
imshow(disparityMap, [min(disparityMap(:)), max(disparityMap(:))]);
title('Disparity Map fould by semi-global matching method (colorbar unit: mm)')
colormap jet
colorbar

figure(8);
imshow(depthmap, [min(depthmap(:)), max(depthmap(:))]);
title('Depth Map (colorbar unit: mm)')
colormap jet
colorbar