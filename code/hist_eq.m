%
% Perform histogram equalization
%
% Created by Rui Zhao, on May 20, 2013. 
% This code is release under BSD license, 
% any problem please contact Rui Zhao rzhao@ee.cuhk.edu.hk
%
% Please cite as
% Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Unsupervised Salience Learning
% for Person Re-identification. In IEEE Conference of Computer Vision and
% Pattern Recognition (CVPR), 2013. 
%

function im_eq = hist_eq(im)
% perform histogram equalization for an image
imhsv = rgb2hsv(im);
V = imhsv(:, :, 3);
imhsv(:, :, 3) = histeq(V);
J = hsv2rgb(imhsv);
im_eq = uint8(255.*J);
% img_hsv = rgb2hsv(im);
% tmp         =   img_hsv(:,:,3);
% tmp         =   histeq(tmp);
% img_hsv     =   cat(3, img_hsv(:,:,1), img_hsv(:,:,2), tmp);
% im_eq = hsv2rgb(img_hsv);