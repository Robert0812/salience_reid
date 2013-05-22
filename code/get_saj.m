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
function salience = get_saj(disp1, sa2)
% compute matched salience for gallery image
% 
% INPUT
%   disp1: query to gallery matching displacement
%   sa2:    original salience map of gallery image
%  
% OUTPUT
%   salience: query matched salience 
%

[ny, nx] = size(disp1);
salience = sa2(sub2ind([ny, nx], repmat((1:ny)', 1, nx), double(disp1)));
