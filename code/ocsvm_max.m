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
function [score, ind] = ocsvm_max(xnn)
% estimate density distribution for a set of data xnn, 
% and locate the largest density data point

model = svmtrain(ones(size(xnn, 1), 1), xnn, '-s 2 -n 0.5 -g 0.07 -q');
w = model.SVs'*model.sv_coef;
b = -model.rho;
[score, ind] = max(xnn*w + b);