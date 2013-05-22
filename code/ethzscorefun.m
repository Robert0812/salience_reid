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

function score = ethzscorefun(feat1, feat2, sa1, sa2, param)
% compute the salience-aided similarity score between two feature
% 
% score = ethzscorefun(feat1, feat2, sa1, sa2, param)
%
% INPUT
%   feat1:      feature 1 [feat][patch]
%   feat2:      feature 2 [feat][patch]
%   sa1:        salience map1
%   sa2:        salience map2
%   param:     parameters
%
% OUTPUT
%   score:      similarity score
%

D = reshape(sqrt(sum((feat1 - feat2).^2)), size(sa1));
% Affinity = exp(-D.^2/(param.sigma^2));
Affinity = exp(-D.^2/(param.sigma1^2)).*sa1.*sa2;
% Affinity = exp(-D.^2/(param.sigma^2))./(1+abs(sa1-sa2));
% Affinity = exp(-D.^2/(param.sigma^2)).*sa1.*sa2./(1+abs(sa1-sa2));
score = sum(Affinity(:));
% score = norm(D);

% for testing
% score = norm(reshape(feat1, [], 1) - reshape(feat2, [], 1));