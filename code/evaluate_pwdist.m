%
% Compute the CMC statistics to evaluate the performance of person re-identification
% Cumulative Matching Characteristics (CMC) 
%   Wang, X., et al., Shape and appearance context modeling. In: ICCV 2007.
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
function cmc = evaluate_pwdist(pwdist)
% evaluate the performance of pwdist
% assume gallery in dim1, query in dim2, and param test in dim3
% 
N = size(pwdist, 3);
gsize = size(pwdist, 1);    
cmc = zeros(gsize, N);
for i = 1:N
    [~, order] = sort(pwdist(:, :, i));
    match = (order == repmat(1:gsize, [gsize, 1]));
    cmc(:, i) = cumsum(sum(match, 2)./gsize);
end