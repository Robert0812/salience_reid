% set paths for specific folders
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


% addpath
addpath(genpath(strcat(project_dir, 'code\')));

% load test dataset
dataset_dir = strcat(project_dir, 'dataset\', dataset, '\');

% mat file directory
mat_dir = strcat(project_dir, 'mat\');

% cache directory
cache_dir = strcat(project_dir, 'cache\', dataset, '\');

% normalized data
dnorm_dir = strcat(cache_dir, 'datanorm\');

% dense feature directory
feat_dir = strcat(cache_dir, 'dfeat\');

% mutual distance map set
pwdist_dir = strcat(cache_dir, 'pwdist\');

% salience directory
salience_dir = strcat(cache_dir, 'salience\');

% result directory
result_dir = strcat(cache_dir, 'result\');

% reid directory
reid_dir = strcat(result_dir, 'reid\');

if ~exist(cache_dir, 'dir')
    
    % create directories
    mkdir(cache_dir);
    mkdir(dnorm_dir);
    mkdir(feat_dir);
    mkdir(pwdist_dir);
    mkdir(salience_dir);
    mkdir(result_dir);
    mkdir(reid_dir);
   
    % copy .mat files to specific directory
    switch lower(dataset)
        case 'viper'
            copyfile([mat_dir, 'partition_viper.mat'], cache_dir);
            copyfile([mat_dir, 'posemask_viper.mat'], salience_dir);
            copyfile([mat_dir, 'MSCRmatch_VIPeR_f1_Exp007.mat'], pwdist_dir);
            copyfile([mat_dir, 'txpatchmatch_VIPeR_f1_Exp007.mat'], pwdist_dir);
            copyfile([mat_dir, 'wHSVmatch_VIPeR_f1_Exp007.mat'], pwdist_dir);
            
        case {'ethz1', 'ethz2', 'ethz3'}
    end

end

