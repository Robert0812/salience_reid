% initialize the contextual settings
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

files = dir([dnorm_dir, '*.png']);
fnames = {files.name};
array_all = cell2mat(fnames');
% '0001001.png'
cell_all = mat2cell(array_all, ones(1, length(fnames)), [4, 3, 4]);
perIds = cell_all(:, 1);
camIds = cell_all(:, 2);

nPerson = length(unique(perIds));
ttsize = size(cell_all, 1);

switch dataset
    
    case 'viper'
        
        % read in the identity, camera view information
        n = 0;
        oldId = '';
        for i = 1:ttsize
            if ~strcmp(perIds{i}, oldId)
                n = n+1;
                switch str2double(camIds{i})
                    case 1
                        dset(n).idx_cam1 = i;
                        dset(n).idx_cam2 = [];
                    case 2
                        dset(n).idx_cam1 = [];
                        dset(n).idx_cam2 = i;
                end
                oldId = perIds{i};
                
            else
                switch str2double(camIds{i})
                    case 1
                        dset(n).idx_cam1 = [dset(n).idx_cam1 i];
                    case 2
                        dset(n).idx_cam2 = [dset(n).idx_cam2 i];
                end
                
            end
        end
        
        % training / testing partition
        if exist([cache_dir, 'partition_viper.mat'], 'file')
            
            load([cache_dir, 'partition_viper.mat']);
            
        else
            
            for t = 1:10
                in = randperm(length(dset));
                % training set
                for p = 1:floor(length(dset)/2)
                    in1 = randperm(length(dset(in(p)).idx_cam1));
                    in2 = randperm(length(dset(in(p)).idx_cam2));
                    partition(t).trnSet(p, 1) = dset(in(p)).idx_cam1(in1(1));
                    partition(t).trnSet(p, 2) = dset(in(p)).idx_cam2(in2(1));
                end
                % testing set
                ntst = 0;
                for p = (floor(length(dset)/2)+1):length(dset)
                    ntst = ntst + 1;
                    in1 = randperm(length(dset(in(p)).idx_cam1));
                    in2 = randperm(length(dset(in(p)).idx_cam2));
                    partition(t).tstSet(ntst, 1) = dset(in(p)).idx_cam1(in1(1));
                    partition(t).tstSet(ntst, 2) = dset(in(p)).idx_cam2(in2(1));
                end
            end
            save([cache_dir, 'partition_viper.mat'], 'partition');
            
        end
        
        tstSet = partition(TRIAL).tstSet;
        S1 = tstSet(:, 1);
        S2 = tstSet(:, 2);
        gsize = length(S1);
        trnSet = partition(TRIAL).trnSet;
        trnS1 = trnSet(:, 1);
        trnS2 = trnSet(:, 2);
        tsize = length(trnS1);
        
    case {'ethz1', 'ethz2', 'ethz3'}
     
        if exist([cache_dir, 'ididx.mat'], 'file')
            
            load([cache_dir, 'ididx.mat']);
            
        else
            n = 0;
            oldId = '';
            for i = 1:ttsize
                if ~strcmp(perIds{i}, oldId)
                    n = n+1;
                    pidx{n} = i;
                    oldId = perIds{i};
                    
                else
                    pidx{n} = [pidx{n}, i];
                end
            end
            save([cache_dir, 'ididx.mat'], 'pidx');
        end
        
end

% image and feature information
[h, w, ~] = size(imread([dnorm_dir, files(1).name]));
% feature information
feat1 = strcat(cache_dir, 'dfeat\feat1.mat');
if exist(feat1, 'file')
    load(feat1);
    [dim, ~] = size(densefeat);
end

% partition image into dense local patches
global nx;
global ny;
nx = length(patchsize/2:gridstep:w-patchsize/2);
ny = length(patchsize/2:gridstep:h-patchsize/2);
grid_x = ceil(linspace(patchsize/2, w-patchsize/2, nx));
grid_y = ceil(linspace(patchsize/2, h-patchsize/2, ny));
X = repmat(grid_x, ny, 1);
Y = repmat(grid_y', 1, nx);
gridxy = [X(:), Y(:)];


