% Main function entry for evaluating on ETHZ dataset
%
% Created by Rui Zhao, rzhao@ee.cuhk.edu.hk
% This code is release under research-only license, please cite the paper
%
% Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Unsupervised Salience Learning
% for Person Re-identification. In IEEE Conference of Computer Vision and
% Pattern Recognition (CVPR), 2013. 
%
% Remark1: this implementation is a little different than the original
% version in the training / testing partition, so that the result may vary
% a little, if you use the default settings and parameters, you are
% supposed to obtain the results for the first trial:
% 25.32% (SDC_knn) and 27.22% (SDC_ocsvm). 
%
% Remark2:
%   this demo was tested on MATLAB (R2010b), 64-bit Win7, Intel Xeon 3.30 GHz CPU
%   testing on ETHZ (seq1) dataset would consume around 5.0 GB memory
%   testing on ETHZ (seq2) dataset would consume around 1.6 GB memory
%   testing on ETHZ (seq3) dataset would consume around 1.4 GB memory
%

clear all;
global dataset baseExp gridstep patchsize par

par = struct(...
    'dataset',              'ethz1', ... 'ethz1', 'ethz2', and 'ethz3'
    'baseExp',              'unsupervised_salience', ...
    'gridstep',             4, ... 
    'patchsize',            10, ...
    'sigma1',               1.6, ...
    'msk_thr',              0.2, ...
    'norm_data',            0, ...
    'new_feat',             1, ...
    'use_salience',         2, ...
    'alpha',                [-1, 0.4, 1, 0.6, 0.5], ...
    'L2',                   0 ...
    );

% par.alpha used when testing with OCSVM
% ethz1: [-1.26, 0.2, 0.75, 1, 0.6]
% ethz2: [-0.33, 0.2, 0.4, 3, 0.6]
% ethz3: [-1.26, 0.2, 0.75, 1, 0.6]

dataset     = par.dataset;
baseExp     = par.baseExp;
gridstep    = par.gridstep;
patchsize   = par.patchsize;

if par.L2 
    par.nor = 2;
else
    par.nor = 1;
end
nor  = par.nor;

project_dir = strcat(pwd, '\');
set_paths;
if par.norm_data
    norm_data;
end
initialcontext_general;

%% extract dense feature
if par.new_feat
    build_densefeature_general;
end

%% load all features for testing
features = zeros(dim, ny*nx, ttsize);
hwait = waitbar(0, 'Loading data for testing ...');
for i = 1:ttsize
    load([feat_dir, 'feat', num2str(i), '.mat'])
    features(:, :, i) = densefeat;
    waitbar(i/ttsize, hwait);
end
close(hwait);

%% patch matching
if ~exist([pwdist_dir, 'pwmap', num2str(1), '.mat'], 'file')
    hwait = waitbar(0, 'Computing patching matching ...');
    for i = 1:ttsize
        i
        densefeat = features(:, :, i);
        for j = 1:ttsize
            [pwmap(j).forward, pwmap(j).fpos] = ...
                mutualmap(densefeat, features(:, :, j));
        end
        save([pwdist_dir, 'pwmap', num2str(i), '.mat'], 'pwmap');
        waitbar(i/ttsize, hwait, 'Computing patching matching ...');
    end
    close(hwait);
end

%% compute salience
switch par.use_salience
    case 0
        
    case 1 % KNN salience
        if exist([salience_dir, 'maxdist_knn.mat'], 'file')
            
            load([salience_dir, 'maxdist_knn.mat']);
            
        else
            hwait = waitbar(0, 'computing knn salience ...');
            for i = 1:ttsize
                load([pwdist_dir, 'pwmap', num2str(i), '.mat']);
                cellmap = struct2cell(pwmap);
                dists = cell2mat(cellmap(1, 1, :));
                pid = str2double(perIds{i});
                index = [];
                for p = 1:length(pidx)
                    nIm = length(pidx{p});
                    if p ~= pid
                        rp = randperm(nIm);
                        index(p) = pidx{p}(rp(1));
                    else
                        temp = setdiff(pidx{p}, i);
                        rp = randperm(nIm-1);
                        index(p) = temp(rp(1));
                    end
                end
                rdists = sort(dists(:, :, index), 3);
                maxdist(:, :, i) = rdists(:, :, floor(nPerson/2));
                waitbar(i/ttsize, hwait);
            end
            save([salience_dir, 'maxdist_knn.mat'], 'maxdist');
            close(hwait);
        end
        
    case 2 % OCSVM salience
        if exist([salience_dir, 'maxdist_ocsvm.mat'], 'file')
            
            load([salience_dir, 'maxdist_ocsvm.mat']);
            
        else
            hwait = waitbar(0, 'computing ocsvm salience ...');
            for i = 1:ttsize
                load([pwdist_dir, 'pwmap', num2str(i), '.mat']);
                cellmap = struct2cell(pwmap);
                dists = cell2mat(cellmap(1, 1, :));
                fpos = squeeze(cellmap(2, 1, :));
                pid = str2double(perIds{i});
                index = [];
                for p = 1:length(pidx)
                    nIm = length(pidx{p});
                    if p ~= pid
                        rp = randperm(nIm);
                        index(p) = pidx{p}(rp(1));
                    else
                        temp = setdiff(pidx{p}, i);
                        rp = randperm(nIm-1);
                        index(p) = temp(rp(1));
                    end
                end
                xnn = {};
                for j = 1:length(index)
                    load([feat_dir, 'feat', num2str(index(j)), '.mat']);
                    feat_cell = reshape(mat2cell(densefeat, dim, ones(1, ny*nx)), ny, nx);
                    P = fpos{index(j)};
                    xnn = cat(3, xnn, feat_cell(sub2ind([ny, nx], repmat((1:ny)', 1, nx), double(P))));
                end
                xnn = reshape(xnn, ny*nx, length(index));
                dists_tmp = reshape(dists(:, :, index), ny*nx, length(index));
                maxidx = zeros(1, ny*nx);
                parfor j = 1:ny*nx
                    X = cell2mat(xnn(j, :))';
                    [~, maxidx(j)] = ocsvm_max(X);
                end
                maxdist(:, i) = dists_tmp(sub2ind([ny*nx, length(index)], 1:ny*nx, maxidx));
                waitbar(i/ttsize, hwait);
            end
            maxdist = reshape(maxdist, ny, nx, ttsize);
            save([salience_dir, 'maxdist_ocsvm.mat'], 'maxdist');
            close(hwait);
        end
        
    otherwise
        error('unknown option');
end

% normalize salience map
lwdist = min(maxdist(:));
updist = max(maxdist(:));
maxdist_norm = (maxdist-lwdist)./(updist-lwdist);
salience = squeeze(mat2cell(maxdist_norm, ny, nx, ones(ttsize, 1)))';
features = squeeze(mat2cell(features, dim, ny*nx, ones(ttsize, 1)))';

hwait = waitbar(0, 'Computing matching scores ...');
pwdist_all = zeros(ttsize, ttsize);
for i = 1:ttsize
    load([pwdist_dir, 'pwmap', num2str(i), '.mat']);
    cellmap = struct2cell(pwmap);
    dists = squeeze(cellmap(1, 1, :))';
    fpos = squeeze(cellmap(2, 1, :))';
    pwdist_all(:, i)  = cellfun(@(f, s) ...
        ethzscorefun(features{i}, f, salience{i}, s, par), ...
        features, salience); % [gallery][probe]
    waitbar(i/ttsize, hwait);
end
close(hwait);

%% combine with other features and evaluation
pwdist_cmb{1} = pwdist_all;
load([pwdist_dir, 'disty.mat']);
pwdist_cmb{2} = pwdist;
load([pwdist_dir, 'color.mat']);
pwdist_cmb{3} = pwdist;
load([pwdist_dir, 'hist.mat']);
pwdist_cmb{4} = pwdist;
load([pwdist_dir, 'epitext.mat']);
pwdist_cmb{5} = pwdist;
nTrial = 10;
CMC = SShotEval_cmb(pwdist_cmb, par.alpha, nTrial, pidx);
fprintf('%2.2f%% at rank1\n', 100*CMC(1));
plot(100*CMC, '-bo'); axis([1, 7, 60, 100]);

