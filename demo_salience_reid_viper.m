% Main function entry for evaluating on VIPeR dataset
%
% Created by Rui Zhao, rzhao@ee.cuhk.edu.hk
% This code is release under BSD license, please cite our paper if you use this code:
%
% Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Unsupervised Salience Learning
% for Person Re-identification. In IEEE Conference of Computer Vision and
% Pattern Recognition (CVPR), 2013. 
%


clear all; clc;
test_trial = 1;

global dataset baseExp TRIAL gridstep patchsize par

par = struct(...
    'dataset',                      'viper', ... % 'viper'
    'baseExp',                      'unsupervised_salience', ...
    'method',                       'salience', ... % 'patchmatch', 'salience' ...
    'TRIAL',                        test_trial, ... % average over 10 trials to obtain stable result
    'gridstep',                     4, ...
    'patchsize',                    10, ...
    'Nr',                           100, ... 
    'sigma1',                       2.8, ...
    'msk_thr',                      0.2, ...
    'norm_data',                    0, ...
    'new_feat',                     0, ...
    'use_mask',                     1, ...
    'use_salience',                 2, ... % set 1 to use knn salience, and set 2 to use ocsvm salience
    'alpha',                        [-1, 0.4, 1, 0.6, 0],  ... %[-1, 0.4, 1, 0.6, 0], ... %
    'L2',                           1, ...
    'swap',                         1 ...
    );

dataset     = par.dataset;
baseExp     = par.baseExp;
TRIAL       = par.TRIAL;
gridstep    = par.gridstep;
patchsize   = par.patchsize;

if par.L2 
    par.nor = 2;
else
    par.nor = 1;
end
nor  = par.nor;

switch par.method
    
    case 'patchmatch'
        phiFun = @(x, y, m1, s1, s2, m2, par) (exp(-double(x).^2/par.sigma1^2).*m1).^nor;
        par.use_salience = 0;
        
    case 'salience'
        phiFun = @(x, y, m1, s1, s2, m2, par) s1.*((exp(-double(x).^2/par.sigma1^2).*m1).^nor).*get_saj(y, s2);
        
end

project_dir = strcat(pwd, '\');
set_paths;
if par.norm_data
    norm_data;
end
initialcontext_general;

if par.swap
    trnSg = trnS1;
    trnSp = trnS2;
    Sg = S1;
    Sp = S2;
else
    trnSg = trnS2;
    trnSp = trnS1;
    Sg = S2;
    Sp = S1;
end


%% extract dense feature
if par.new_feat
    build_densefeature_general;
end


%% Compute patch match for computing salience
clear ref_pwmap_prb ref_pwmap_gal;
Nr = par.Nr;

if ~exist([pwdist_dir, 'pwmap_ref_trial', num2str(TRIAL), '.mat'], 'file')
    
    % load testing data
    feat_gal = zeros(dim, ny*nx, gsize);
    feat_prb = zeros(dim, ny*nx, gsize);
    hwait = waitbar(0, 'Loading data for testing ...');
    for i = 1:gsize
        load([feat_dir, 'feat', num2str(Sg(i)), '.mat']);
        feat_gal(:, :, i) = densefeat;
        load([feat_dir, 'feat', num2str(Sp(i)), '.mat']);
        feat_prb(:, :, i) = densefeat;
        waitbar(i/gsize, hwait);
    end
    
    % load reference data
    ref_feat_gal = zeros(dim, ny*nx, Nr);
    ref_feat_prb = zeros(dim, ny*nx, Nr);
    hwait = waitbar(0, 'Loading reference data ...');
    for i = 1:Nr
        load([feat_dir, 'feat', num2str(trnSg(i)), '.mat']);
        ref_feat_gal(:, :, i) = densefeat;
        load([feat_dir, 'feat', num2str(trnSp(i)), '.mat']);
        ref_feat_prb(:, :, i) = densefeat;
        waitbar(i/Nr, hwait);
    end

    % patch match with reference data for computing salience
    for i = 1:gsize
        for j = 1:Nr
            [ref_pwmap_prb(i, j).forward, ref_pwmap_prb(i, j).fpos, ~] = ...
                mutualmap(feat_prb(:, :, i), ref_feat_gal(:, :, j));
            [ref_pwmap_gal(i, j).forward, ref_pwmap_gal(i, j).fpos, ~] = ...
                mutualmap(feat_gal(:, :, i), ref_feat_prb(:, :, j));
            waitbar(j/Nr, hwait, ['Computing mutual map with reference sample ', ...
                num2str(i), ' out of ', num2str(gsize), '...']);
        end
    end
    
    save([pwdist_dir, 'pwmap_ref_trial', num2str(TRIAL), '.mat'], 'ref_pwmap_prb', 'ref_pwmap_gal');
    close(hwait);
    
else
    load([pwdist_dir, 'pwmap_ref_trial', num2str(TRIAL), '.mat']);
end


%% Compute patch match for testing
clear pwmap_prb pwmap_gal;

if ~exist([pwdist_dir, 'pwmap_test_trial', num2str(TRIAL), '.mat'], 'file')
    
    % load testing data
    feat_gal = zeros(dim, ny*nx, gsize);
    feat_prb = zeros(dim, ny*nx, gsize);
    hwait = waitbar(0, 'Loading data for testing ...');
    for i = 1:gsize
        load([feat_dir, 'feat', num2str(Sg(i)), '.mat']);
        feat_gal(:, :, i) = densefeat;
        load([feat_dir, 'feat', num2str(Sp(i)), '.mat']);
        feat_prb(:, :, i) = densefeat;
        waitbar(i/gsize, hwait);
    end

    % patch match for testing
    for i = 1:gsize
        for j = 1:gsize
            [pwmap_prb(i, j).forward, pwmap_prb(i, j).fpos, ~] = ...
                mutualmap(feat_prb(:, :, i), feat_gal(:, :, j));
            [pwmap_gal(i, j).forward, pwmap_gal(i, j).fpos, ~] = ...
                mutualmap(feat_gal(:, :, i), feat_prb(:, :, j));
            waitbar(j/gsize, hwait, ['Computing mutual map for sample ', ...
                num2str(i), ' out of ', num2str(gsize), '...']);
        end
    end
    
    save([pwdist_dir, 'pwmap_test_trial', num2str(TRIAL), '.mat'], 'pwmap_prb', 'pwmap_gal');
    close(hwait);
    
else
    load([pwdist_dir, 'pwmap_test_trial', num2str(TRIAL), '.mat']);
end


%% convert pairwise distances and displacement matrix to cells for computation convenience

pwmap_tst_cell = struct2cell(pwmap_prb);
D_cell = squeeze(pwmap_tst_cell(1, :, :));
P_cell = squeeze(pwmap_tst_cell(2, :, :));


%% load mask produced by pose estimation

if par.use_mask
    load([salience_dir, 'posemask_viper.mat']);
    mask_prb = squeeze(mat2cell(mask(:, :, Sp) >= par.msk_thr, ny, nx, ones(1, length(Sp))));
    mp_cell = repmat(mask_prb, 1, gsize);
    mask_gal = squeeze(mat2cell(mask(:, :, Sg) >= par.msk_thr, ny, nx, ones(1, length(Sg))));
    mg_cell = repmat(mask_gal', gsize, 1);
end


%% compute salience (knn or ocsvm)
switch par.use_salience
    case 0
        sg_cell = cell(gsize, gsize);
        sp_cell = cell(gsize, gsize);
        
    case 1 % KNN salience
        
        if ~exist([salience_dir, 'knn_salience_trial', num2str(TRIAL), '.mat'], 'file')
            
            cellmap_gal = struct2cell(ref_pwmap_gal);
            cellmap_prb = struct2cell(ref_pwmap_prb);
            dists_gal = cell2mat(reshape(cellmap_gal(1, :, :), 1, 1, gsize, Nr));
            dists_prb = cell2mat(reshape(cellmap_prb(1, :, :), 1, 1, gsize, Nr));
            rdists_gal = sort(dists_gal, 4);
            rdists_prb = sort(dists_prb, 4);
            maxdist_gal = rdists_gal(:, :, :, floor(Nr/2));
            maxdist_prb = rdists_prb(:, :, :, floor(Nr/2));
            
            % normalize
            lwdist = min(maxdist_gal(:));
            updist = max(maxdist_gal(:));
            salience_gal = (maxdist_gal-lwdist)./(updist-lwdist);
            salience_prb = (maxdist_prb-lwdist)./(updist-lwdist);
            save([salience_dir, 'knn_salience_trial', num2str(TRIAL), '.mat'], 'salience_gal', 'salience_prb');
            
        else
            load([salience_dir, 'knn_salience_trial', num2str(TRIAL), '.mat']);
        end
        
        % convert to cell for computation convenience
        salience_gal_cell = squeeze(mat2cell(salience_gal, ny, nx, ones(1, gsize)));
        salience_prb_cell = squeeze(mat2cell(salience_prb, ny, nx, ones(1, gsize)));
        sg_cell = repmat(salience_gal_cell', gsize, 1);
        sp_cell = repmat(salience_prb_cell, 1, gsize);
        
    case 2 % One-Class SVM salience, which may consume more computation time
        
        if ~exist([salience_dir, 'ocsvm_salience_trial', num2str(TRIAL), '.mat'], 'file')
            
            % load reference data
            ref_feat_gal = zeros(dim, ny*nx, Nr);
            ref_feat_prb = zeros(dim, ny*nx, Nr);
            hwait = waitbar(0, 'Loading reference data ...');
            for i = 1:Nr
                load([feat_dir, 'feat', num2str(trnSg(i)), '.mat']);
                ref_feat_gal(:, :, i) = densefeat;
                load([feat_dir, 'feat', num2str(trnSp(i)), '.mat']);
                ref_feat_prb(:, :, i) = densefeat;
                waitbar(i/Nr, hwait);
            end
            
            cellmap_gal = struct2cell(ref_pwmap_gal);
            cellmap_prb = struct2cell(ref_pwmap_prb);
            dists_gal = cell2mat(reshape(cellmap_gal(1, :, :), 1, 1, gsize, Nr));
            dists_prb = cell2mat(reshape(cellmap_prb(1, :, :), 1, 1, gsize, Nr));
            fpos_gal = squeeze(cellmap_gal(2, :, :));
            fpos_prb = squeeze(cellmap_prb(2, :, :));
            maxdist_gal = zeros(ny*nx, gsize);
            maxdist_prb = zeros(ny*nx, gsize);
            
            % compute ocsvm salience for gallery images
            for i = 1:gsize
                xnn = {};
                for j = 1:Nr
                    feat_cell = reshape(mat2cell(ref_feat_prb(:, :, j), dim, ones(1, ny*nx)), ny, nx);
                    P = fpos_gal{i, j};
                    xnn = cat(3, xnn, feat_cell(sub2ind([ny, nx], repmat((1:ny)', 1, nx), double(P))));
                end
                xnn = reshape(xnn, ny*nx, Nr);
                dists = reshape(dists_gal(:, :, i, :), ny*nx, Nr);
                maxidx = zeros(1, ny*nx);
                parfor j = 1:ny*nx
                    X = cell2mat(xnn(j, :))';
                    [~, maxidx(j)] = ocsvm_max(X);
                end
                maxdist_gal(:, i) = dists(sub2ind([ny*nx, Nr], 1:ny*nx, maxidx));
                waitbar(i/gsize, hwait, ['Computing OCSVM salience for gallery image ', num2str(i), '/', num2str(gsize)]);
            end
            maxdist_gal = reshape(maxdist_gal, ny, nx, gsize);
            
            % compute ocsvm salience for probe images
            for i = 1:gsize
                xnn = {};
                for j = 1:Nr
                    feat_cell = reshape(mat2cell(ref_feat_gal(:, :, j), dim, ones(1, ny*nx)), ny, nx);
                    P = fpos_prb{i, j};
                    xnn = cat(3, xnn, feat_cell(sub2ind([ny, nx], repmat((1:ny)', 1, nx), double(P))));
                end
                xnn = reshape(xnn, ny*nx, Nr);
                dists = reshape(dists_prb(:, :, i, :), ny*nx, Nr);
                maxidx = zeros(1, ny*nx);
                parfor j = 1:ny*nx
                    X = cell2mat(xnn(j, :))';
                    [~, maxidx(j)] = ocsvm_max(X);
                end
                maxdist_prb(:, i) = dists(sub2ind([ny*nx, Nr], 1:ny*nx, maxidx));
                waitbar(i/gsize, hwait, ['Computing OCSVM salience for probe image ', num2str(i), '/', num2str(gsize)]);
            end
            maxdist_prb = reshape(maxdist_prb, ny, nx, gsize);
            
            % normalization
            lwdist = min(maxdist_gal(:));
            updist = max(maxdist_gal(:));
            salience_gal = (maxdist_gal-lwdist)./(updist-lwdist);
            salience_prb = (maxdist_prb-lwdist)./(updist-lwdist);
            
            save([salience_dir, 'ocsvm_salience_trial', num2str(TRIAL), '.mat'], 'salience_gal', 'salience_prb');
            close(hwait);
            
        else
            load([salience_dir, 'ocsvm_salience_trial', num2str(TRIAL), '.mat']);
        end
        
        % convert to cells for computation convenience
        salience_gal_cell = squeeze(mat2cell(salience_gal, ny, nx, ones(1, gsize)));
        salience_prb_cell = squeeze(mat2cell(salience_prb, ny, nx, ones(1, gsize)));
        sg_cell = repmat(salience_gal_cell', gsize, 1);
        sp_cell = repmat(salience_prb_cell, 1, gsize);
    
    otherwise
        error('unknown option');
end

   
%% compute pairwise matching scores 
clear phi;
phi = cell(gsize, gsize);

parfor i = 1:numel(D_cell)
        phi{i} = phiFun(D_cell{i}, P_cell{i}, mp_cell{i}, sp_cell{i}, sg_cell{i}, mg_cell{i}, par);
end

pwdist_sal = cellfun(@(x) dot(ones(1, numel(x)), x(:)), phi);
pwdist_sal = pwdist_sal';


%% combine with features proposed in SDALF
load([pwdist_dir, 'MSCRmatch_VIPeR_f1_Exp007.mat']);
load([pwdist_dir, 'txpatchmatch_VIPeR_f1_Exp007.mat']);
load([pwdist_dir, 'wHSVmatch_VIPeR_f1_Exp007.mat']);
pwdist_y = final_dist_y(Sg, Sp);
pwdist_color = final_dist_color(Sg, Sp);
pwdist_y = pwdist_y./repmat(max(pwdist_y, [], 1), gsize, 1);
pwdist_color = pwdist_color./repmat(max(pwdist_color, [], 1), gsize, 1);
pwdist_hist = final_dist_hist(Sg, Sp);
pwdist_epitext = dist_epitext(Sg, Sp);

pwdist = par.alpha(1).*pwdist_sal + ...
    par.alpha(2).*pwdist_y + par.alpha(3).*pwdist_color + ...
    par.alpha(4).*pwdist_hist + par.alpha(5).*pwdist_epitext;


%% evaluate re-identification performance
CMC = evaluate_pwdist(pwdist); 
fprintf('CMC-rank1:%2.2f%%\n', CMC(1)*100);

close all force;
