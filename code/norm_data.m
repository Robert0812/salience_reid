%
% Normalize the original image data
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
switch lower(dataset)

    case 'viper'
        
        files_a = dir([dataset_dir, '\cam_a\*.bmp']);
        files_b = dir([dataset_dir, '\cam_b\*.bmp']);
        
        np = length(files_a);
        hwait = waitbar(0, ['Normalizing VIPeR dataset ...']);
        for p = 1:np
            im_a = imread([dataset_dir, '\cam_a\', files_a(p).name]);
            img_hsv     =   rgb2hsv(im_a);
            tmp         =   img_hsv(:,:,3);
            tmp         =   histeq(tmp); % Color Equalization
            img_hsv     =   cat(3, img_hsv(:,:,1), img_hsv(:,:,2), tmp);
            im_a = hsv2rgb(img_hsv);
            name_a = sprintf('%04d%03d.png', p, 1);
            imwrite(im_a, [dnorm_dir, name_a]);
            
            im_b = imread([dataset_dir, '\cam_b\', files_b(p).name]);
            img_hsv     =   rgb2hsv(im_b);
            tmp         =   img_hsv(:,:,3);
            tmp         =   histeq(tmp); % Color Equalization
            img_hsv     =   cat(3, img_hsv(:,:,1), img_hsv(:,:,2), tmp);
            im_b = hsv2rgb(img_hsv);
            name_b = sprintf('%04d%03d.png', p, 2);
            imwrite(im_b, [dnorm_dir, name_b]);
            waitbar(p/np, hwait);
        end
        close(hwait);
        
    case {'ethz1', 'ethz2', 'ethz3'}
        
        seq = strcat(strrep(lower(dataset), 'ethz', 'seq'), '\');
        switch lower(dataset)
            case 'ethz1'
                np = 83;
            case 'ethz2'
                np = 35;
            case 'ethz3'
                np = 28;
        end
        hwait = waitbar(0, ['Normalizing ', dataset, ' dataset ...']);
        for p = 1:np
            imdir = sprintf('%s\\%sp%03d', dataset_dir(1:end-2), seq, p);
            files = dir([imdir, '\*.png']);
            ttsize = length(files);
            
            for i = 1:ttsize
                im          =   imread([imdir, '\', files(i).name]);
                img_hsv     =   rgb2hsv(im);
                tmp         =   img_hsv(:,:,3);
                tmp         =   histeq(tmp); % Color Equalization
                img_hsv     =   cat(3, img_hsv(:,:,1), img_hsv(:,:,2), tmp);
                im_scale    =   imresize(hsv2rgb(img_hsv), [64, 32]);
                fname       =   sprintf('%04d%03d.png', p, i);
                imwrite(im_scale, [dnorm_dir, fname]);
            end
            waitbar(p/np, hwait);
        end
        close(hwait);
        
%         files = dir([dataset_dir, '*.png']);
%         hwait = waitbar(0, 'Converting ethz dataset ...');
%         for p = 1:length(files)
%             im = imread([dataset_dir, files(p).name]);
%             % normalize
% %             im_res = imresize(im, [160 60]);
%             % histogram equalization
%             im_eq = hist_eq(im);
%             imwrite(im_eq, [dnorm_dir, files(p).name]);
%             waitbar(p/length(files), hwait);
%         end
%         close(hwait); 
        
    otherwise
        
        disp('Unknown dataset!');
        
end
    


