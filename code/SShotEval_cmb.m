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
function cmc = SShotEval_cmb(pwdist, pb, nTrial, pidx)
% Single-shot evaluation
%   cmc = SShotEval(pwdist, nTrial, pidx)
%
% INPUT
%   pwdist:     pairwise distance in cell form
%   pb:          probabilitic weight for each feature in array
%   nTrial:      number of trials
%   pidx:        person index
%
% OUTPUT
%   cmc:        the average CMC curve
%

single_query = 0; % set 0 to average over multiple queries of each person
nPerson = length(pidx);
seed = reshape(1:nPerson*nTrial, nPerson, nTrial);
Nf = length(pwdist);
gsize = nPerson;
if single_query
    psize = nPerson;
else
    psize = size(pwdist{1}, 2) - nPerson;
end

for t = 1:nTrial
    % for each person, select one for gallery, and rest for probe
    final_dist = zeros(gsize, psize);
    idprobe = [];
    for p = 1:nPerson
        index_p = pidx{p};
        nImage = length(index_p);
        s = RandStream('mcg16807','Seed', seed(p, t));
        RandStream.setDefaultStream(s);
        rp = randperm(nImage);
        gallery_index{p} = index_p(rp(1));
        if single_query
            probe_index{p} = index_p(rp(2));
            idprobe = cat(2, idprobe, p*ones(nPerson, 1));
        else
            probe_index{p} = index_p(rp(2:end));
            idprobe = cat(2, idprobe, p*ones(nPerson, nImage-1));
        end
    end
    
    p_index = cell2mat(probe_index);
    g_index = cell2mat(gallery_index);
    for i = 1:length(p_index)
        pid = p_index(i);
        for f = 1:Nf
            if f == 2 || f == 3
                pwdist{f}(g_index, pid) = pwdist{f}(g_index, pid)./max(abs(pwdist{f}(g_index, pid)));
            end
            final_dist(:, i) = final_dist(:, i) +pwdist{f}(g_index, pid)*pb(f);
        end
    end
    [~, ordered] = sort(final_dist, 'ascend');
    match = (ordered == idprobe);
    cmc(t, :) = cumsum(sum(match, 2)./size(match, 2));
end

cmc = mean(cmc, 1);