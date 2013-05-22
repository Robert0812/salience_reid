Unsupervised Salience Learning for Re-id
===========================================================

MATLAB package for CVPR 13 paper "R. Zhao, W. Ouyang, and X. Wang. [Unsupervised Salience Learning for Person Re-identification](http://www.ee.cuhk.edu.hk/~rzhao/papers/zhaoOWcvpr13.pdf). In CVPR 2013."

Created by Rui Zhao, on May 20, 2013.

Summary
========
In this package, you find an updated version of MATLAB code for the following paper:
Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Unsupervised Salience Learning for Person Re-identification. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013. 


Install
========
- Download VIPeR dataset, and put the subfolders (\cam_a and \cam_b) into directory .\dataset\viper\
- Download ETHZ dataset, and put the subfolders (\seq1, \seq2, and \seq3) into directory .\dataset\ethz\


Demos
======
Two demos are available for reproducing the results.
- demo_salience_reid_viper.m : perform evaluation over VIPeR dataset
- demo_salience_reid_ethz.m  : perform evaluation over ETHZ dataset (including Seq.1, Seq.2, and Seq.3).


Remarks
========
- This implementation is a little different than the original version in the training / testing partition, so that the result may vary a little, if you use the default settings and parameters, you are supposed to obtain the rank-1 matching rate for the trial 1 on VIPeR dataset: 25.32% (SDC_knn) and 27.22% (SDC_ocsvm). 
- The training / testing partition is generated following the approach [SDALF](http://www.lorisbazzani.info/code-datasets/sdalf-descriptor/) 
- Parallel Toolbox can accellerate the computation, use matlabpool if necessary
- This demo was tested on MATLAB (R2010b), 64-bit Win7, Intel Xeon 3.30 GHz CPU
- Memory cost:
  - Running demo on VIPeR dataset would consume around 1.0 GB memory
	- Running demo on ETHZ (seq1) dataset would consume around 5.0 GB memory
	- Running demo on ETHZ (seq2) dataset would consume around 1.6 GB memory
	- Running demo on ETHZ (seq3) dataset would consume around 1.4 GB memory


Additional Libs
===============
We provide with our package some additional libraries we used in our implementation.
- svmtrain.mexw64 in LibSVM, http://www.csie.ntu.edu.tw/~cjlin/libsvm/
- slmetric_pw.m in sltoolbox, http://www.mathworks.com/matlabcentral/fileexchange/12333-statistical-learning-toolbox
- dense feature in Scenes/Objects classification toolbox, http://www.mathworks.com/matlabcentral/fileexchange/29800-scenesobjects-classification-toolbox/content/reco_toolbox/html/demo_denseSIFT.html 
Please note that the dense features codes have been heavily re-written for modification flexibility. 


BibTex
======
@inproceedings{zhao2013unsupervised,
 
 title = {Unsupervised Salience Learning for Person Re-identification},
 
 author={Zhao, Rui and Ouyang, Wanli and Wang, Xiaogang},
 
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 
 year = {2013},
 
 month = {June},
 
 address = {Portland, USA}
 
}


Acknoledgement
==============
This work is supported by the General Research Fund sponsored by the Research Grants Council of Hong Kong (Project No. CUHK 417110 and CUHK 417011) and National Natural Science Foundation of China (Project No. 61005057). The code was written by Rui Zhao, for any problem please contact the first author. 


License
=======
Copyright (c) 2013, Rui Zhao
All rights reserved. 

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright 
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the distribution
      
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
