# mvs
Multi-view satellite imagery-based 3D reconstruction method

Official Implementation of Optimized Image Combination Selection and Confidence-guided DSM Fusion for 3D Reconstruction from Multi-view Satellite Imagery


Data Preparation

US3D dataset: https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019

MVS3D dataset: https://spacenet.ai/iarpa-multi-view-stereo-3d-mapping/


Algorithm Steps

1. Read multi-view satellite image metadata and select an optimized image combination.
2. Perform pairwise stereo matching.
3. Compute confidence estimates.
4. DSM fusion.


Acknowledgements

Thanks to the authors of SatMVSF, Satellite Stereo Pipeline, Adapted COLMAP for open sourcing their fantastic projects. You may want to visit these projects at:

https://gpcv.whu.edu.cn/data

https://github.com/centreborelli/s2p

https://github.com/Kai-46/VisSatSatelliteStereo

The code will be continuously updated and refined.
