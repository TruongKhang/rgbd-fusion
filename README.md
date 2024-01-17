# Implicit Surface Reconstruction from RGB-D images

## Overview

Given a set of RGB-D images capturing a scene, our goal is to reconstruct the 3D model of this scene. Conventional methods estimate a pointcloud following these steps:
- First, for each image, these methods eliminate the pixels with low confidence values (the confidence map is estimated from a multi-view stereo method). This step is optional.
- Next, they convert the highly confident image pixels to 3D points using depth information and camera matrices. 
- Finally, these methods perform a geometric consistency checking step to filter noisy 3D points. More specifically, we assume that each image has a set of $N$ neighboring images (given in a `pair.txt` file). An 3D point is kept if its reprojection errors are smaller than an $\epsilon$ value in $N_c/N$ neighbors.

If you want to understand more about the process above, Secion 4.2 in [this paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf) is a good start. 

The given code implements these steps. You can try to run and see the results.

    python concistency_fusion.py --dataset_dir <path to your data folder> --conf_thr <confidence threshold> --nview_thr <number of consistent neighbors> --disp_thr <reprojection error>

    # For example,
    python consistency_fusion.py --dataset_dir data/Family --conf_thr 0.8 --nview_thr 5 --disp_thr 0.8

## Data

The input data is organized as follows:

    <scene_name>
        images 
            00000000.jpg    
            00000001.jpg    
            ...
        cams
            00000000_cam.txt  # camera matrices (intrinsic, extrinsic)
            00000001_cam.txt  # for each view 
            ...
        depth_est
            00000000.pfm   # depth maps, shape is (H, W)
            00000001.pfm
            ...
        confidence
            00000000.pfm   # confidence maps: shape is (H, W, 3)
            00000001.pfm
        pair.txt

You can download the data [here](https://drive.google.com/drive/folders/1DyV-7vhpJdPeOr1jteHAEdZwJB5d949s?usp=sharing). This dataset is originally from [Tanks&Temples](https://www.tanksandtemples.org/download/) benchmark.

## Your tasks

You'll try to produce a neural surface reconstruction of the provided RGB-D dataset using available methods. You can chooese one of the following:

- Neural RGB-D Surface Reconstruction ([paper](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/static/pdf/neural_rgbd_surface_reconstruction.pdf), [code](https://github.com/dazinovic/neural-rgbd-surface-reconstruction/tree/main))
- GO-Surf: Neural Feature Grid Optimization for
Fast, High-Fidelity RGB-D Surface Reconstruction ([paper](https://arxiv.org/pdf/2206.14735v2.pdf), [code](https://github.com/JingwenWang95/go-surf))

You can clone these code and import necessary functions to your code.

I want to see how neural reconstruction approach can deal with large scenes. 

You can ask me if you have any problems when running the code.



