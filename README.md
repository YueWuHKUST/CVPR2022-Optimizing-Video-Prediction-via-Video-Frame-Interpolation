# Optimizing Video Prediction via Video Frame Interpolation

<p align="center"> 
<img src="/images/demo.gif">
</p>

This is an official pytorch implementation of the following paper:

**Optimizing Video Prediction via Video Frame Interpolation**, IEEE Computer Vision and Pattern Recognition (CVPR), 2022.
[Yue Wu](https://yuewuhkust.github.io/), Qiang Wen, [Qifeng Chen](https://cqf.io/)

### [Project page](https://yuewuhkust.github.io/OVP_VFI/) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Optimizing_Video_Prediction_via_Video_Frame_Interpolation_CVPR_2022_paper.pdf) | [Video](https://www.youtube.com/watch?v=sRudlC9r8VM) ###

Abstract:
_Video prediction is an extrapolation task that predicts future frames given past frames, and video frame interpolation is an interpolation task that estimates intermediate frames between two frames. We have witnessed the tremendous advancement of video frame interpolation, but the general video prediction in the wild is still an open question. Inspired by the photo-realistic results of video frame interpolation, we present a new optimization framework for video prediction via video frame interpolation, in which we solve an extrapolation problem based on an interpolation model. Our video prediction framework is based on optimization with a pretrained differentiable video frame interpolation module without the need for a training dataset, and thus there is no domain gap issue between training and test data. Also, our approach does not need any additional information such as semantic or instance maps, which makes our framework applicable to any video. Extensive experiments on the Cityscapes, KITTI, DAVIS, Middlebury, and Vimeo90K datasets show that our video prediction results are robust in general scenarios, and our approach outperforms other video prediction methods that require a large amount of training data or extra semantic information._

## Requirements
- Currently only Linux is supported.
- 64-bit Python 3.6 installation or newer. We recommend using Anaconda3.
- All the experiments are conducted using NVIDIA RTX 3090Ti. 

## Installation
Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/YueWuHKUST/CVPR2022-Optimizing-Video-Prediction-via-Video-Frame-Interpolation
cd CVPR2022-Optimizing-Video-Prediction-via-Video-Frame-Interpolation
conda env create -f environment.yml
source activate videopred
```

## Results
We provide our results on various datasets as indicted below

|Dataset|Resolution|Download|
|:----:|:-----------:|:-----------:|
|Cityscapes | 256x256 |[link]() |
|Kitti | 256x256 |[link]() |
|DAVIS | 256x256 |[link]() |
|Middlebury | 256x256 |[link]() |
|DAVIS | 256x256 |[link]() |

## Evaluation datasets
All the evaluation datasets are downloaded from official link and conduct only resize operations without any other special preprocessing.

```
./CVPR2022-Optimizing-Video-Prediction-via-Video-Frame-Interpolation/
|
./datasets/
    │
    └───   Cityscapes
    └───    KTTIT
    ...
```

### Running Optimization
We provide optimization scripts in folder "./scripts/"
For example, 
```
bash ./scrips/cityscapes.sh
```
All the experiments in the paper are conducted using multiprocessing for reducing time. We also provide a script, ./scripts/run.py as a reference.

Note that all the scores reported in the paper are using the setting of optimized every frame for 3K iterations. And We also provide a convergence analysis as indicted in Paper Section 4.4, that our method conveges around 1K iterations. 

## Evaluation

Our evaluation follows the setting of [FVS](https://github.com/YueWuHKUST/CVPR2020-FutureVideoSynthesis).
For the evaluation of Cityscapes and Kitti datasets, we use the same test split as FVS, and also adopt the MS-SSIM and LPIPS as evaluation metric.

## Citation

Please cite the following paper if this work helps your research:

    @inproceedings{yue2022videopred,
		title={Optimizing Video Prediction via Video Frame Interpolation},
		author={Wu, Yue and Wen Qiang and Chen Qifeng},
		booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
		year={2022}
	}

## Contact
If you have any questions, please contact Yue Wu (wu.kathrina@gmail.com) and Qifeng Chen (chenqifeng22@gmail.com)

## Acknowledgements
Our Work is built upon [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) and [RAFT](https://github.com/princeton-vl/RAFT). We thank the authors for their excellent work. 