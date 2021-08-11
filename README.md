## Introduction

This project is trying to reproduce IoUNet, which is proposed in paper [Acquisition of Localization Confidence for Accurate Object Detection](https://arxiv.org/pdf/1807.11590.pdf). 

This project is also for the contest [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021).

This is NOT the official implementation.

Quick peek at the result:

> **The paper:** improved the R50 baseline by 1.7 AP.
>
> **This implementation:** improved the R50 baseline by 1.7 AP.
>
> **Extra:** by replacing the original SmoothL1Loss with BCE loss, we gain another 0.3 AP.



## Implementation

**PreciseRoIPooling:** it is a novel RoI pooling method proposed by the paper, they did release the CUDA implementation [here](https://github.com/vacancy/PreciseRoIPooling). So I directly use their implementation. To my best knowledge, this is the only code released. 

**RoI Generation:** in the paper, they generate RoIs to train IoUNet instead of using RPN's proposals. But they didn't say much about the generating process except for the fact that RoIs are IoU-balanced. Here is what I did, I create sufficient amount of deltas in each IoU category during initialization and sample a subset on each IoU category during training. Then I apply sampled delta to GT bboxes to generate IoU-balanced RoIs. I uniformly sample 1000 RoIs per image. 

For the rest, I closely follow the paper. 

**Code:** all the code is at `mmdet/iounet`.



## Experiments

**MMDetection:** this project is based on version v2.14.0.

**MMCV:** version v1.3.8

**Dataset:** coco_train_2017(117k) as training dataset and coco_val_2017(5k) as testing dataset. All the results are reported on coco_val_2017.

**Baseline:** The paper compared their methods on FasterRCNN, MaskRCNN and CascadeRCNN. Here I only focus on FasterRCNN due to limited computing resource. In the paper, they report their implementation of FasterRCNN to have AP=36.4, which is lower than what's reported by MMDetection(37.4). For fair comparison, I retrained FasterRCNN([faster_rcnn_r50_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/v2.14.0/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py)) on my machine and use it as the baseline(37.6). And I primarily compare the relative improvement instead of the absolute AP.



Results reported in the paper:

|                   | AP   | AP50 | AP60 | AP70 | AP80 | AP90 |
| ----------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| FasterRCNN        | 36.4 | 58.0 | 53.1 | 44.9 | 31.2 | 9.8  |
| FasterRCNN+IoUNet | 38.1 | 56.3 | 52.4 | 46.3 | 35.1 | 15.5 |
| Improvement       | +1.7 | -1.7 | -0.7 | +1.4 | +3.9 | +5.7 |

Results by this implementation:

|                      | AP   | AP50 | AP60 | AP70 | AP80 | AP90 |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| FasterRCNN_retrained | 37.6 | 58.4 | 54.1 | 46.6 | 33.2 | 11.2 |
| FasterRCNN+IoUNet    | 39.3 | 57.6 | 53.2 | 47.0 | 37.0 | 17.5 |
| Improvement          | +1.7 | -0.8 | -0.9 | +0.4 | +3.8 | +6.3 |



Log and model:

|                      | backbone | Lr schd | bbox AP | Config                                                       | Log                                                          | Model                                                        | GPUs |
| -------------------- | -------- | ------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| FasterRCNN_retrained | R-50-FPN | 1x      | 37.6    | [config](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/faster_rcnn_r50_fpn_1x_coco.py) | [log](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/faster_rcnn_r50_fpn_1x_coco_20210803_233510.log.json) | [baidu ](https://pan.baidu.com/s/1_IAGw_65fmcPFz8RQDzREw) [wuef] | 2    |
| FasterRCNN+IoUNet    | R-50-FPN | 1x      | 39.3    | [config](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/faster_rcnn_iou_r50_fpn_1x_coco.py) | [log](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/faster_rcnn_iou_r50_fpn_1x_coco_20210805_085322.log.json) | [baidu](https://pan.baidu.com/s/1hvWcMA4V9TdcFqaw8NFMRw)  [evrp] | 2    |

**AP in the log:** it is 39.2 in the log, later I get 39.3 by fixing a bug in inference code.



## Extra

This is not part of the reproduce. 

The paper uses SmoothL1Loss to estimate IoU prediction error, we replace it with BCE loss and gain another 0.3 AP:

|                       | AP   | AP50 | AP60 | AP70 | AP80 | AP90 |
| --------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| FasterRCNN_retrained  | 37.6 | 58.4 | 54.1 | 46.6 | 33.2 | 11.2 |
| FasterRCNN+IoUNet_BCE | 39.6 | 58.0 | 53.5 | 46.9 | 37.2 | 18.3 |
| Improvement           | +2.0 | -0.4 | -0.6 | +0.3 | +4.0 | +7.1 |

Log and model:

|                       | backbone | Lr schd | bbox AP | Config                                                       | Log                                                          | Model                                                        | GPUs |
| --------------------- | -------- | ------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| FasterRCNN+IoUNet_BCE | R-50-FPN | 1x      | 39.6    | [config](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/faster_rcnn_iou_r50_fpn_bce_1x_coco.py) | [log](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/faster_rcnn_iou_r50_fpn_bce_1x_coco_20210808_232046.log.json) | [baidu](https://pan.baidu.com/s/1PMHNagwZFRwZTKkk88k8IA) [8kt9] | 2    |



## Usage

You can train and inference the model like any other models in MMDetection, see [docs](https://mmdetection.readthedocs.io/) for details.

You probably need **Ninja** in order to use ReciseRoIPooling. 



## Acknowledgement

 [Acquisition of Localization Confidence for Accurate Object Detection](https://arxiv.org/pdf/1807.11590.pdf)

[PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling)

[MMDetection](https://github.com/open-mmlab/mmdetection)

[MMCV](https://github.com/open-mmlab/mmcv)
