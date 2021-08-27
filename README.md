## Introduction

This project is trying to reproduce IoUNet, which is proposed in paper [Acquisition of Localization Confidence for Accurate Object Detection](https://arxiv.org/pdf/1807.11590.pdf). 

This project is also for the contest [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021).

This is NOT the official implementation.

Quick peek at the result:

> **The paper:** improved the R50 baseline by 1.7 AP.
>
> **This implementation:** improved the R50 baseline by 1.7 AP.
>
> **Extra:** 
>
> - by replacing the original SmoothL1Loss with BCE loss, we gain another 0.3 AP.
> - by changing update step lambda from 0.5 to 2.0, we gain another 0.1 AP.
> - by multiplying classification score with IoU score in IoU-guided NMS, we gain another 0.3 AP.
> - so in total we improved our baseline by 2.4 AP, reaching AP 40.0.



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

  

## Update: 2021/0821 - 2021/0826



### Code

- remove SimpleGFLLoss, use mmdet's CrossEntropyLoss instead because we are only using BCE loss.
- all code are still at `mmdet/iounet` , but I refactored the code so that adding IoUNet to mmdet is as easy as copy-and-paste.
- batched_iou_nms now uses `mmcv.ops.nms` as backend.
- use `torch.autograd.grad` to backprop gradient during inference instead of calling `backward()`, since by default parameters still requires grad during inference.



### More Experiments

Here are some of the other trainings I have tried:

- The paper has not mentioned the loss weight they use to train IoU head. I use 5.0 in my main model and also tried 10.0 and AP has dropped by 0.3.
- The paper has not mentioned how they generate RoIs to train IoU head. First I tried fixed number of RoIs per IoU per GT and allow at most 1000 RoIs per image. Later I tried fixed number of RoIs per image and every GT gets the same number of IoU-balanced RoIs. Those two strategies have similar results as shown in the following table:

| loss_weight | roi_generate | AP   | AP50 | AP75 |
| ----------- | ------------ | ---- | ---- | ---- |
| 5.0         | per_iou=30   | 39.1 | 57.7 | 42.0 |
| 5.0         | per_iou=50   | 39.2 | 57.8 | 42.3 |
| 10.0        | per_iou=50   | 38.9 | 57.2 | 42.2 |
| 5.0         | per_img=1000 | 39.3 | 57.6 | 42.2 |



### Ablation Study

The paper compares the contribution of IoU-guided NMS and Optimization-based Refinement, the two main algorithms proposed by the paper. Here is the comparison of my implementation:

| model            | paper | this_impl |
| ---------------- | ----- | --------- |
| FasterRCNN       | 36.4  | 37.6      |
| IoUNet_RCNN      | 37.0  | 37.8      |
| + IoU-guided NMS | 37.6  | 38.7      |
| + Refine         | 38.1  | 39.3      |

**IoUNet_RCNN** means turning IoUNet off during inference.



### Further Improvements

**Lambda=0.5 is too small:** lambda is the optimization step in Optimized-based Refinement. The paper uses 0.5 but I tried bigger value and gained good performance increase across all the 4 experiments mentioned above.

![large](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/better_lambda.PNG?raw=true)

From above chart we see lambda is optimal around 2.0 for all 4 experiments.

**Improve IoU-guided NMS:** in IoU-guided NMS, bboxes are sorted by predicted IoU score. I found that by multiplying IoU score with classification score, we gain further performance increase in all 4 experiments.  

| model                   | lambda | official | multiply by cls_score |
| ----------------------- | ------ | -------- | --------------------- |
| max1000_lw10.0_periou50 | 2.0    | 39.1     | 39.6                  |
| max1000_lw5.0_periou30  | 2.0    | 39.3     | 39.6                  |
| max1000_lw5.0_periou50  | 2.0    | 39.3     | 39.5                  |
| samp1000_lw5.0          | 2.0    | 39.4     | 39.5                  |

**The final improved version:**  previously I showed that using BCE loss is better. Now let's combine above two improvements with BCE loss to push the  performance further. 

| model                | AP   | Imprv |
| -------------------- | ---- | ----- |
| FasterRCNN           | 37.6 | NA    |
| IoUNet_first_version | 39.3 | +1.7  |
| + BCE loss           | 39.6 | +2.0  |
| + lambda=2.0         | 39.7 | +2.1  |
| + new iou_nms        | 40.0 | +2.4  |

The config for the improved version can be found [here](https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/assets/faster_rcnn_iou_r50_fpn_imprv_1x_coco.py).



### Inference Speed

All the following tests are done in a RTX 2080 ti GPU with batch_size=1.

| model                    | FPS  | Latency(ms) | AP    |
| ------------------------ | ---- | ----------- | ----- |
| FasterRCNN               | 20   | 50.00       | 0.376 |
| IoUNet_RCNN              | 18.9 | 52.91       | 0.378 |
| +  IoU-guided NMS        | 16.1 | 62.11       | 0.387 |
| replace with regular NMS | 16.3 | 61.35       | 0.377 |
| + Refine                 | 11.8 | 84.75       | 0.393 |

Explanation:

- In IoUNet_RCNN, IoUNet part is turned off, the only difference from FasterRCNN is RoI pooling method. I use PrRoIPool for RCNN too, which means PrRoIPool costs 2.9 ms more than traditional RoIAlign.
- By adding IoU-guided NMS, we add 9.2 ms to latency. Note that this is not solely caused by IoU-guided NMS. A PrRoIPool and a iou_head forward are also applied in order to calculate iou_score, in order to guide NMS.
- By replacing IoU-guided NMS with regular NMS, latency decreases by 0.8 ms. This means IoU-guided NMS costs 0.7 ms more than regular NMS in this context.
- By adding Optimization-based Refine we add 22.6 ms to the latency. During refinement up to 5 forward and backward data passes are performed to gradually refine RoI. However this still seems a little too high, I speculate that there is room to improve.  



## Usage

You can train and inference the model like any other models in MMDetection, see [docs](https://mmdetection.readthedocs.io/) for details.

You probably need **Ninja** in order to use ReciseRoIPooling. 



## Acknowledgement

 [Acquisition of Localization Confidence for Accurate Object Detection](https://arxiv.org/pdf/1807.11590.pdf)

[PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling)

[MMDetection](https://github.com/open-mmlab/mmdetection)

[MMCV](https://github.com/open-mmlab/mmcv)
