from ..models import HEADS, build_loss, SmoothL1Loss
from ..core import bbox_overlaps
import torch
from torch import nn
from mmcv.runner import BaseModule, force_fp32

@HEADS.register_module()
class IoUHead(BaseModule):
    """IoU prediction head from paper:
    `Acquisition of Localization Confidence for Accurate Object Detection`
    
    Args:
        in_channels (int): channels of roi pooling result
        fc_channels (tuple(int, int)): channels for all fc layers
        num_classes (int): number of classes
        target_norm (dict): IoU prediction values will be normed by this
        class_agnostic (bool): if treat all classes as one
        loss_iou(dict): IoU prediction loss
        init_cfg(dict or list[dict]): initialization config dict.
    """
    def __init__(self,
                 in_channels,
                 fc_channels=[1024, 1024],
                 num_classes=80,
                 target_norm=dict(mean=0.5, std=0.5),
                 class_agnostic=False,
                 loss_iou=dict(type='SmoothL1Loss'),
                 init_cfg=None):
        super(IoUHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.num_classes = num_classes
        self.target_norm = target_norm
        self.class_agnostic = class_agnostic
        
        self.loss_type = loss_iou.type
        assert self.loss_type in ('SmoothL1Loss', 'SimpleGFLLoss'), \
            'IoU loss only support SmoothL1Loss or SimpleGFLLoss'
        self.loss_iou = build_loss(loss_iou)

        self.shared_fcs = nn.ModuleList()
        for cur_fc_channels in fc_channels:
            self.shared_fcs.append(nn.Linear(in_channels, cur_fc_channels))
            in_channels = cur_fc_channels
        self.fc_iou = nn.Linear(fc_channels[-1], 1 if class_agnostic else num_classes)
        
        self.relu = nn.ReLU(inplace=True)

        if self.init_cfg is None:
            self.init_cfg = [
                dict(type='Normal', std=0.01, override=dict(name='fc_iou')),
                dict(type='Xavier', layer='Linear', override=dict(name='shared_fcs'))]

    def forward(self, x):
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        out = self.fc_iou(x)
        if not self.training:
            if self.loss_type == 'SmoothL1Loss':
                out = out * self.target_norm.std + self.target_norm.mean
            elif self.loss_type == 'SimpleGFLLoss':
                out = out.sigmoid()
        return out

    @force_fp32(apply_to=('iou_score'))
    def loss(self,
             iou_score,
             sampling_results,
             gt_bboxes,
             gt_labels,
             rois,
             img_metas):
        target_bboxes = torch.cat([gt_bboxes[i][res.pos_assigned_gt_inds] \
                                   for i, res in enumerate(sampling_results)])
        target_labels = torch.cat([gt_labels[i][res.pos_assigned_gt_inds] \
                                   for i, res in enumerate(sampling_results)])

        target_ious = bbox_overlaps(rois[:, 1:], target_bboxes, is_aligned=True)
        if self.loss_type == 'SmoothL1Loss':
            target_ious = (target_ious - self.target_norm.mean) / self.target_norm.std

        if not self.class_agnostic:
            iou_score = iou_score[torch.arange(rois.size(0)), target_labels]

        loss_iou = self.loss_iou(iou_score, target_ious, avg_factor=rois.size(0))
        return loss_iou
