from mmdet.models import HEADS, StandardRoIHead, build_head
from mmdet.core import bbox2roi, bbox2result, build_sampler, build_assigner
from .roi_generator import RoIGenerator
from .nms import batched_iou_nms
import torch

@HEADS.register_module()
class IoURoIHead(StandardRoIHead):
    """RoI head with a bbox_head and a iou_head"""
    def __init__(self, *args, iou_head=None, roi_generator=None, **kwargs):
        super(IoURoIHead, self).__init__(*args, **kwargs)
        assert iou_head is not None, 'IoU head must be present for StandardIoUHead'
        assert not self.with_shared_head, 'shared head is not supported for now'
        assert not self.with_mask, 'mask is not supported for now'
        self.iou_head = build_head(iou_head)
        self.roi_generator = roi_generator
        if roi_generator is not None:
            self.roi_generator = RoIGenerator(**roi_generator)

    def init_assigner_sampler(self):
        self.bbox_assigner = None
        self.bbox_sampler = None
        self.iou_assigner = None
        self.iou_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.bbox_assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.bbox_sampler, context=self)
            self.iou_assigner = build_assigner(self.train_cfg.iou_assigner)
            self.iou_sampler = build_sampler(
                self.train_cfg.iou_sampler, context=self)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        ################ the RCNN part #################
        assert gt_bboxes_ignore is None, 'do not support gt_bboxes_ignore'
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results['loss_bbox'])

        ################ the IoU part #################
        # IoUHead forward and loss
        # next do forward train on iou_head, we follow following pipeline:
        # 1, use RoIGenerator to generate rois, it also controlls number of rois in each iou
        #    so it also does the sampling
        # 2, assign rois to gt_bboxes, use default MaxIoUAssigner
        # 3, do sampling, here we use PseudoSampler, since RoIGenerator has sampling inside.
        # TODO: it may be more reasonable to let sampler do the sampling
        iou_rois_list = []
        iou_sampling_results = []
        for i in range(len(img_metas)):
            iou_rois = self.roi_generator.generate_roi(
                gt_bboxes[i], img_metas[i]['img_shape'][:2])
            iou_rois_list.append(iou_rois)
            iou_assign_result = self.iou_assigner.assign(
                iou_rois, gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            iou_sampling_result = self.iou_sampler.sample(
                iou_assign_result,
                iou_rois,
                gt_bboxes[i],
                gt_labels=gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            iou_sampling_results.append(iou_sampling_result)
        iou_losses = self._iou_forward_train(
            x, iou_sampling_results, gt_bboxes, gt_labels, img_metas)
        losses.update(iou_losses)
        return losses

    def _iou_forward(self, x, rois):
        assert rois.size(1) == 5, 'dim of rois should be [K, 5]'
        iou_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        return self.iou_head(iou_feats)
    
    def _iou_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Forward IoU head and calculate loss"""
        rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        iou_score = self._iou_forward(x, rois)
        loss_iou = self.iou_head.loss(
            iou_score, sampling_results, gt_bboxes, gt_labels, rois, img_metas)
        return dict(loss_iou=loss_iou)


    # since it calls roi_extractor, we put refinement in roi_head
    # TODO: this could be done together for all images
    def refine_by_iou(self, x, bbox, score, label, img_idx, img_meta, cfg):
        """Refine bboxes by gradient of iou_score w.r.t the bboxes in one image"""
        det_bboxes, det_scores, det_ious, det_labels = [], [], [], []
        with torch.set_grad_enabled(True):
            prev_bbox, prev_label, prev_score = bbox, label, score
            prev_bbox.requires_grad_(True)
            bbox_roi = torch.cat(
                [prev_bbox.new_full((prev_bbox.size(0), 1), img_idx), prev_bbox], dim=1)
            prev_iou = self._iou_forward(x, bbox_roi)
            prev_iou = prev_iou[torch.arange(prev_bbox.size(0)), prev_label]
            keep_mask = None
            # in the loop we do:
            #   1, backward to obtain bboxes' grad
            #   2, update bboxes according to the grad
            #   3, forward to obtain iou of new bboxes
            #   4, filter bboxes that need no more refinement
            for i in range(cfg.t):
                if prev_score.size(0) <= 0:
                    break
                #prev_iou.sum().backward()
                prev_bbox_grad = torch.autograd.grad(
                    prev_iou.sum(), prev_bbox, only_inputs=True)[0]
                if keep_mask is not None:
                    # filter bbox and grad after backward
                    bbox_grad = prev_bbox_grad[~keep_mask]
                    prev_bbox = prev_bbox[~keep_mask]
                else:
                    bbox_grad = prev_bbox_grad
                w, h = prev_bbox[..., 2]-prev_bbox[..., 0], prev_bbox[..., 3]-prev_bbox[..., 1]
                scale = torch.stack([w, h, w, h], dim=1)
                delta = cfg.lamb * bbox_grad * scale
                # apply gradient ascent
                new_bbox = prev_bbox + delta
                new_bbox = new_bbox.detach().requires_grad_(True)
                bbox_roi = torch.cat(
                    [new_bbox.new_full((new_bbox.size(0), 1), img_idx), new_bbox], dim=1)
                new_iou = self._iou_forward(x, bbox_roi)
                new_iou = new_iou[torch.arange(new_iou.size(0)), prev_label]
                keep_mask = ((prev_iou - new_iou).abs() < cfg.omega_1) | \
                            ((new_iou - prev_iou) < cfg.omega_2)
                det_bboxes.append(new_bbox[keep_mask])
                det_ious.append(new_iou[keep_mask])
                det_scores.append(prev_score[keep_mask])
                det_labels.append(prev_label[keep_mask])
                # we will filter bbox and its grad after backward in next loop
                # because new_bbox[~keep_mask].grad will be None
                prev_bbox = new_bbox
                prev_iou = new_iou[~keep_mask]
                prev_score = prev_score[~keep_mask]
                prev_label = prev_label[~keep_mask]
            # add the rest of the bboxes
            if prev_score.size(0) > 0:
                det_bboxes.append(prev_bbox[~keep_mask])
                det_scores.append(prev_score)
                det_labels.append(prev_label)
                det_ious.append(prev_iou)
        # mind that det results are not sorted by score
        det_bboxes = torch.cat(det_bboxes)
        det_scores = torch.cat(det_scores)
        det_labels = torch.cat(det_labels)
        det_ious   = torch.cat(det_ious)
        if cfg.use_iou_score:
            det_scores *= det_ious
        return det_bboxes, det_scores, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
        det_bboxes, det_labels = [], []
        iou_cfg = rcnn_test_cfg.get('iou', None)
        if iou_cfg is None:
            for i in range(len(proposals)):
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            return det_bboxes, det_labels

        # next apply iou_head, it will use some of the configs from rcnn
        for i in range(len(proposals)):
            cur_cls_score = cls_score[i].softmax(1)[:, :-1] # rm bg scores
            cur_max_score, cur_bbox_label = cur_cls_score.max(1)
            regressed = self.bbox_head.regress_by_class(
                rois[i], cur_bbox_label, bbox_pred[i], img_metas[i])
            cur_iou_score = self._iou_forward(x, regressed)

            if iou_cfg.nms.multiclass:
                nms_cls_score = cur_cls_score.reshape(-1)
                nms_iou_score = cur_iou_score.view(-1)
                nms_regressed = regressed[:, 1:].view(-1, 1, 4).repeat(
                    1, self.bbox_head.num_classes, 1).view(-1, 4)
                nms_label = torch.arange(
                    self.bbox_head.num_classes,
                    device=nms_cls_score.device).repeat(rois[i].size(0))
            else:
                nms_cls_score = cur_max_score
                nms_iou_score = cur_iou_score[
                    torch.arange(cur_iou_score.size(0)), cur_bbox_label]
                nms_regressed = regressed[:, 1:]
                nms_label = cur_bbox_label
            # apply iou_nms
            det_bbox, det_score, det_iou, det_label = batched_iou_nms(
                nms_regressed, nms_cls_score, nms_iou_score, nms_label,
                iou_cfg.nms.iou_threshold, rcnn_test_cfg.score_thr,
                guide=iou_cfg.nms.get('guide', 'rank'))
            if iou_cfg.get('refine', None) is not None and det_bbox.size(0) > 0:
                det_bbox  = det_bbox[:iou_cfg.refine.pre_refine]
                det_score = det_score[:iou_cfg.refine.pre_refine]
                det_iou   = det_iou[:iou_cfg.refine.pre_refine]
                det_label = det_label[:iou_cfg.refine.pre_refine]
                det_bbox, det_score, det_label = self.refine_by_iou(
                    x, det_bbox, det_score, det_label, i, img_metas[i],
                    iou_cfg.refine)
            if rescale and det_bbox.size(0) > 0:
                scale_factor = det_bbox.new_tensor(scale_factors[i])
                det_bbox = (det_bbox.view(det_bbox.size(0), -1, 4)/scale_factor)\
                           .view(det_bbox.size(0), -1)
            det_score, srt_idx = det_score.sort(descending=True)
            det_bbox = det_bbox[srt_idx]
            det_label = det_label[srt_idx]
            det_bbox = torch.cat([det_bbox, det_score.view(-1, 1)], dim=1)
            det_bboxes.append(det_bbox[:rcnn_test_cfg.max_per_img])
            det_labels.append(det_label[:rcnn_test_cfg.max_per_img])
        return det_bboxes, det_labels
