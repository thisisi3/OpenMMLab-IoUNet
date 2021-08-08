from torchvision.ops import nms as nms_torch
from ..core import bbox_overlaps
import torch

def batched_iou_nms(bboxes, scores, locs, labels, iou_threshold, score_thr=None):
    """IoU guided NMS, bboxes are sorted by iou, score of the current bbox is taking the max of 
    scores of all the bboxes suppressed by the bbox.
    
    Args:
        bboxes (Tensor): shape (n, 4)
        scores (Tensor): shape (n,), classification score
        locs (Tensor): shape(n,) iou score
        labels (Tensor): label of bboxes, help to do batched nms
        iou_threshold (float): iou threshold
        score_thr (float): filter scores belowing this number"""
    filt_mask = scores >= score_thr
    if filt_mask.sum().item() == 0:
        return \
            bboxes.new_empty(0, 4), \
            scores.new_empty(0), \
            locs.new_empty(0), \
            labels.new_empty(0)
    bboxes = bboxes[filt_mask]
    scores = scores[filt_mask]
    locs   = locs[filt_mask]
    labels = labels[filt_mask]
    nms_bboxes = bboxes + (labels * (bboxes.max() + 1)).view(-1, 1)
    
    keep = nms_torch(nms_bboxes, locs, iou_threshold)
    keep_bboxes = nms_bboxes[keep]
    overlaps = bbox_overlaps(keep_bboxes, nms_bboxes)
    # find the suppressed bboxes for each kept bbox
    suppressed = overlaps > iou_threshold
    # accumulate suppression count
    suppressed = suppressed.long().cummax(0)[0].cumsum(0)
    # the real suppressed bboxes should be the ones that are suppressed exactly once
    suppressed = (suppressed == 1)
    span_scores = scores.view(1, overlaps.size(1)).repeat(overlaps.size(0), 1)
    span_scores[~suppressed] = scores.min() - 1
    # taking the max of the suppressed group
    keep_scores = span_scores.max(1)[0]
    # sort by scores, following tradition
    keep_scores, srt_idx = keep_scores.sort(descending=True)
    keep = keep[srt_idx]
    return bboxes[keep], keep_scores, locs[keep], labels[keep]

