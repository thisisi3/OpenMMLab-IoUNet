from mmdet.core import bbox_overlaps
from mmcv.ops import nms as nms_mmcv
import torch

def iou_nms(bboxes, scores, locs, iou_threshold):
    """
    Use localization score to guide NMS. Bboxes are sorted by locs, the final score is
    taking the max of scores of all suppressed bboxes.
    """
    _, keep = nms_mmcv(bboxes, locs, iou_threshold)
    keep_bboxes = bboxes[keep]
    overlaps = bbox_overlaps(keep_bboxes, bboxes)
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
    return keep[srt_idx], keep_scores


def batched_iou_nms(bboxes, scores, locs, labels, iou_threshold, score_thr=None, guide='rank'):
    """IoU guided NMS.
    
    Args:
        bboxes (Tensor): shape (n, 4)
        scores (Tensor): shape (n,), classification score
        locs (Tensor): shape(n,) localization score/iou score
        labels (Tensor): label of bboxes, help to do batched nms
        iou_threshold (float): iou threshold
        score_thr (float): filter scores belowing this number
        guide (str): decide how to use iou score to guide nms, 
            supported key words: none, rank, weight"""
    if score_thr is not None:
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
    # 'rank' is the official iou guided nms
    if guide == 'rank':
        keep, keep_scores = iou_nms(nms_bboxes, scores, locs, iou_threshold)
        return bboxes[keep], keep_scores, locs[keep], labels[keep]
    # use the product of cls_score * iou_score as the nms_locs
    elif guide == 'weight':
        nms_locs = scores * locs
        keep, keep_scores = iou_nms(nms_bboxes, scores, nms_locs, iou_threshold)
        return bboxes[keep], keep_scores, locs[keep], labels[keep]
    # do not utilize iou_score in nms
    elif guide == 'none':
        _, keep = nms_mmcv(nms_bboxes, scores, iou_threshold)
        return bboxes[keep], scores[keep], locs[keep], labels[keep]
    else:
        raise RuntimeError('guide type not supported: {}'.format(guide))

