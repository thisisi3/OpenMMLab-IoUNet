import torch

def xyxy2wh(bboxes):
    return bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]

def xyxy2cxcywh(bboxes):
    w, h = xyxy2wh(bboxes)
    return torch.stack([
        (bboxes[..., 0] + bboxes[..., 2]) / 2,
        (bboxes[..., 1] + bboxes[..., 3]) / 2,
        w,
        h], dim=-1)

def cxcywh2xyxy(bboxes):
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    return torch.stack([
        cx - w/2,
        cy - h/2,
        cx + w/2,
        cy + h/2], dim=-1)

def rand_choose_index(n, c):
    if n <= c:
        return torch.randperm(n)
    return torch.randperm(n)[:c]


def restrict_bbox(bboxes, max_shape):
    max_h, max_w = max_shape[:2]
    return torch.stack([
        bboxes[..., 0].clamp(0, max_w),
        bboxes[..., 1].clamp(0, max_h),
        bboxes[..., 2].clamp(0, max_w),
        bboxes[..., 3].clamp(0, max_h)], dim=-1)

def mirror(vals):
    all_vals = set(vals.tolist())
    all_vals.update(set((-vals).tolist()))
    all_vals = sorted(list(all_vals))
    return torch.tensor(all_vals, device=vals.device)

def nonlinear_map(space, order):
    ma, mi = space.max(), space.min()
    normed = (space - mi) / (ma - mi)
    normed = normed.pow(order)
    return normed * (ma-mi) + mi
