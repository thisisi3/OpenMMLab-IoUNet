from mmdet.core import bbox_overlaps, bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from .utils import clip_bboxes_to_image
import math, torch
        
class RoIGenerator(torch.nn.Module):
    """It generates IoU balanced rois from a given set of GT bboxes.
    We use delta since it is scale-invariant. A large amount of deltas are 
    generated in the begining and later a small amount is sampled and applied 
    to any given GT bbox to generate IoU balanced rois. A delta is a (cy, cy, w, h) 
    tuple that represent a scale normalized transform of a bbox.

    Args:
        pre_sample (int): number of deltas to sample in each IoU category during initialization.
        xy_steps (int): discretization of x, y coordinates
        wh_steps (int): discretization of w, h coordinates
        xy_range (tuple(int, int)): range of deviation along x, y axis
        area_range (tuple(int, int)): range of deviation along w, h axis
        nonlinearity (int): discretize distance in linearly spaced or nonlinearly spaced fashion
        per_iou (int): number of rois sampled per iou per GT
        sample_num (int): number of rois to sample per image, per_iou and sample_num can not 
            both provided or both not provided
        start_bin (int): start IoU = start_bin * 0.1
        end_bin (int): end IoU = end_bin * 0.1
        max_num (int): max number of rois per image
        compensate (dict | None): oversample some category 
    """
    def __init__(self,
                 pre_sample=1000,
                 xy_steps=21,
                 wh_steps=16,
                 xy_range=(0, 1),
                 area_range=(1/3, 3),
                 nonlinearity=1,
                 per_iou=None,
                 sample_num=None,
                 start_bin=5,
                 end_bin=10,
                 max_num=1000,
                 compensate=None):
        super(RoIGenerator, self).__init__()
        assert 0 <= start_bin < end_bin <= 10, 'invalid start_bin and end_bin values'
        assert (per_iou is None) ^ (sample_num is None), \
            'choose either per_iou or sample_num'
        assert compensate is None or isinstance(compensate, dict)
        if isinstance(compensate, dict):
            assert min([v for k, v in compensate.items()]) >= 1.0
        self.pre_sample = pre_sample
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.xy_steps = xy_steps
        self.wh_steps = wh_steps
        self.xy_range = xy_range
        self.area_range = area_range
        self.nonlinearity = nonlinearity
        self.per_iou = per_iou
        self.sample_num = sample_num
        self.max_num = max_num
        self.compensate = compensate
        delta = self.generate_mesh().view(-1, 4) # [n, 4]
        bboxes = torch.tensor([[0, 0, 1000, 1000]], dtype=torch.float) # [1, 4]
        applied = self.apply_delta(bboxes, delta)[0] # [n, 4]
        iou = bbox_overlaps(bboxes, applied).view(-1) # [n]
        delta_list = []
        delta_starts = [0]
        delta_nums = []
        for i, lo in enumerate([0.1 * i for i in range(10)]):
            hi = lo + 0.1
            lo, hi = round(lo, 1), round(hi, 1)
            mask = (iou >= lo) & (iou < hi)
            pre_num_chosen = mask.sum().item()
            chosen_delta = delta[mask]
            if pre_num_chosen > self.pre_sample:
                chosen_delta = chosen_delta[
                    self._rand_choose_index(pre_num_chosen, self.pre_sample)]
            delta_list.append(chosen_delta)
            num_chosen = chosen_delta.size(0)
            delta_starts.append(delta_starts[-1] + num_chosen)
            delta_nums.append(num_chosen)
        self.register_buffer('delta', torch.cat(delta_list))
        self.delta_starts = delta_starts
        self.delta_nums = delta_nums

    def generate_mesh(self):
        """Generate cross product of discrete x, y, w and h."""
        xyd = torch.linspace(self.xy_range[0], self.xy_range[1], self.xy_steps)
        whd = torch.linspace(math.sqrt(self.area_range[0]),
                             math.sqrt(self.area_range[1]),
                             self.wh_steps)
        xyd = self._nonlinear_map(xyd, self.nonlinearity)
        xyd = self._mirror(xyd)
        whd = self._nonlinear_map(whd, self.nonlinearity)
        meshes = torch.stack(torch.meshgrid([xyd, xyd, whd, whd]), dim=-1)
        return meshes

    def apply_delta(self, bboxes, delta):
        """Apply each delta to each bboxes.

        Args:
            bboxes (Tensor[n, 4]): e.g. GT bboxes
            delta  (Tensor[m, 4]): a set of deltas

        Returns:
            Tensor[n, m, 4]: n x m transoformed bboxes
        """
        bboxes = bboxes.view(-1, 1, 4)
        cxcywh = bbox_xyxy_to_cxcywh(bboxes)
        cx, cy, w, h = [cxcywh[..., i] for i in range(4)]
        cx = cx + w * delta[..., 0]
        cy = cy + h * delta[..., 1]
        w = w * delta[..., 2]
        h = h * delta[..., 3]
        cxcywh = torch.stack([cx, cy, w, h], dim=-1)
        return bbox_cxcywh_to_xyxy(cxcywh)

    def generate_roi(self, gt_bboxes, max_shape=None):
        """Generate rois around gt_bboxes such that they have certain IoU overlaps with gt_bboxes"""
        sampled_delta = []

        if self.sample_num is not None:
            per_iou = self.sample_num / (self.end_bin - self.start_bin)
            per_iou = int(per_iou) + 1
        else:
            per_iou = self.per_iou
        for b in range(self.start_bin, self.end_bin):
            num_delta = self.delta_nums[b]
            start_mark = self.delta_starts[b]
            if self.compensate is not None and b in self.compensate:
                cur_per_iou = int(per_iou * self.compensate[b]) + 1
            else:
                cur_per_iou = per_iou
            sampled_idx = start_mark + self._rand_choose_index(num_delta, cur_per_iou)
            sampled_delta.append(self.delta[sampled_idx])
        sampled_delta = torch.cat(sampled_delta)
        rois = self.apply_delta(gt_bboxes, sampled_delta).view(-1, 4)
        if self.sample_num is not None:
            rois = rois[self._rand_choose_index(rois.size(0), self.sample_num)]
        else:
            rois = rois[self._rand_choose_index(rois.size(0), self.max_num)]
        if max_shape is not None:
            rois = clip_bboxes_to_image(rois, max_shape[:2])
        return rois

    # mirror values to both sides of 0
    def _mirror(self, vals):
        all_vals = set(vals.tolist())
        all_vals.update(set((-vals).tolist()))
        all_vals = sorted(list(all_vals))
        return torch.tensor(all_vals, device=vals.device)

    # map values so that smaller values get more dense
    def _nonlinear_map(self, space, order):
        ma, mi = space.max(), space.min()
        normed = (space - mi) / (ma - mi)
        normed = normed.pow(order)
        return normed * (ma-mi) + mi

    def _rand_choose_index(self, n, c):
        if n <= c:
            return torch.randperm(n)
        return torch.randperm(n)[:c]

