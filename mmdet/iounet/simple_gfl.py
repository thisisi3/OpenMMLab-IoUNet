from ..models import LOSSES
import torch.nn.functional as F
import torch
def simple_gfl(pred, target, beta):
    """Simply add a pow of abs difference in front of BCE"""
    assert pred.size() == target.size(), \
        'simple GFL assume pred and target to have the same shape'
    loss = (pred.sigmoid() - target).abs().pow(beta)
    loss = F.binary_cross_entropy_with_logits(pred, target) * loss
    return loss

@LOSSES.register_module()
class SimpleGFLLoss(torch.nn.Module):
    """Simple version of Quality Focal Loss from paper `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_. """
    def __init__(self, beta=2.0, loss_weight=1.0):
        super(SimpleGFLLoss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):
        loss = simple_gfl(pred, target, self.beta)
        if weight is not None:
            loss *= weight
        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / avg_factor
        return loss * self.loss_weight
            
