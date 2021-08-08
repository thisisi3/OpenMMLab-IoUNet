from ..models import ROI_EXTRACTORS, SingleRoIExtractor
from mmcv import ops
from .prroi_pool import PrRoIPool2D
from torch import nn


@ROI_EXTRACTORS.register_module()
class PrRoIExtractor(SingleRoIExtractor):
    """Precise RoI Pooling method from paper 
    `Acquisition of Localization Confidence for Accurate Object Detection 
    <https://arxiv.org/pdf/1807.11590.pdf>`_.

    The CUDA implementation is from the official release: 
    https://github.com/vacancy/PreciseRoIPooling"""
    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        if layer_type == 'PrRoIPool2D':
            layer_cls = PrRoIPool2D
        elif hasattr(ops, layer_type):
            layer_cls = getattr(ops, layer_type)
        else:
            raise RuntimeError('unknown roi layer: {}'.format(layer_type))
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1.0/s, **cfg) for s in featmap_strides])
        for roi_layer in roi_layers:
            if not hasattr(roi_layer, 'output_size'):
                setattr(roi_layer, 'output_size', (cfg.pooled_height, cfg.pooled_width))
        return roi_layers
