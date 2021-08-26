import torch

# from torchvision
def clip_bboxes_to_image(bboxes, size):
    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 4]: clipped boxes
    """
    dim = bboxes.dim()
    bboxes_x = bboxes[..., 0::2]
    bboxes_y = bboxes[..., 1::2]
    height, width = size

    bboxes_x = bboxes_x.clamp(min=0, max=width)
    bboxes_y = bboxes_y.clamp(min=0, max=height)

    clipped_bboxes = torch.stack((bboxes_x, bboxes_y), dim=dim)
    return clipped_bboxes.reshape(bboxes.shape)
