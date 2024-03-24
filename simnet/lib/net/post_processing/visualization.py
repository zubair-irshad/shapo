import numpy as np
import cv2
from simnet.lib import color_stuff
from simnet.lib import datapoint
import torch
from torch.nn import functional as F

_MAX_DISP = 128
def get_seg_visualization_img(left_image, seg_pred):
    seg_pred = np.ascontiguousarray(seg_pred.cpu().numpy())
    return draw_segmentation_mask(left_image, seg_pred[0])

def get_depth_visualization_img(left_img_np, depth_pred, corner_scale=1, raw_disp=True):

    depth_pred = np.ascontiguousarray(depth_pred.cpu().numpy())
    disp = depth_pred[0]

    if raw_disp:
        return disp_map_visualize(disp)
    disp_scaled = disp[::corner_scale, ::corner_scale]
    left_img_np[:disp_scaled.shape[0], -disp_scaled.shape[1]:] = disp_map_visualize(disp_scaled)
    return left_img_np

def draw_segmentation_mask_gt(color_img, seg_mask, num_classes=5):
    assert len(seg_mask.shape) == 2
    seg_mask = seg_mask.astype(np.uint8)
    colors = color_stuff.get_panoptic_colors()
    color_img = color_img_to_gray(color_img)
    for ii, color in zip(range(num_classes), colors):
        colored_mask = np.zeros([seg_mask.shape[0], seg_mask.shape[1], 3])
        colored_mask[seg_mask == ii, :] = color
        color_img = cv2.addWeighted(
            color_img.astype(np.uint8), 0.9, colored_mask.astype(np.uint8), 0.4, 0
        )
    return cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

def color_img_to_gray(image):
    gray_scale_img = np.zeros(image.shape)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        gray_scale_img[:, :, i] = img
    gray_scale_img[:, :, i] = img
    return gray_scale_img

def draw_segmentation_mask(color_img, seg_mask):
    assert len(seg_mask.shape) == 3
    num_classes = seg_mask.shape[0]
    # Convert to predictions
    seg_mask_predictions = np.argmax(seg_mask, axis=0)
    return draw_segmentation_mask_gt(color_img, seg_mask_predictions, num_classes=num_classes)

def downsample_disparity(disparity, factor):
  """Downsample disparity using a min-pool operation

    Input can be either a Numpy array or Torch tensor.
    """
  with torch.no_grad():
    # Convert input to tensor at the appropriate number of dimensions if needed.
    is_numpy = type(disparity) == np.ndarray
    if is_numpy:
      disparity = torch.from_numpy(disparity)
    new_dims = 4 - len(disparity.shape)
    for i in range(new_dims):
      disparity = disparity.unsqueeze(0)

    disparity = F.max_pool2d(disparity, kernel_size=factor, stride=factor) / factor

    # Convert output disparity back into same format and number of dimensions as input.
    for i in range(new_dims):
      disparity = disparity.squeeze(0)
    if is_numpy:
      disparity = disparity.numpy()
    return disparity

def get_disparity_valid_mask(disparity, max_disparity, right=False):
    """Generate mask where disparity is valid based on the given max_disparity"""
    IGNORE_EDGE = False
    result = torch.logical_and(disparity > 1e-3, disparity < (max_disparity - 1 - 1e-3))

    if IGNORE_EDGE:
        width = disparity.shape[-1]
        edge_mask = torch.arange(width, dtype=disparity.dtype, device=disparity.device) - 1
        if right:
            edge_mask = torch.flip(edge_mask, (0,))
        edge_mask = edge_mask.expand_as(disparity)
        valid_edge = disparity < edge_mask
        result = torch.logical_and(result, valid_edge)
    return result

def turbo_vis(heatmap, normalize=False, uint8_output=False):
    assert len(heatmap.shape) == 2
    if normalize:
        heatmap = heatmap.astype(np.float32)
        heatmap -= np.min(heatmap)
        heatmap /= np.max(heatmap)
    assert heatmap.dtype != np.uint8

    x = heatmap
    x = x.clip(0, 1)
    a = (x * 255).astype(int)
    b = (a + 1).clip(max=255)
    f = x * 255.0 - a
    turbo_map = datapoint.TURBO_COLORMAP_DATA_NP[::-1]
    pseudo_color = (turbo_map[a] + (turbo_map[b] - turbo_map[a]) * f[..., np.newaxis])
    pseudo_color[heatmap < 0.0] = 0.0
    pseudo_color[heatmap > 1.0] = 1.0
    if uint8_output:
        pseudo_color = (pseudo_color * 255).astype(np.uint8)
    return pseudo_color


def disp_map_visualize(x, max_disp=_MAX_DISP):
    min_disp = -10000
    assert len(x.shape) == 2
    x = x.astype(np.float64)
    valid = ((x < max_disp) & np.isfinite(x) & (x> min_disp))
    if valid.sum() == 0 or x.sum() ==0 or x[valid].sum() ==0:
        return np.zeros_like(x).astype(np.uint8)
    if np.min(x[valid]) == np.max(x[valid]):
        return np.zeros_like(x).astype(np.uint8)
    x -= np.min(x[valid])
    x /= np.max(x[valid])
    x = 1. - x
    x[~valid] = 0.
    x = turbo_vis(x)
    x = (x * 255).astype(np.uint8)
    return x[:, :, ::-1]

