
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class MaskedL1Loss(nn.Module):

  def __init__(self, centroid_threshold=0.3, downscale_factor=8):
    super().__init__()
    self.loss = nn.L1Loss(reduction='none')
    self.centroid_threshold = centroid_threshold
    self.downscale_factor = downscale_factor

  def forward(self, output, target, valid_mask):
    '''
        output: [N,16,H,W]
        target: [N,16,H,W]
        valid_mask: [N,H,W]
        '''
    valid_count = torch.sum(
        valid_mask[:, ::self.downscale_factor, ::self.downscale_factor] > self.centroid_threshold
    )
    loss = self.loss(output, target)
    if len(output.shape) == 4:
      loss = torch.sum(loss, dim=1)
    loss[valid_mask[:, ::self.downscale_factor, ::self.downscale_factor] < self.centroid_threshold
        ] = 0.0
    if valid_count == 0:
      return torch.sum(loss)
    return torch.sum(loss) / valid_count


class MSELoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, output, target):
    '''
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        '''
    loss = self.loss(output, target)
    return torch.mean(loss)


class MaskedMSELoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, output, target, ignore_mask):
    '''
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        '''
    valid_sum = torch.sum(torch.logical_not(ignore_mask))
    loss = self.loss(output, target)
    loss[ignore_mask > 0] = 0.0
    return torch.sum(loss) / valid_sum
  
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


class DisparityLoss(nn.Module):
  """Smooth L1-loss for disparity with check for valid ground truth"""

  def __init__(self, max_disparity, stdmean_scaled):
    super().__init__()

    self.max_disparity = max_disparity
    self.stdmean_scaled = stdmean_scaled
    self.loss = nn.SmoothL1Loss(reduction="none")

  def forward(self, disparity, disparity_gt, right=False, low_range_div=None):
    # Scale ground truth disparity based on output scale.
    scale_factor = disparity_gt.shape[2] // disparity.shape[2]
    disparity_gt = downsample_disparity(disparity_gt, scale_factor)
    max_disparity = self.max_disparity / scale_factor
    if low_range_div is not None:
      max_disparity /= low_range_div
    batch_size, _, _ = disparity.shape
    loss = torch.zeros(1, dtype=disparity.dtype, device=disparity.device)
    # Not all batch elements may have ground truth for disparity, so we compute the loss for each batch element
    # individually.
    valid_count = 0
    for batch_idx in range(batch_size):
      if torch.sum(disparity_gt[batch_idx, :, :]) < 1e-3:
        continue

      single_loss = self.loss(disparity[batch_idx, :, :], disparity_gt[batch_idx, :, :])
      valid_count += 1

      if self.stdmean_scaled:
        # Scale loss by standard deviation and mean of ground truth to reduce influence of very high disparities.
        gt_std, gt_mean = torch.std_mean(disparity_gt[batch_idx, :, :])
        loss += torch.mean(single_loss) / (gt_mean + 2.0 * gt_std)
      else:
        # Scale loss by scale factor due to difference of expected magnitude of disparity at different scales.
        loss += torch.mean(single_loss) * scale_factor
    # Avoid potential divide by 0.
    if valid_count > 0:
      return loss / batch_size
    else:
      return loss
    
def compute_depth_loss(disp_loss, depth_pred, depth_target, log, name, hparams):
    depth_loss = disp_loss(depth_pred, depth_target)
    log[name] = depth_loss.item()
    return hparams.loss_depth_mult * depth_loss

def compute_seg_loss(seg_pred, seg_target, log, name, hparams):
    seg_loss = F.cross_entropy(seg_pred, seg_target, reduction="mean", ignore_index=-100)
    # log['segmentation'] = seg_loss
    log[name] = seg_loss.item()
    return hparams.loss_seg_mult * seg_loss

_mask_l1_loss = MaskedL1Loss()
_mse_loss = MSELoss()

def compute_pose_shape_loss(self, heatmap_target, shape_emb_target, appearance_emb_target, abs_pose_target, heatmap, shape_emb, appearance_emb, abs_pose_field, log, prefix, hparams):
        
    shape_emb_loss = _mask_l1_loss(shape_emb_target, shape_emb, heatmap_target)
    log[f'{prefix}/shape_emb_loss'] = shape_emb_loss.item()

    appearance_emb_loss = _mask_l1_loss(appearance_emb_target, appearance_emb, heatmap_target)
    log[f'{prefix}/appearance_emb_loss'] = appearance_emb_loss.item()

    heatmap_loss = _mse_loss(heatmap_target, heatmap)
    log[f'{prefix}/heatmap'] = heatmap_loss.item()

    # svd rotation loss
    abs_rotation_loss = _mask_l1_loss(abs_pose_target[:,:9,:,:], abs_pose_field[:,:9,:,:], heatmap_target)
    log[f'{prefix}/abs_svd_rotation'] = abs_rotation_loss.item()

    # svd translation_ scale loss
    abs_trans_scale_loss = _mask_l1_loss(abs_pose_target[:,9:,:,:], abs_pose_field[:,9:,:,:], heatmap_target)
    log[f'{prefix}/abs_svd_translation+scale'] = abs_trans_scale_loss.item()

    abs_pose_loss = abs_rotation_loss + abs_trans_scale_loss
    log[f'{prefix}/abs_svd_pose'] = abs_pose_loss.item()
    return hparams.loss_heatmap_mult * heatmap_loss + hparams.loss_shape_emb_mult * shape_emb_loss + hparams.loss_appearance_emb_mult * appearance_emb_loss + hparams.loss_abs_pose_mult * abs_pose_loss
