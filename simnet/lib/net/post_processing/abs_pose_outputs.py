import numpy as np
import torch
import torch.nn as nn
from simnet.lib.net.post_processing import pose_outputs
from simnet.lib.net import losses
import torch.nn.functional as F

_mask_l1_loss = losses.MaskedL1Loss()
_mse_loss = losses.MSELoss()

class OBBOutput:
  def __init__(self, heatmap, shape_code, appearance_code, abs_pose_field, hparams):
    self.heatmap = heatmap
    self.shape_emb = shape_code
    self.appearance_emb = appearance_code
    self.abs_pose_field = abs_pose_field
    self.is_numpy = False
    self.hparams = hparams

  def convert_to_numpy_from_torch(self):
    self.heatmap = np.ascontiguousarray(self.heatmap.cpu().numpy())

    #latent emb shape
    self.shape_emb = np.ascontiguousarray(self.shape_emb.cpu().numpy())
    self.shape_emb = self.shape_emb.transpose((0, 2, 3, 1))
    self.shape_emb = self.shape_emb / 100.0
    # self.shape_emb = self.shape_emb / 10.0

    #latent emb appearance
    self.appearance_emb = np.ascontiguousarray(self.appearance_emb.cpu().numpy())
    self.appearance_emb = self.appearance_emb.transpose((0, 2, 3, 1))
    self.appearance_emb = self.appearance_emb / 100.0

    #abs pose emb
    self.abs_pose_field = np.ascontiguousarray(self.abs_pose_field.cpu().numpy())
    self.abs_pose_field = self.abs_pose_field.transpose((0, 2, 3, 1))
    self.abs_pose_field = self.abs_pose_field / 100.0
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    #latent embedding shape
    self.shape_emb = self.shape_emb.transpose((2, 0, 1))
    self.shape_emb = 100.0 * self.shape_emb
    self.shape_emb = torch.from_numpy(np.ascontiguousarray(self.shape_emb)).float()

    #latent embedding appearance
    self.appearance_emb = self.appearance_emb.transpose((2, 0, 1))
    self.appearance_emb = 100.0 * self.appearance_emb
    self.appearance_emb = torch.from_numpy(np.ascontiguousarray(self.appearance_emb)).float()

    #abs pose
    self.abs_pose_field = self.abs_pose_field.transpose((2, 0, 1))
    self.abs_pose_field = 100.0 * self.abs_pose_field
    self.abs_pose_field = torch.from_numpy(np.ascontiguousarray(self.abs_pose_field)).float()
    self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()

  def nms_heatmap(self, heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

  def compute_shape_pose_and_appearance(self, min_confidence, is_target):
    if is_target:
      heatmap = np.ascontiguousarray(self.heatmap.cpu().numpy())
      
      shape_emb = np.ascontiguousarray(self.shape_emb.cpu().numpy())
      shape_emb = shape_emb.transpose((1, 2, 0))
      shape_emb = shape_emb / 100.0

      appearance_emb = np.ascontiguousarray(self.appearance_emb.cpu().numpy())
      appearance_emb = appearance_emb.transpose((1, 2, 0))
      appearance_emb = appearance_emb / 100.0

      abs_pose_field = np.ascontiguousarray(self.abs_pose_field.cpu().numpy())
      abs_pose_field = abs_pose_field.transpose((1, 2, 0))
      abs_pose_field = abs_pose_field / 100.0
      shape_embs, appearance_embs, abs_pose_outputs, img, scores, indices = compute_shape_pose_and_appearance(heatmap, shape_emb, appearance_emb, abs_pose_field, min_confidence)
    else:
      if not self.is_numpy:
        self.convert_to_numpy_from_torch()
      shape_embs, appearance_embs, abs_pose_outputs, img, scores, indices = compute_shape_pose_and_appearance(np.copy(self.heatmap[0]), np.copy(self.shape_emb[0]), np.copy(self.appearance_emb[0]),  np.copy(self.abs_pose_field[0]), min_confidence)
    return shape_embs, appearance_embs, abs_pose_outputs, img, scores, indices

  def compute_loss(self, obb_targets, log, prefix):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")

    heatmap_target = torch.stack([obb_target.heatmap for obb_target in obb_targets])

    # Move to GPU
    heatmap_target = heatmap_target.to(torch.device('cuda:0'))

    if obb_targets[0].shape_emb is not None:
      shape_emb_target = torch.stack([obb_target.shape_emb for obb_target in obb_targets])
      shape_emb_target = shape_emb_target.to(torch.device('cuda:0'))
      
      shape_emb_loss = _mask_l1_loss(shape_emb_target, self.shape_emb, heatmap_target)
      log[f'{prefix}/shape_emb_loss'] = shape_emb_loss.item()

    if obb_targets[0].appearance_emb is not None:
      appearance_emb_target = torch.stack([obb_target.appearance_emb for obb_target in obb_targets])
      appearance_emb_target = appearance_emb_target.to(torch.device('cuda:0'))
      
      appearance_emb_loss = _mask_l1_loss(appearance_emb_target, self.appearance_emb, heatmap_target)
      log[f'{prefix}/appearance_emb_loss'] = appearance_emb_loss.item()

    heatmap_loss = _mse_loss(heatmap_target, self.heatmap)
    log[f'{prefix}/heatmap'] = heatmap_loss.item()

    #Naively regressing the absolute pose
    if obb_targets[0].abs_pose_field is not None:
      abs_pose_target = torch.stack([obb_target.abs_pose_field for obb_target in obb_targets])
      abs_pose_target = abs_pose_target.to(torch.device('cuda:0'))

      # svd rotation loss
      abs_rotation_loss = _mask_l1_loss(abs_pose_target[:,:9,:,:], self.abs_pose_field[:,:9,:,:], heatmap_target)
      log[f'{prefix}/abs_svd_rotation'] = abs_rotation_loss.item()

      # svd translation_ scale loss
      abs_trans_scale_loss = _mask_l1_loss(abs_pose_target[:,9:,:,:], self.abs_pose_field[:,9:,:,:], heatmap_target)
      log[f'{prefix}/abs_svd_translation+scale'] = abs_trans_scale_loss.item()

      abs_pose_loss = abs_rotation_loss + abs_trans_scale_loss
      log[f'{prefix}/abs_svd_pose'] = abs_pose_loss.item()
      return self.hparams.loss_heatmap_mult * heatmap_loss + self.hparams.loss_shape_emb_mult * shape_emb_loss + self.hparams.loss_appearance_emb_mult * appearance_emb_loss + self.hparams.loss_abs_pose_mult * abs_pose_loss

def compute_shape_appearance_embeddings(heatmap_output, shape_emb_output, appearance_emb_output, min_confidence):
  peaks = pose_outputs.extract_peaks_from_centroid_sorted(np.copy(heatmap_output), min_confidence)
  #peaks_image = None
  peaks_image = pose_outputs.draw_peaks(np.copy(heatmap_output), np.copy(peaks))
  shape_embs, appearance_embs, indices, scores = pose_outputs.extract_latent_emb_from_peaks(
      np.copy(heatmap_output), np.copy(peaks), np.copy(shape_emb_output), np.copy(appearance_emb_output)
  )
  return shape_embs, appearance_embs, peaks, peaks_image,scores, indices

def compute_shape_pose_and_appearance(
    heatmap_output,
    shape_emb_output,
    appearance_emb_output,
    abs_pose_output,
    min_confidence
):
  shape_embs, appearance_embs, peaks, img,scores, indices = compute_shape_appearance_embeddings(np.copy(heatmap_output), np.copy(shape_emb_output), np.copy(appearance_emb_output), min_confidence)
  
  abs_pose_outputs = pose_outputs.extract_abs_pose_from_peaks(np.copy(peaks), abs_pose_output)
  return shape_embs, appearance_embs, abs_pose_outputs, img, scores, indices