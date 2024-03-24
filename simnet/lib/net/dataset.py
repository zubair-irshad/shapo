# Copyright 2019 Toyota Research Institute.  All rights reserved.

import os
import random
import pathlib
import cv2
import numpy as np
import torch
import IPython
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from simnet.lib import datapoint
from simnet.lib.net.post_processing.segmentation_outputs import SegmentationOutput
from simnet.lib.net.post_processing.depth_outputs import DepthOutput
from simnet.lib.net.post_processing.abs_pose_outputs import OBBOutput

def extract_left_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
  anaglyph_np = anaglyph_np.transpose((1, 2, 0))
  left_img = anaglyph_np[..., 0:3] * 255.0
  return left_img

def extract_right_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
  anaglyph_np = anaglyph_np.transpose((1, 2, 0))
  left_img = anaglyph_np[..., 3:6] * 255.0
  return left_img

def create_anaglyph(stereo_dp):
  height, width, _ = stereo_dp.left_color.shape
  image = torch.zeros(4, height, width, dtype=torch.float32)
  cv2.normalize(stereo_dp.left_color, stereo_dp.left_color, 0, 255, cv2.NORM_MINMAX)

  rgb = stereo_dp.left_color* 1. / 255.0
  norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  rgb = norm(torch.from_numpy(rgb.astype(np.float32).transpose((2,0,1))))

  if len(stereo_dp.right_color.shape) == 2:
    depth = stereo_dp.right_color
    depth = torch.from_numpy(depth.astype(np.float32))

  image[0:3, :] = rgb
  image[3, :] = depth
  return image

class Dataset(Dataset):
  def __init__(self, dataset_uri, hparams, preprocess_image_func=None, datapoint_dataset=None):
    super().__init__()
    if datapoint_dataset is None:
      datapoint_dataset = datapoint.make_dataset(dataset_uri)
    self.datapoint_handles = datapoint_dataset.list()
    # No need to shuffle, already shufled based on random uids
    self.hparams = hparams
    if preprocess_image_func is None:
      self.preprocces_image_func = create_anaglyph
    else:
      self.preprocces_image_func = preprocess_image_func

  def __len__(self):
    return len(self.datapoint_handles)

  # def __getitem__(self, idx):
  #   dp = self.datapoint_handles[idx].read()
  #   anaglyph = self.preprocces_image_func(dp.stereo)
  #   segmentation_target = SegmentationOutput(dp.segmentation, self.hparams)
  #   segmentation_target.convert_to_torch_from_numpy()
  #   depth_target = DepthOutput(dp.depth, self.hparams)
  #   depth_target.convert_to_torch_from_numpy()
  #   pose_target = None
  #   # for pose_dp in dp.object_poses:
  #   #   pose_target = OBBOutput(
  #   #       pose_dp.heat_map, pose_dp.shape_emb, pose_dp.appearance_emb, pose_dp.abs_pose, self.hparams
  #   #   )
  #   #   pose_target.convert_to_torch_from_numpy()

  #   pose_dp = dp.object_poses[0]
  #   pose_target = OBBOutput(
  #       pose_dp.heat_map, pose_dp.shape_emb, pose_dp.appearance_emb, pose_dp.abs_pose, self.hparams
  #   )
  #   pose_target.convert_to_torch_from_numpy()
  
  #   scene_name = dp.scene_name
  #   return anaglyph, segmentation_target, depth_target, pose_target, dp.detections, scene_name
  
  def pose_convert_to_torch_from_numpy(self, heatmap, shape_emb, appearance_emb, abs_pose_field):
    #latent embedding shape
    shape_emb = shape_emb.transpose((2, 0, 1))
    shape_emb = 100.0 * shape_emb
    shape_emb = torch.from_numpy(np.ascontiguousarray(shape_emb)).float()

    #latent embedding appearance
    appearance_emb = appearance_emb.transpose((2, 0, 1))
    appearance_emb = 100.0 * appearance_emb
    appearance_emb = torch.from_numpy(np.ascontiguousarray(appearance_emb)).float()

    #abs pose
    abs_pose_field =abs_pose_field.transpose((2, 0, 1))
    abs_pose_field = 100.0 * abs_pose_field
    abs_pose_field = torch.from_numpy(np.ascontiguousarray(abs_pose_field)).float()
    heatmap = torch.from_numpy(np.ascontiguousarray(heatmap)).float()

    return heatmap, shape_emb, appearance_emb, abs_pose_field

  def __getitem__(self, idx):
    dp = self.datapoint_handles[idx].read()
    anaglyph = self.preprocces_image_func(dp.stereo)
    # segmentation_target = SegmentationOutput(dp.segmentation, self.hparams)
    segmentation_target = dp.segmentation

    segmentation_target.convert_to_torch_from_numpy()
    segmentation_target = torch.from_numpy(segmentation_target).long()


    # depth_target = DepthOutput(dp.depth, self.hparams)
    # depth_target.convert_to_torch_from_numpy()

    depth_target = dp.depth
    depth_target = torch.from_numpy(depth_target).float()

    heatmap, shape_emb, appearance_emb, abs_pose_field = self.pose_convert_to_torch_from_numpy(
      dp.object_poses[0].heat_map, dp.object_poses[0].shape_emb, dp.object_poses[0].appearance_emb, dp.object_poses[0].abs_pose
    )



    
    # pose_target = None
    # # for pose_dp in dp.object_poses:
    # #   pose_target = OBBOutput(
    # #       pose_dp.heat_map, pose_dp.shape_emb, pose_dp.appearance_emb, pose_dp.abs_pose, self.hparams
    # #   )
    # #   pose_target.convert_to_torch_from_numpy()

    # pose_dp = dp.object_poses[0]
    # pose_target = OBBOutput(
    #     pose_dp.heat_map, pose_dp.shape_emb, pose_dp.appearance_emb, pose_dp.abs_pose, self.hparams
    # )
    # pose_target.convert_to_torch_from_numpy()
  
    # scene_name = dp.scene_name
    # return anaglyph, segmentation_target, depth_target, pose_target, dp.detections, scene_name
    return anaglyph, segmentation_target, depth_target, heatmap, shape_emb, appearance_emb, abs_pose_field
  

def get_config_value(hparams, prefix, key):
  full_key = "{}_{}".format(prefix, key)
  if hasattr(hparams, full_key):
    return getattr(hparams, full_key)
  else:
    return None
  
if __name__ == "__main__":
  
  from simnet.lib.net import common
  import argparse

  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  hparams = parser.parse_args()


  path = get_config_value(hparams, 'train', 'path')

  train_ds = datapoint.make_dataset(hparams.train_path)

  dataset = Dataset(
          path, hparams, preprocess_image_func=None, datapoint_dataset=train_ds
      )
  
  print(len(dataset))

  anaglyph, segmentation_target, depth_target, heatmap, shape_emb, appearance_emb, abs_pose_field = dataset[0]

  print("anaglyph, segmentation_target, depth_target, heatmap, shape_emb, appearance_emb, abs_pose_field", anaglyph, segmentation_target, depth_target, heatmap, shape_emb, appearance_emb, abs_pose_field)