import os
import cv2
import argparse
import pathlib

import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.models.auto_encoder import PointCloudAE

from utils.nocs_utils import load_img_NOCS, create_input_norm, get_masks_out, get_aligned_masks_segout, get_masked_textured_pointclouds
from sdf_latent_codes.get_surface_pointcloud import get_sdfnet
from sdf_latent_codes.get_rgb import get_rgbnet
from utils.transform_utils import get_abs_pose_vector_from_matrix, get_abs_pose_from_vector, transform_pcd_to_canonical
from utils.viz_utils import draw_colored_shape, draw_colored_mesh_mcubes
from opt.optimization_all import Optimizer

def inference(
    hparams,
    data_dir, 
    output_path,
    min_confidence=0.1,
    use_gpu=True,
):
  model = PanopticModel(hparams, 0, None, None)
  model.eval()
  if use_gpu:
    model.cuda()
  data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
  sdf_pretrained_dir = os.path.join(data_dir, 'sdf_rgb_pretrained')
  rgb_model_dir = os.path.join(data_dir, 'sdf_rgb_pretrained', 'rgb_net_weights')
  _CAMERA = camera.NOCS_Real()
  min_confidence = 0.50
  sdf_pretrained_dir = os.path.join(data_dir, 'sdf_rgb_pretrained')
  rgb_model_dir = os.path.join(data_dir, 'sdf_rgb_pretrained', 'rgb_net_weights')
  
  data_path = data_path[1]
  for i, img_path in enumerate(data_path):
    img_full_path = os.path.join(data_dir, 'Real', img_path)
    color_path = img_full_path + '_color.png' 
    if not os.path.exists(color_path):
      continue
    depth_full_path = img_full_path + '_depth.png'
    img_vis = cv2.imread(color_path)
    left_linear, depth, actual_depth = load_img_NOCS(color_path, depth_full_path)
    input = create_input_norm(left_linear, depth)
    input = input[None, :, :, :]
    if use_gpu:
      input = input.to(torch.device('cuda:0'))
    with torch.no_grad():
      seg_output, _, _ , pose_output = model.forward(input)
      shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, img_output, scores_out, output_indices = pose_output.compute_shape_pose_and_appearance(min_confidence,is_target = False)
      #shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices = nms(
      #  shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices, _CAMERA
      #  )

    # get masks and masked pointclouds of each object in the image
    depth_ = np.array(depth, dtype=np.float32)*255.0
    seg_output.convert_to_numpy_from_torch()
    masks_out = get_masks_out(seg_output, depth_)
    masks_out = get_aligned_masks_segout(masks_out, output_indices, depth_)
    masked_pointclouds, areas, masked_rgb = get_masked_textured_pointclouds(masks_out, depth_, left_linear[:,:,::-1], camera = _CAMERA)

    optimization_out = {}
    latent_opt = []
    RT_opt = []
    scale_opt = []

    do_optim = True    
    latent_opt = []
    RT_opt = []
    scale_opt = []
    appearance_opt = []
    colored_opt_pcds = []
    colored_opt_meshes = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psi, theta, phi, t = (0, 0, 0, 0)
    shape_latent_noise = np.random.normal(loc=0, scale=0.02, size=64)
    add_noise = False
    viz_type = None
    # psi, theta, phi, t = 0, 0, 4, 0.05
    if do_optim: 
      for k in range(len(shape_emb_outputs)):
          print("Starting optimization, object:", k, "\n", "----------------------------", "\n")
          if viz_type is not None:
            optim_foldername = str(output_path) + '/optim_images_'+str(k)
            if not os.path.exists(optim_foldername):
                os.makedirs(optim_foldername)
          else:
            optim_foldername = None
          #optimization starts here:
          abs_pose = abs_pose_outputs[k]
          mask_area = areas[k]
          RT, s = get_abs_pose_vector_from_matrix(abs_pose.camera_T_object, abs_pose.scale_matrix, add_noise = False)
          
          if masked_pointclouds[k] is not None:
            shape_emb = shape_emb_outputs[k]
            appearance_emb = appearance_emb_outputs[k]
            decoder = get_sdfnet(sdf_latent_code_dir = sdf_pretrained_dir)
            rgbnet = get_rgbnet(rgb_model_dir)
            params = {}
            weights = {}

            if add_noise:
              shape_emb += shape_latent_noise
            
            #Set latent vectors to optimize
            params['latent'] = shape_emb
            params['RT'] = RT
            params['scale'] = np.array(s)
            params['appearance'] = appearance_emb
            weights['3d'] = 1

            optimizer = Optimizer(params, rgbnet, device, weights, mask_area)
            # Optimize the initial pose estimate
            iters_optim = 200
            optimizer.optimize_oct_grid(
                iters_optim,
                masked_pointclouds[k],
                masked_rgb[k],
                decoder,
                rgbnet, 
                optim_foldername, 
                viz_type='3d'
            )
            #save latent vectors after optimization
            latent_opt.append(params['latent'].detach().cpu().numpy())
            RT_opt.append(params['RT'].detach().cpu().numpy())
            scale_opt.append(params['scale'].detach().cpu().numpy())
            appearance_opt.append(params['appearance'].detach().cpu().numpy())
            abs_pose = get_abs_pose_from_vector(params['RT'].detach().cpu().numpy(), params['scale'].detach().cpu().numpy())
            obj_colored = draw_colored_shape(params['latent'].detach().cpu().numpy(), abs_pose, params['appearance'].detach().cpu().numpy(), rgbnet, sdf_pretrained_dir, is_oct_grid=True)
            colored_opt_pcds.append(obj_colored)
          else:
            latent_opt.append(shape_emb_outputs[k])
            RT_opt.append(RT)
            scale_opt.append(np.array(s))
            appearance_opt.append(appearance_emb_outputs[k])
            print("Done with optimization, object:", k, "\n", "----------------------------", "\n")
    o3d.visualization.draw_geometries(colored_opt_pcds)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  app_group = parser.add_argument_group('app')
  app_group.add_argument('--app_output', default='optimize', type=str)
  app_group.add_argument('--result_name', default='ShAPO_Real', type=str)
  app_group.add_argument('--data_dir', default='nocs_data', type=str)

  hparams = parser.parse_args()
  print(hparams)
  result_name = hparams.result_name
  path = 'results/'+result_name
  output_path = pathlib.Path(path) / hparams.app_output
  output_path.mkdir(parents=True, exist_ok=True)
  inference(hparams, hparams.data_dir, output_path)
