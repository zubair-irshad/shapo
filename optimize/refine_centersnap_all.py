import argparse
import pathlib
import time
from simnet.lib import transform
import cv2
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import open3d as o3d
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import iterative_closest_point
import matplotlib.pyplot as plt
import os
import time
import trimesh
import pytorch_lightning as pl
import _pickle as cPickle
import time
from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.dataset import Dataset, extract_left_numpy_img
from sdf_latent_codes.save_canonical_mesh import save_mesh
from sdf_latent_codes.get_surface_pointcloud import get_surface_pointclouds, get_grid_sdfnet
from app.utils.nocs_utils import get_mug_meta, get_real_scales, load_img_NOCS, process_data, create_anaglyph_w_depth, create_anaglyph_norm, load_mcrnn_results, get_aligned_masks,get_aligned_masks_nocs, get_masked_pointclouds, get_mug_meta, get_real_scales, get_gt_RTS
from app.utils.viz_utils import depth2inv, viz_inv_depth
from app.utils.transform_utils import get_pointclouds_abspose, get_pointclouds_gtpose, transform_coordinates_3d, calculate_2d_projections
from app.utils.transform_utils import project, get_3D_rotated_box, get_trimesh_scales
from app.utils.viz_utils import draw_registration_result, save_projected_points, draw_bboxes, object_key_to_name, line_set_mesh, custom_draw_geometry_with_rotation, draw_registration_result_scene
from app.panoptic_tidying.align.optimization_all import Optimizer
from tqdm import tqdm
#./runner.sh app/panoptic_tidying/align/refine_centersnap.py @app/panoptic_tidying/configs/net_config.txt --checkpoint /home/zubair/cs_implicit_ckpt/_ckpt_epoch_12.ckpt
def evaluate_dataset(
    hparams,
    input_path,
    output_path,
    min_confidence=0.1,
    overlap_thresh=0.75,
    num_samples=80,
    num_to_draw=20,
    use_gpu=True,
    is_training=False
):
  real_data = True
  model = PanopticModel(hparams, 0, None, None)
  model.eval()
  if use_gpu:
    model.cuda()
  dataset = Dataset(input_path, hparams)
  print('Dataset size:', len(dataset))
  data_dir = '/home/ubuntu/Downloads/nocs_data'
  cam = False
  norm = True
  pcd_output = True
  if cam:
      #CHANGE THIS TO VAL_LIST: NOT ALL DEPTH/IMG PATHS are VALID
    data_path = open(os.path.join(data_dir, 'CAMERA', 'val_list.txt')).read().splitlines()
    data_type = 'val'
    type = 'val'
    _CAMERA = camera.NOCS_Camera()
    min_confidence = 0.50  
  else:
    data_path = open(os.path.join(data_dir, 'Real', 'test_list.txt')).read().splitlines()
    data_type = 'real_test'
    type = 'test'
    _CAMERA = camera.NOCS_Real()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    mug_meta = get_mug_meta(data_dir)
    scale_factors = get_real_scales(data_dir)
    min_confidence = 0.50

  for i, img_path in enumerate(tqdm(data_path)):
    # mcrnn_results = load_mcrnn_results(img_path, data_type)
    if cam:
      img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
    else:
      img_full_path = os.path.join(data_dir, 'Real', img_path)
    if real_data:
      color_path = img_full_path + '_color.png'
      depth_path = img_full_path + '_depth.png'
      if not os.path.exists(color_path):
        continue
      depth_path = img_full_path + '_depth.png'

      if cam:
        depth_composed_path = img_path+'_composed.png'
        depth_full_path = os.path.join(data_dir,'camera_full_depths', depth_composed_path)
      else:
        depth_full_path = depth_path
      if not os.path.exists(depth_full_path):
        continue
      
      left_linear, depth, actual_depth = load_img_NOCS(color_path, depth_full_path)
      masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, actual_depth)
      if norm:
        anaglyph = create_anaglyph_norm(left_linear, depth)
      else:
        anaglyph = create_anaglyph_w_depth(left_linear, depth)
      anaglyph = anaglyph[None, :, :, :]
    else: 
      sample = dataset[i]
      anaglyph, seg_target, depth_target, pose_target, _,_, detections_gt, scene_name = sample
      anaglyph = anaglyph[None, :, :, :]
    if use_gpu:
      anaglyph = anaglyph.to(torch.device('cuda:0'))

    img_path_parsing = img_path.split('/')
    mrcnn_path = os.path.join('/home/ubuntu/Downloads/nocs_data/results/mrcnn_results', data_type, 'results_{}_{}_{}.pkl'.format(type, img_path_parsing[-2], img_path_parsing[-1]))
    with open(mrcnn_path, 'rb') as f:
      mrcnn_result = cPickle.load(f)

    inference_time = time.time()
    with torch.no_grad():
      _, _, _,  pose_output, _, _ = model.forward(anaglyph)
      latent_emb_outputs, abs_pose_vector_outputs, img_output, scores_out, output_indices,  class_ids_predicted = pose_output.compute_pointclouds_and_poses(min_confidence,is_target = False)
    print("inference time", time.time()-inference_time)
    
    depth = np.array(depth, dtype=np.float32)*255.0
    class_ids = np.array(mrcnn_result['class_ids'])
    scores = np.array(mrcnn_result['scores'])

    masks_time = time.time()
    masks, class_ids, scores = get_aligned_masks_nocs(mrcnn_result['masks'], output_indices, class_ids, class_ids_predicted, scores, scores_out, depth)
    print("masks time", time.time()-masks_time )
    
    # gt_sRT = get_gt_RTS(masks, coords, class_ids, model_list, scale_factors, mug_meta, intrinsics)
    # depth = np.array(depth, dtype=np.float32)*255.0
    # masks, class_ids, gt_SRT = get_aligned_masks(masks, out_indices, gt_sRT, depth)
    
    masks_pc_time = time.time()
    masked_pointclouds = get_masked_pointclouds(masks, depth, camera = _CAMERA)
    print("get pc time", time.time()-masks_pc_time)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimization_out = {}
    latent_opt = []
    RT_opt = []
    scale_opt = []
    if pcd_output: 
      for j in range(len(latent_emb_outputs)):
        if masked_pointclouds[j] is not None:
          emb = latent_emb_outputs[j]
          sdf_net, grid3d = get_grid_sdfnet()
          #start optimization for each shape:
          params = {}
          weights = {}
          params['latent'] = emb
          params['RT'] = abs_pose_vector_outputs[j][0:12]
          params['scale'] = np.array(abs_pose_vector_outputs[j][12])
          weights['3d'] = 1
          start_time = time.time()
          optimizer = Optimizer(params, device,weights)
          # Optimize the initial pose estimate
          iters_optim = 100
          optimizer.optimize(
              iters_optim,
              masked_pointclouds[j],
              sdf_net,
              grid3d,
              viz_type=None
          )
          print("optimizaiton time for object, ", j, ":",  time.time()-start_time)
          latent_opt.append(params['latent'].detach().cpu().numpy())
          RT_opt.append(params['RT'].detach().cpu().numpy())
          scale_opt.append(params['scale'].detach().cpu().numpy())
        else:
          latent_opt.append(latent_emb_outputs[j])
          RT_opt.append(abs_pose_vector_outputs[j][0:12])
          scale_opt.append(np.array(abs_pose_vector_outputs[j][12]))

    optimization_out['latent'] = latent_opt
    optimization_out['RT'] = RT_opt
    optimization_out['scale'] = scale_opt
    if cam:
      optim_dir = os.path.join(output_path, 'CAMERA', img_path)
    else:
      optim_dir = os.path.join(output_path, 'Real', img_path)
    os.makedirs(optim_dir, exist_ok=True)
    optim_filename = os.path.join(optim_dir, 'optim.pkl')
    with open(optim_filename, 'wb') as handle:
        pickle.dump(optimization_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("done with image: ", i , "\n\n")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  app_group = parser.add_argument_group('app')
  app_group.add_argument('--app_output', default='validation_inference', type=str)
  app_group.add_argument(
      '--is_training', default=False, type=bool, help='Is tested on validation data or not.'
  )
  hparams = parser.parse_args()
  #for confidence in confidences:
  path = 'data/optimization_real_ckpt13_testtime/'
  output_path = pathlib.Path(path)
  output_path.mkdir(parents=True, exist_ok=True)
  evaluate_dataset(hparams, hparams.val_path, output_path, is_training=hparams.is_training)