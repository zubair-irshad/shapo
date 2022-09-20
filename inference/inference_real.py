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

from utils.nocs_utils import load_img_NOCS, create_input_norm, nms, get_masks_out, get_aligned_masks_segout, get_masked_textured_pointclouds
from utils.viz_utils import depth2inv, viz_inv_depth
from utils.transform_utils import transform_coordinates_3d, calculate_2d_projections
from utils.transform_utils import project, get_absposes
from utils.viz_utils import save_projected_points, draw_bboxes, draw_bboxes_mpl_glow
from sdf_latent_codes.get_surface_pointcloud import get_surface_pointclouds_octgrid_viz, get_surface_pointclouds
from sdf_latent_codes.get_rgb import get_rgbnet, get_rgb_from_rgbnet

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
      shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices = nms(
        shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices, _CAMERA
        )

    # get masks and masked pointclouds of each object in the image
    depth_ = np.array(depth, dtype=np.float32)*255.0
    masks_out = get_masks_out(seg_output, depth_)
    masks_out = get_aligned_masks_segout(masks_out, output_indices, depth_)
    masked_pointclouds, areas, masked_rgb = get_masked_textured_pointclouds(masks_out, depth_, left_linear[:,:,::-1], camera = _CAMERA)

    cv2.imwrite(
        str(output_path / f'{i}_image.png'),
        np.copy(np.copy(img_vis))
    )
    cv2.imwrite(
        str(output_path / f'{i}_peaks_output.png'),
        np.copy(img_output)
    )
    depth_vis = depth2inv(torch.tensor(depth).unsqueeze(0).unsqueeze(0))
    depth_vis = viz_inv_depth(depth_vis)
    depth_vis = depth_vis*255.0
    cv2.imwrite(
        str(output_path / f'{i}_depth_vis.png'),
        np.copy(depth_vis)
    )

    rotated_pcds = []
    points_2d = []
    box_obb = []
    axes = [] 
    lod = 7 # Choose from LOD 3-7 here, going higher means more memory and finer details

    # Here we visualize the output of our network
    for j in range(len(shape_emb_outputs)):
        shape_emb = shape_emb_outputs[j]
        # appearance_emb = appearance_emb_putputs[j]
        appearance_emb = appearance_emb_outputs[j]
        is_oct_grid = True
        if is_oct_grid:
            pcd_dsdf, nrm_dsdf = get_surface_pointclouds_octgrid_viz(shape_emb, lod=lod, sdf_pretrained_dir=sdf_pretrained_dir)
        else:
            pcd_dsdf = get_surface_pointclouds(shape_emb)

        rgbnet = get_rgbnet(rgb_model_dir)
        pred_rgb = get_rgb_from_rgbnet(shape_emb, pcd_dsdf, appearance_emb, rgbnet)
        rotated_pc, rotated_box, _ = get_absposes(abs_pose_outputs[j], pcd_dsdf, camera_model = _CAMERA)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.copy(rotated_pc))
        pcd.colors = o3d.utility.Vector3dVector(pred_rgb)
        pcd.normals = o3d.utility.Vector3dVector(nrm_dsdf)
        rotated_pcds.append(pcd)
        points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
        points_2d.append(project(_CAMERA.K_matrix, points_mesh).T)
        #2D output
        points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
        box_obb.append(project(_CAMERA.K_matrix, points_obb).T)
        xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
        sRT = abs_pose_outputs[j].camera_T_object @ abs_pose_outputs[j].scale_matrix
        transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
        axes.append(calculate_2d_projections(transformed_axes, _CAMERA.K_matrix[:3,:3]))
        
    save_projected_points(img_vis, points_2d, str(output_path), str(output_path), i)
    colors_box = [(234, 237, 63)]
    colors_mpl = ['#08F7FE']
    im = np.array(np.copy(img_vis)).copy()
    plt.figure()
    plt.xlim((0, im.shape[1]))
    plt.ylim((0, im.shape[0]))
    plt.gca().invert_yaxis()
    plt.axis('off')
    for k in range(len(colors_box)):
        for points_2d, axis in zip(box_obb, axes):
            points_2d = np.array(points_2d)
            im = draw_bboxes_mpl_glow(im, points_2d, axis, colors_mpl[k])
    plt.imshow(im[...,::-1])
    box_plot_name = str(output_path)+'/box3d_'+str(i)+'.png'
    plt.savefig(box_plot_name, bbox_inches='tight',pad_inches = 0)
    print("done with image: ", i )

if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  app_group = parser.add_argument_group('app')
  app_group.add_argument('--app_output', default='inference_best', type=str)
  app_group.add_argument('--result_name', default='centersnap_nocs', type=str)
  app_group.add_argument('--data_dir', default='nocs_data', type=str)

  hparams = parser.parse_args()
  print(hparams)
  result_name = hparams.result_name
  path = 'data/'+result_name
  output_path = pathlib.Path(path) / hparams.app_output
  output_path.mkdir(parents=True, exist_ok=True)
  inference(hparams, hparams.data_dir, output_path)
