import os
import pathlib
import math
import glob
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import _pickle as cPickle
from tqdm import tqdm
from utils.nocs_utils import load_depth, align_nocs_to_depth
from simnet.lib import camera
from simnet.lib import transform
from simnet.lib import datapoint
from simnet.lib.net.pre_processing import obb_inputs
from simnet.lib.depth_noise import DepthManager
from simnet.lib.net.models.auto_encoder import PointCloudAE
from PIL import Image
import argparse
from pathlib import Path
import json

def create_img_list(data_dir):
    """ Create train/val/test data list for CAMERA and Real. """
    # CAMERA dataset
    for subset in ['train', 'val']:
        img_list = []
        img_dir = os.path.join(data_dir, 'CAMERA', subset)
        folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        for i in range(10*len(folder_list)):
            folder_id = int(i) // 10
            img_id = int(i) % 10
            img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
            img_list.append(img_path)
        with open(os.path.join(data_dir, 'CAMERA', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    # Real dataset
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, 'Real', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')

def align_rotation(R):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    return rotation

def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. """
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754' or model_id == 'd3b53f56b4a7b3b3c9f016d57db96408':
                continue
            # process foreground objects
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_camera_train(data_dir, start, end):
    DATASET_NAME = 'ShAPO_Data'
    DATASET_DIR = pathlib.Path(f'/data_2/{DATASET_NAME}')
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # _DATASET = datapoint.make_dataset(f's3://scratch-tri-global/zubair.irshad/{DATASET_DIR}/CAMERA/train')
    _DATASET = datapoint.make_dataset(f'file://{DATASET_DIR}/CAMERA/train')

    _camera = camera.NOCS_Camera()
    """ Generate gt labels for CAMERA train data. """
    camera_train = open(os.path.join(data_dir, 'CAMERA', 'train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    #TEST MODELS
    obj_model_dir = os.path.join(data_dir, 'obj_models')
    with open(os.path.join(obj_model_dir, 'camera_train.pkl'), 'rb') as f:
        obj_models = cPickle.load(f)

    camera_train = camera_train[start:end]
    rgb_feat_rgb_only = load_rgb_latent_dict(data_dir, 'reconstructor.pt')

    valid_img_list = []
    for img_path in tqdm(camera_train):
        img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
        depth_composed_path = img_path+'_composed.png'
        depth_full_path = os.path.join(data_dir,'camera_full_depths', depth_composed_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt') and \
                    os.path.exists(depth_full_path)
        if not all_exist:
            continue
        depth = load_depth(depth_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
        if error_messages:
            continue
        # re-label for mug category
        for i in range(len(class_ids)):
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = translations[i] - scales[i] * rotations[i] @ T0
                s = scales[i] / s0
                scales[i] = s
                translations[i] = T

        #Get SDF and RGB latent embeddings
        latent_embeddings = get_sdf_latent_embeddings(data_dir, model_list)
        latent_embeddings_rgb = get_rgb_embeddings(model_list,rgb_feat_rgb_only)

        #get poses 
        abs_poses=[]
        seg_mask = np.zeros([_camera.height, _camera.width])
        masks_list = []
        for i in range(len(class_ids)):
            R = rotations[i]
            T = translations[i]
            s = scales[i]
            sym_ids = [0, 1, 3]
            cat_id = np.array(class_ids)[i] - 1 
            if cat_id in sym_ids:
                R = align_rotation(R)
            
            scale_matrix = np.eye(4)
            scale_mat = s*np.eye(3, dtype=float)
            scale_matrix[0:3, 0:3] = scale_mat
            camera_T_object = np.eye(4)
            camera_T_object[:3,:3] = R
            camera_T_object[:3,3] = T
            seg_mask[masks[:,:,i] > 0] = np.array(class_ids)[i]
            masks_list.append(masks[:,:,i])
            abs_poses.append(transform.Pose(camera_T_object=camera_T_object, scale_matrix=scale_matrix))

        # emb_dim = 64 for SDF and 128 for pointclouds
        obb_datapoint = obb_inputs.compute_nocs_network_targets(masks_list, latent_embeddings, latent_embeddings_rgb, abs_poses,_camera.height, _camera.width, emb_dim = 64)
        color_img = cv2.imread(img_full_path + '_color.png')
        colorjitter = transforms.ColorJitter(0.5, 0.3, 0.5, 0.05)
        rgb_img = colorjitter(Image.fromarray(color_img))
        jitter_img = colorjitter(rgb_img)
        color_img = np.asarray(jitter_img)
        depth_array = np.array(depth, dtype=np.float32)/255.0
        DM = DepthManager()
        noisy_depth  = DM.prepare_depth_data(depth_array)
        stereo_datapoint = datapoint.Stereo(left_color=color_img, right_color=noisy_depth)
        panoptic_datapoint = datapoint.Panoptic(
        stereo=stereo_datapoint,
        depth=depth_array,
        segmentation=seg_mask,
        object_poses=[obb_datapoint],
        boxes=[],
        detections=[]
        )
        _DATASET.write(panoptic_datapoint)
        ### Finish writing datapoint

def annotate_real_train(data_dir, start, end):

    DATASET_NAME = 'ShAPO_Data'
    DATASET_DIR = pathlib.Path(f'/data_2/{DATASET_NAME}')
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # _DATASET = datapoint.make_dataset(f's3://scratch-tri-global/zubair.irshad/{DATASET_DIR}/Real/train')
    _DATASET = datapoint.make_dataset(f'file://{DATASET_DIR}/Real/train')


    """ Generate gt labels for Real train data through PnP. """
    _camera = camera.NOCS_Real()
    real_train = open(os.path.join(data_dir, 'Real/train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # scale factors for all instances
    scale_factors = {}
    path_to_size = glob.glob(os.path.join(data_dir, 'obj_models/real_train', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]
        bbox_dims = np.loadtxt(inst_path)
        scale_factors[instance] = np.linalg.norm(bbox_dims)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)
    #TEST MODELS
    obj_model_dir = os.path.join(data_dir, 'obj_models')
    with open(os.path.join(obj_model_dir, 'real_train.pkl'), 'rb') as f:
        obj_models = cPickle.load(f)

    real_train = real_train[start:end]
    rgb_feat_rgb_only = load_rgb_latent_dict(data_dir, 'reconstructor.pt')

    valid_img_list = []

    # augment this data 4 times with random Colorjitter 
    for _ in range(4):
        for img_path in tqdm(real_train):
            img_full_path = os.path.join(data_dir, 'Real', img_path)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_coord.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_mask.png') and \
                        os.path.exists(img_full_path + '_meta.txt')
            if not all_exist:
                continue
            depth_full_path = img_full_path+'_depth.png'
            depth = load_depth(depth_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
            if instance_ids is None:
                continue
            # compute pose
            num_insts = len(class_ids)
            scales = np.zeros(num_insts)
            rotations = np.zeros((num_insts, 3, 3))
            translations = np.zeros((num_insts, 3))
            for i in range(num_insts):
                s = scale_factors[model_list[i]]
                mask = masks[:, :, i]
                idxs = np.where(mask)
                coord = coords[:, :, i, :]
                coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
                coord_pts = coord_pts[:, :, None]
                img_pts = np.array([idxs[1], idxs[0]]).transpose()
                img_pts = img_pts[:, :, None].astype(float)
                distCoeffs = np.zeros((4, 1))    # no distoration
                retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
                assert retval
                R, _ = cv2.Rodrigues(rvec)
                T = np.squeeze(tvec)
                # re-label for mug category
                if class_ids[i] == 6:
                    T0 = mug_meta[model_list[i]][0]
                    s0 = mug_meta[model_list[i]][1]
                    T = T - s * R @ T0
                    s = s / s0
                scales[i] = s
                rotations[i] = R
                translations[i] = T

            #Get SDF and RGB latent embeddings
            latent_embeddings = get_sdf_latent_embeddings(data_dir, model_list)
            latent_embeddings_rgb = get_rgb_embeddings(model_list,rgb_feat_rgb_only)

            #get poses 
            abs_poses=[]
            class_num=1
            seg_mask = np.zeros([_camera.height, _camera.width])
            masks_list = []
            for i in range(len(class_ids)):
                R = rotations[i]
                T = translations[i]
                s = scales[i]
                sym_ids = [0, 1, 3]
                cat_id = np.array(class_ids)[i] - 1 
                if cat_id in sym_ids:
                    R = align_rotation(R)
                scale_matrix = np.eye(4)
                scale_mat = s*np.eye(3, dtype=float)
                scale_matrix[0:3, 0:3] = scale_mat
                camera_T_object = np.eye(4)
                camera_T_object[:3,:3] = R
                camera_T_object[:3,3] = T
                seg_mask[masks[:,:,i] > 0] = np.array(class_ids)[i]
                class_num += 1
                masks_list.append(masks[:,:,i])
                abs_poses.append(transform.Pose(camera_T_object=camera_T_object, scale_matrix=scale_matrix))

            # emb_dim = 64 for SDF and 128 for pointclouds
            obb_datapoint = obb_inputs.compute_nocs_network_targets(masks_list, latent_embeddings, latent_embeddings_rgb, abs_poses,_camera.height, _camera.width, emb_dim = 64)
            color_img = cv2.imread(img_full_path + '_color.png')
            colorjitter = transforms.ColorJitter(0.8, 0.5, 0.5, 0.05)
            rgb_img = colorjitter(Image.fromarray(color_img))
            jitter_img = colorjitter(rgb_img)
            color_img = np.asarray(jitter_img)
            depth_array = np.array(depth, dtype=np.float32)/255.0
            DM = DepthManager()
            noisy_depth  = DM.prepare_depth_data(depth_array)
            stereo_datapoint = datapoint.Stereo(left_color=color_img, right_color=noisy_depth)
            panoptic_datapoint = datapoint.Panoptic(
            stereo=stereo_datapoint,
            depth=depth_array,
            segmentation=seg_mask,
            object_poses=[obb_datapoint],
            boxes=[],
            detections=[]
            )
            _DATASET.write(panoptic_datapoint)
        ### Finish writing datapoint

def annotate_test_data(data_dir, source, subset, start, end):
    """ Generate gt labels for test data.
        Properly copy handle_visibility provided by NOCS gts.
    """
    DATASET_NAME = 'ShAPO_Data'
    DATASET_DIR = pathlib.Path(f'/data/{DATASET_NAME}')
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # _DATASET = datapoint.make_dataset(f's3://scratch-tri-global/zubair.irshad/{DATASET_DIR}/{source}/{subset}')
    _DATASET = datapoint.make_dataset(f'file://{DATASET_DIR}/{source}/{subset}')
        
    # compute model size
    model_file_path = ['obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
    models = {}
    for path in model_file_path:
        with open(os.path.join(data_dir, path), 'rb') as f:
            models.update(cPickle.load(f))
    model_sizes = {}
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)
    
    if source == 'CAMERA':
        _camera = camera.NOCS_Camera()
        camera_val = open(os.path.join(data_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()
        camera_val = camera_val[start:end]
        camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
        subset_meta = [('CAMERA', camera_val, camera_intrinsics, 'val')]
    else:
        _camera = camera.NOCS_Real()
        real_test = open(os.path.join(data_dir, 'Real', 'test_list_all.txt')).read().splitlines()
        real_test = real_test[start:end]
        real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        subset_meta = [('Real', real_test, real_intrinsics, 'test')]
    # subset_meta = [('CAMERA', camera_val, camera_intrinsics, 'val'), ('Real', real_test, real_intrinsics, 'test')]
    for source, img_list, intrinsics, subset in subset_meta:
        valid_img_list = []
        # img_list = np.array(img_list)[np.array([2, 500, 1000, 1500, 1700, 1300, 2000, 2300, 2350, 2750])].tolist()
        for img_path in tqdm(img_list):
            img_full_path = os.path.join(data_dir, source, img_path)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_coord.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_mask.png') and \
                        os.path.exists(img_full_path + '_meta.txt')
            if not all_exist:
                continue
            if source == 'CAMERA':
                depth_composed_path = img_path+'_composed.png'
                depth_full_path = os.path.join(data_dir,'camera_full_depths', depth_composed_path)
            else:
                depth_full_path = img_full_path + '_depth.png'
            if not os.path.exists(depth_full_path):
                continue
            depth = load_depth(depth_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
            if instance_ids is None:
                continue
            num_insts = len(instance_ids)
            # match each instance with NOCS ground truth to properly assign gt_handle_visibility
            nocs_dir = os.path.join(data_dir, 'results/nocs_results')
            if source == 'CAMERA':
                nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            else:
                nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            with open(nocs_path, 'rb') as f:
                nocs = cPickle.load(f)
            gt_class_ids = nocs['gt_class_ids']
            gt_bboxes = nocs['gt_bboxes']
            gt_sRT = nocs['gt_RTs']
            gt_handle_visibility = nocs['gt_handle_visibility']
            map_to_nocs = []
            for i in range(num_insts):
                gt_match = -1
                for j in range(len(gt_class_ids)):
                    if gt_class_ids[j] != class_ids[i]:
                        continue
                    if np.sum(np.abs(bboxes[i] - gt_bboxes[j])) > 5:
                        continue
                    # match found
                    gt_match = j
                    break
                # check match validity
                assert gt_match > -1, print(img_path, instance_ids[i], 'no match for instance')
                assert gt_match not in map_to_nocs, print(img_path, instance_ids[i], 'duplicate match')
                map_to_nocs.append(gt_match)
            # copy from ground truth, re-label for mug category
            handle_visibility = gt_handle_visibility[map_to_nocs]
            sizes = np.zeros((num_insts, 3))
            poses = np.zeros((num_insts, 4, 4))
            scales = np.zeros(num_insts)
            rotations = np.zeros((num_insts, 3, 3))
            translations = np.zeros((num_insts, 3))
            for i in range(num_insts):
                gt_idx = map_to_nocs[i]
                sizes[i] = model_sizes[model_list[i]]
                sRT = gt_sRT[gt_idx]
                s = np.cbrt(np.linalg.det(sRT[:3, :3]))
                R = sRT[:3, :3] / s
                T = sRT[:3, 3]
                # re-label mug category
                if class_ids[i] == 6:
                    T0 = mug_meta[model_list[i]][0]
                    s0 = mug_meta[model_list[i]][1]
                    T = T - s * R @ T0
                    s = s / s0
                # used for test during training
                scales[i] = s
                rotations[i] = R
                translations[i] = T
                # used for evaluation
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = s * R
                sRT[:3, 3] = T
                poses[i] = sRT

            ### GET CENTERPOINT DATAPOINTS
            # #get latent embeddings (Incomment this for point latent embeddings)
            # model_points = [models[model_list[i]].astype(np.float32) for i in range(len(class_ids))]
            # latent_embeddings = get_latent_embeddings(model_points, estimator)
            
            #Get SF latent embeddings
            latent_embeddings = get_sdf_latent_embeddings_val(data_dir, model_list)
            # print("latent_embeddings", latent_embeddings[0].shape)
            # Here we use random rgb embeddings since we don't train RGB MLP on validaiton set and hence 
            # make early stopping decision based on SDF and pose loss
            # latent_embeddings_rgb = get_rgb_embeddings(model_list,rgb_feat_rgb_only)
            latent_embeddings_rgb = [np.random.randn(64) for i in range(len(latent_embeddings))]
            
            #get poses 
            abs_poses=[]
            class_num=1
            seg_mask = np.zeros([_camera.height, _camera.width])
            masks_list = []
            for i in range(len(class_ids)):
                R = rotations[i]
                T = translations[i]
                s = scales[i]
                sym_ids = [0, 1, 3]
                cat_id = np.array(class_ids)[i] - 1 
                if cat_id in sym_ids:
                    R = align_rotation(R)
                scale_matrix = np.eye(4)
                scale_mat = s*np.eye(3, dtype=float)
                scale_matrix[0:3, 0:3] = scale_mat
                camera_T_object = np.eye(4)
                camera_T_object[:3,:3] = R
                camera_T_object[:3,3] = T
                seg_mask[masks[:,:,i] > 0] = np.array(class_ids)[i]
                class_num += 1
                masks_list.append(masks[:,:,i])
                abs_poses.append(transform.Pose(camera_T_object=camera_T_object, scale_matrix=scale_matrix))
            obb_datapoint = obb_inputs.compute_nocs_network_targets(masks_list, latent_embeddings, latent_embeddings_rgb,  abs_poses,_camera.height, _camera.width, emb_dim=64)

            color_img = cv2.imread(img_full_path + '_color.png')
            depth_array = np.array(depth, dtype=np.float32)/255.0

            # DM = DepthManager()
            # noisy_depth  = DM.prepare_depth_data(depth_array)
            stereo_datapoint = datapoint.Stereo(left_color=color_img, right_color=depth_array)
            panoptic_datapoint = datapoint.Panoptic(
            stereo=stereo_datapoint,
            depth=depth_array,
            segmentation=seg_mask,
            object_poses=[obb_datapoint],
            boxes=[],
            detections=[]
            )
            _DATASET.write(panoptic_datapoint)
            ### Finish writing datapoint

def load_latent_code(filename):
    data = torch.load(filename)
    num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape
    lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)
    lat_vecs.load_state_dict(data["latent_codes"])
    return lat_vecs.weight.data.detach()

def get_sdf_latent_embeddings(data_dir, model_list):
    latent_embeddings =[]
    LATENT_CODE_DIR_NAME = data_dir
    latent_codes_subdir = "LatentCodes"
    checkpoint = '2000'
    for i in range(len(model_list)):
        category_name = 'sdf_rgb_pretrained'
        latent_filename = os.path.join(
        LATENT_CODE_DIR_NAME, category_name, latent_codes_subdir, checkpoint + ".pth"
        )
        latent_vectors = load_latent_code(latent_filename)
        instance_filename = os.path.join(
            LATENT_CODE_DIR_NAME,category_name,latent_codes_subdir,'all_train_ids.json'
        )
        instance_filename = Path(instance_filename)
        with open(instance_filename, "r") as f:
            instance_ids = json.load(f)
        index = instance_ids[model_list[i]]
        latent_embeddings.append(latent_vectors[index].cpu().numpy())
    return latent_embeddings

def get_sdf_latent_embeddings_val(data_dir, model_list):
    latent_embeddings =[]
    # category_name = 'sdf_rgb_pretrained'
    # LATENT_CODE_DIR_NAME = os.path.join(data_dir, category_name,'Reconstructions','2000','Codes')
    for i in range(len(model_list)):
        # latent_filename = os.path.join(
        # LATENT_CODE_DIR_NAME, model_list[i] + ".pth"
        # )
        # latent_vector = load_latent_code_val(latent_filename)
        latent_vector = torch.rand(64)
        latent_embeddings.append(latent_vector.cpu().numpy())
    return latent_embeddings

def load_latent_code_val(filename):
    data = torch.load(filename)
    return data.data.detach().squeeze(0).squeeze(0)

def load_rgb_latent_dict(data_dir, name):
    category_name = 'sdf_rgb_pretrained'
    rgbnet_dict = torch.load(os.path.join(data_dir, category_name,'rgb_net_weights', name))
    rgb_feat = rgbnet_dict['feats_rgb']
    return rgb_feat

def get_rgb_embeddings(model_list, rgb_feat):
    latent_embeddings =[]
    for i in range(len(model_list)):
        model_id = model_list[i]
        if model_id == '70172e6afe6aff7847f90c1ac631b97e':
            model_id = '70172e6afe6aff7847f90c1ac631b97f'
        latent_embeddings.append(rgb_feat[model_id].detach().cpu().numpy())
    return latent_embeddings

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, required=True)
  args = parser.parse_args()
  data_dir = args.data_dir
  print("Generating image lists")
  create_img_list(data_dir)
  print("Image lists generated...\n")
  
  # All other scripts support dataparallel
#   print("Generating Camera Train data...")
#   annotate_camera_train(data_dir, start, end)
#   print("Generating Real Train data...")
#   annotate_real_train(data_dir, estimator)
#   print("Generating Camera Val data...")
#   annotate_test_data(data_dir, estimator, 'CAMERA', 'val')
#   print("Generating Real Test data...")
#   annotate_test_data(data_dir, estimator, 'Real', 'test')