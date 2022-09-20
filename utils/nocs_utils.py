import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from utils.transform_utils import get_2d_box
from sdf_latent_codes.get_surface_pointcloud import  get_surface_pointclouds_octgrid_sparse
from torchvision import ops

def load_depth(depth_path):
    """ Load depth image from img_path. """
    # depth_path = depth_path + '_depth.png'
    # print("depth_path", depth_path)
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def load_img_NOCS(color, depth):
  left_img = cv2.imread(color)
  actual_depth = load_depth(depth)
  right_img = np.array(actual_depth, dtype=np.float32)/255.0
  return left_img, right_img, actual_depth

def create_input_w_depth(left_color,right_color ):
  height, width, _ = left_color.shape
  image = np.zeros([height, width, 4], dtype=np.uint8)
  cv2.normalize(left_color, left_color, 0, 255, cv2.NORM_MINMAX)
  # cv2.normalize(stereo_dp.right_color, stereo_dp.right_color, 0, 255, cv2.NORM_MINMAX)
  image[..., 0:3] = left_color
  image = image * 1. / 255.0
  if len(right_color.shape  ) == 2:
    image[..., 3] = right_color
  # print(image.shape)
  image = image.transpose((2, 0, 1))
  return torch.from_numpy(np.ascontiguousarray(image)).float()

def create_input_norm(left_color,right_color):
  height, width, _ = left_color.shape
  image = torch.zeros(4, height, width, dtype=torch.float32)
  cv2.normalize(left_color, left_color, 0, 255, cv2.NORM_MINMAX)

  rgb = left_color* 1. / 255.0
  norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  rgb = norm(torch.from_numpy(rgb.astype(np.float32).transpose((2,0,1))))

  if len(right_color.shape) == 2:
    depth = right_color
    depth = torch.from_numpy(depth.astype(np.float32))

  image[0:3, :] = rgb
  image[3, :] = depth
  return image


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

"""
    RANSAC for Similarity Transformation Estimation
    Modified from https://github.com/hughw19/NOCS_CVPR2019
    Originally Written by Srinath Sridhar
"""
import time
import numpy as np


def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # rotation
    Rotation = np.matmul(U, Vh)
    # scale
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    Scale = 1 / varP * np.sum(D)
    # translation
    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(Scale*Rotation.T)
    # transformation matrix
    OutTransform = np.identity(4)
    OutTransform[:3, :3] = Scale * Rotation
    OutTransform[:3, 3] = Translation

    return Scale, Rotation, Translation, OutTransform


def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
    """ Add RANSAC algorithm to account for outliers.

    """
    assert source.shape[0] == target.shape[0], 'Source and Target must have same number of points.'
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
    # Auto-parameter selection based on source heuristics
    # Assume source is object model or gt nocs map, which is of high quality
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    SourceDiameter = 2 * np.amax(np.linalg.norm(CenteredSource, axis=0))
    InlierT = SourceDiameter / 10.0  # 0.1 of source diameter
    maxIter = 128
    confidence = 0.99

    if verbose:
        print('Inlier threshold: ', InlierT)
        print('Max number of iterations: ', maxIter)

    BestInlierRatio = 0
    BestInlierIdx = np.arange(nPoints)
    for i in range(0, maxIter):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(nPoints, size=5)
        Scale, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        PassThreshold = Scale * InlierT    # propagate inlier threshold to target scale
        Diff = TargetHom - np.matmul(OutTransform, SourceHom)
        ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
        InlierIdx = np.where(ResidualVec < PassThreshold)[0]
        nInliers = InlierIdx.shape[0]
        InlierRatio = nInliers / nPoints
        # update best hypothesis
        if InlierRatio > BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if verbose:
            print('Iteration: ', i)
            print('Inlier ratio: ', BestInlierRatio)
        # early break
        if (1 - (1 - BestInlierRatio ** 5) ** i) > confidence:
            break

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    SourceInliersHom = SourceHom[:, BestInlierIdx]
    TargetInliersHom = TargetHom[:, BestInlierIdx]
    Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scale:', Scale)

    return Scale, Rotation, Translation, OutTransform


def backproject(depth, intrinsics, instance_mask):
    """ Back-projection, use opencv camera coordinate frame.

    """
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]

    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    idxs = np.where(final_instance_mask)

    z = depth[idxs[0], idxs[1]]
    x = (idxs[1] - cam_cx) * z / cam_fx
    y = (idxs[0] - cam_cy) * z / cam_fy
    pts = np.stack((x, y, z), axis=1)

    return pts, idxs


def align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path, verbose=False):
    num_instances = len(instance_ids)
    error_messages = ''
    elapses = []
    scales = np.zeros(num_instances)
    rotations = np.zeros((num_instances, 3, 3))
    translations = np.zeros((num_instances, 3))

    for i in range(num_instances):
        mask = masks[:, :, i]
        coord = coords[:, :, i, :]
        pts, idxs = backproject(depth, intrinsics, mask)
        coord_pts = coord[idxs[0], idxs[1], :] - 0.5
        try:
            start = time.time()
            s, R, T, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
            elapsed = time.time() - start
            if verbose:
                print('elapsed: ', elapsed)
            elapses.append(elapsed)
        except Exception as e:
            message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(instance_ids[i], img_path, str(e))
            print(message)
            error_messages += message + '\n'
            s = 1.0
            R = np.eye(3)
            T = np.zeros(3)
            outtransform = np.identity(4, dtype=np.float32)

        scales[i] = s / 1000.0
        rotations[i, :, :] = R
        translations[i, :] = T / 1000.0

    return scales, rotations, translations, error_messages, elapses

def nms(latent_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices, _CAMERA):
    boxes = torch.zeros(len(abs_pose_outputs), 4)
    scores_torch = torch.zeros(len(abs_pose_outputs))
    for p, (pose, score, emb) in enumerate(zip(abs_pose_outputs, scores_out, latent_emb_outputs)):
        pcd_dsdf = get_surface_pointclouds_octgrid_sparse(emb)
        bbox = get_2d_box(pose, pcd_dsdf, camera_model = _CAMERA)
        boxes[p, :] = torch.tensor([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][0]]) 
        scores_torch[p] = torch.tensor(score)

    keep = ops.nms(boxes, scores_torch, iou_threshold=0.30)
    keep = torch.sort(keep)[0]
    keep = np.asarray(keep.numpy())
    latent_emb_outputs = np.array(latent_emb_outputs)[keep]
    appearance_emb_outputs = np.array(appearance_emb_outputs)[keep]
    abs_pose_outputs = np.array(abs_pose_outputs)[keep]
    scores_out= np.array(scores_out)[keep]
    output_indices = np.array(output_indices)[keep]
    return latent_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices

def get_aligned_masks_segout(masks, output_indices, depth):
  mask_out = []
  for p in range(masks.shape[0]):
    mask = np.logical_and(masks[p, :, :], depth > 0)
    mask_out.append(mask)
  mask_out = np.array(mask_out)
  index_centers = []
  for m in range(mask_out.shape[0]):
    pos = np.where(mask_out[m,:,:])
    center_x = np.average(pos[0])
    center_y = np.average(pos[1])
    index_centers.append([center_x, center_y])
  new_masks = []
  index_centers = np.array(index_centers)
  if np.any(np.isnan(index_centers)):
    bool_is_inf = ~np.any(np.isnan(index_centers), axis=1)
    index_centers = index_centers[bool_is_inf]
    mask_out = mask_out[bool_is_inf]
  mask_out = np.array(mask_out)
  for l in range(len(output_indices)):
    point = output_indices[l]
    if len(output_indices) == 0:
      continue
    distances = np.linalg.norm(index_centers-point, axis=1)
    min_index = np.argmin(distances)
    if distances[min_index]<28:
      new_masks.append(mask_out[min_index, :,:])
    else: 
      new_masks.append(None)
  masks = np.array(new_masks)
  return masks

def get_masked_textured_pointclouds(masks, depth,rgb, n_pts=2048, camera= None):
  num_objs = masks.shape[0]
  xmap = np.array([[y for y in range(640)] for z in range(480)])
  ymap = np.array([[z for y in range(640)] for z in range(480)])
  boxes=[]
  areas = []
  for m in range(num_objs):
      if masks[m] is not None:
        pos = np.where(masks[m]>0)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
        areas.append(((ymax-ymin)* (xmax-xmin))/100)
      else:
        boxes.append(None)
        areas.append(None)
  rgbd_points = []
  colors_masked = []
  for h in range(num_objs):
    if masks[h] is not None:
      x1, y1, x2, y2 = boxes[h]
      mask = masks[h]>0
      mask = np.logical_and(mask, depth > 0)
      choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]
      if len(choose) ==0:
        rgbd_points.append(None)
        continue
      cam_cx = camera.c_x
      cam_fx = camera.f_x
      cam_cy = camera.c_y
      cam_fy = camera.f_y
      depth_masked = depth[y1:y2, x1:x2].flatten()[choose][:, np.newaxis]
      xmap_masked = xmap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis]
      ymap_masked = ymap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis]
      rgb_masked = rgb[y1:y2, x1:x2].reshape(-1,3)[choose, :]
      colors_masked.append(rgb_masked/255.0)
      pt2 = depth_masked/1000.0
      pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
      pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
      points = np.concatenate((pt0, pt1, pt2), axis=1)
      rgbd_points.append(points)
    else:
      rgbd_points.append(None)
      colors_masked.append(None)
  return rgbd_points, areas, colors_masked

def get_masks_out(seg_output, depth):
  category_seg_output = np.ascontiguousarray(seg_output.seg_pred)
  category_seg_output = np.argmax(category_seg_output[0], axis=0)
  obj_ids = np.unique(category_seg_output)
  obj_ids = obj_ids[1:]
  masks_target = category_seg_output == obj_ids[:, None, None]
  contour_centers = []
  act_contours = []
  for m in range(len(obj_ids)):
    contours, _ = cv2.findContours(masks_target[m].astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    if len(contours)>1:
      for c in range(len(contours)):
        if cv2.contourArea(contours[c]) < 650:
          continue
        act_contours.append(contours[c])
        moments = cv2.moments(contours[c])
        contour_centers.append((int(moments['m01']/moments['m00']), int(moments['m10']/moments['m00'])))
    else:
      if cv2.contourArea(contours[0]) < 650:
        continue
      act_contours.append(contours[0])
      moments = cv2.moments(contours[0])
      contour_centers.append((int(moments['m01']/moments['m00']), int(moments['m10']/moments['m00']) ))
  contour_centers = np.array(contour_centers)
  mask_out = []
  class_ids = []
  class_ids_name = []
  class_centers = []
  for l in range(len(act_contours)):
    out = np.zeros_like(depth)
    viz = act_contours[l]
    idx = act_contours[l]
    center = contour_centers[l]
    temp = np.copy(contour_centers)
    temp[l] = 1000
    distance = np.linalg.norm(temp-center, axis=1)
    closest_index = np.argmin(distance)
    if distance[closest_index]<30:
      if cv2.contourArea(act_contours[l]) < cv2.contourArea(act_contours[closest_index]):
        continue
    out = np.zeros_like(depth)
    class_id_from_mask = category_seg_output[center[0], center[1]]
    class_ids.append(np.int(class_id_from_mask))
    class_ids_name.append(class_id_from_mask)
    class_centers.append((center[1], center[0]))
    cv2.drawContours(out, [idx], -1, 255, cv2.FILLED, 1)
    mask_out.append(out) 
  mask_out = np.array(mask_out)
  return mask_out