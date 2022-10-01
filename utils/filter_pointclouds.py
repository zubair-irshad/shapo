import os
import numpy as np
from PIL import Image
import cv2
import _pickle as cPickle

import numpngw
from utils.transform_utils import project, get_pc_absposes, transform_pcd_to_canonical
import open3d as o3d
import matplotlib.pyplot as plt

data_list_file = '/home/pwl/Work/Dataset/NOCS/nocs/Real/train_list_all.txt'
data_path = '/home/zubair/Downloads/nocs_data/Real'
result_path = '/home/pwl/Work/Dataset/NOCS/nocs/Real/depth'

cam_fx, cam_fy, cam_cx, cam_cy = [591.0125, 590.16775, 322.525, 244.11084]
xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])

CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}

def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)

    if len(depth.shape) == 3:
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def display_inlier_outlier_all(cloud, ind1, ind2, ind3):
    outlier_cloud_1 = cloud.select_by_index(ind1, invert=True)
    inlier_cloud_1 = cloud.select_by_index(ind1)

    outlier_cloud_2 = inlier_cloud_1.select_by_index(ind2, invert=True)
    inlier_cloud_2 = inlier_cloud_1.select_by_index(ind2)

    outlier_cloud_3 = inlier_cloud_2.select_by_index(ind3, invert=True)
    inlier_cloud_3 = inlier_cloud_2.select_by_index(ind3)

    outlier_cloud_1.paint_uniform_color([1, 0, 0])
    outlier_cloud_2.paint_uniform_color([0, 1, 0])
    outlier_cloud_3.paint_uniform_color([0, 0, 1])
    inlier_cloud_3.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud_3, outlier_cloud_1, outlier_cloud_2, outlier_cloud_3])

def depth2pc(depth, choose, scaled=None):
    depth_masked = depth.flatten()[choose][:, np.newaxis]
    xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    ymap_masked = ymap.flatten()[choose][:, np.newaxis]
    if scaled:
        pt2 = depth_masked/1000.0
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)  
    else:      
        pt2 = depth_masked
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)

    return points

def get_bbox(bbox, img_h, img_w):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = img_h  
    img_length = img_w  
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def outliner_removal_laptop(depth, mask_raw, gts, idx):
    instance_id = gts['instance_ids'][idx]
    mask_i = np.equal(mask_raw, instance_id)
    mask_depth = np.logical_and(mask_i, depth > 0)
    choose = mask_depth.flatten().nonzero()[0]

    points_raw = depth2pc(depth, choose)

    pcd_pc_raw = o3d.geometry.PointCloud()
    pcd_pc_raw.points = o3d.utility.Vector3dVector(points_raw)

    cl, ind_1 = pcd_pc_raw.remove_statistical_outlier(nb_neighbors=80, std_ratio=1.3)
    pcd_pc_inler1 = pcd_pc_raw.select_by_index(ind_1)
    cl, ind_2 = pcd_pc_inler1.remove_statistical_outlier(nb_neighbors=2000, std_ratio=4.5)
    pcd_pc_inler2 = pcd_pc_inler1.select_by_index(ind_2)

    labels = np.array(pcd_pc_inler2.cluster_dbscan(eps=60, min_points=200))

    if(len(labels) == 0):
        return choose


    max_label = labels.max()
    min_label = labels.min()

    biggest_cluster_idx = 0
    biggest_cluster_elem_count = 0
    if max_label >= 1 or min_label == -1:
        for label_idx in range(max_label + 1):
            cluster_elem_count =  len(np.where(labels == label_idx)[0])
            if cluster_elem_count > biggest_cluster_elem_count:
                biggest_cluster_elem_count = len(np.where(labels == label_idx)[0])
                biggest_cluster_idx = label_idx

        ind_3 = list(np.where(labels == biggest_cluster_idx)[0])
    else:
        pcd_pc_inler = pcd_pc_inler2
        ind_3 = np.array(range(labels.shape[0]))

    #display_inlier_outlier_all(pcd_pc_raw, ind_1, ind_2, ind_3)

    choose_f1 = choose[ind_1]
    choose_del1 = np.delete(choose, ind_1)

    choose_f2 = choose_f1[ind_2]
    choose_del2 = np.delete(choose_f1, ind_2)

    choose_f3 = choose_f2[ind_3]
    choose_del3 = np.delete(choose_f2, ind_3)
    
    choose_final = choose_f3
    choose_deleted = list(set(choose)-set(choose_final))

    return choose_deleted


def outliner_removal_bottle_can(depth, mask_raw, gts, idx):
    instance_id = gts['instance_ids'][idx]
    rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx], depth.shape[0], depth.shape[1])

    mask_i = np.equal(mask_raw, instance_id)
    mask_depth = np.logical_and(mask_i, depth > 0)
    choose_mask = mask_depth.flatten().nonzero()[0]

    black_img = np.zeros_like(depth)    
    black_img[rmin:rmax, cmin:cmax] = 1.0
    choose_bbox = black_img.flatten().nonzero()[0]

    points_bbox = depth2pc(depth, choose_bbox)

    pcd_pc_bbox = o3d.geometry.PointCloud()
    pcd_pc_bbox.points = o3d.utility.Vector3dVector(points_bbox)

    plane_model, inliers_plane = pcd_pc_bbox.segment_plane(distance_threshold=5,
                                                    ransac_n=10,
                                                    num_iterations=1000)

    #display_inlier_outlier(pcd_pc_bbox, inliers_plane)

    choose_bbox = np.delete(choose_bbox, inliers_plane)
    choose = np.intersect1d(choose_bbox, choose_mask)

    points_raw = depth2pc(depth, choose)
    pcd_pc_raw = o3d.geometry.PointCloud()
    pcd_pc_raw.points = o3d.utility.Vector3dVector(points_raw)

    #o3d.visualization.draw_geometries([pcd_pc_raw])

    cl, ind_1 = pcd_pc_raw.remove_statistical_outlier(nb_neighbors=80, std_ratio=1.3)
    pcd_pc_inler1 = pcd_pc_raw.select_by_index(ind_1)
    cl, ind_2 = pcd_pc_inler1.remove_statistical_outlier(nb_neighbors=2000, std_ratio=4.5)
    pcd_pc_inler2 = pcd_pc_inler1.select_by_index(ind_2)

    labels = np.array(pcd_pc_inler2.cluster_dbscan(eps=60, min_points=200))

    if(len(labels) == 0):
        return choose_mask

    max_label = labels.max()
    min_label = labels.min()

    biggest_cluster_idx = 0
    biggest_cluster_elem_count = 0
    if max_label >= 1 or min_label == -1:
        for label_idx in range(max_label + 1):
            cluster_elem_count =  len(np.where(labels == label_idx)[0])
            if cluster_elem_count > biggest_cluster_elem_count:
                biggest_cluster_elem_count = len(np.where(labels == label_idx)[0])
                biggest_cluster_idx = label_idx

        ind_3 = list(np.where(labels == biggest_cluster_idx)[0])
    else:
        pcd_pc_inler = pcd_pc_inler2
        ind_3 = np.array(range(labels.shape[0]))

    #display_inlier_outlier_all(pcd_pc_raw, ind_1, ind_2, ind_3)

    choose_f1 = choose[ind_1]
    choose_del1 = np.delete(choose, ind_1)

    choose_f2 = choose_f1[ind_2]
    choose_del2 = np.delete(choose_f1, ind_2)

    choose_f3 = choose_f2[ind_3]
    choose_del3 = np.delete(choose_f2, ind_3)
    
    choose_final = choose_f3
    choose_deleted = list(set(choose_mask)-set(choose_final))

    return choose_deleted
    

def outliner_removal_bowl_mug(depth, mask_raw, gts, idx):
    instance_id = gts['instance_ids'][idx]
    rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx], depth.shape[0], depth.shape[1])

    mask_i = np.equal(mask_raw, instance_id)
    mask_depth = np.logical_and(mask_i, depth > 0)
    choose_mask = mask_depth.flatten().nonzero()[0]

    black_img = np.zeros_like(depth)    
    black_img[rmin:rmax, cmin:cmax] = 1.0
    choose_bbox = black_img.flatten().nonzero()[0]

    points_bbox = depth2pc(depth, choose_bbox)

    pcd_pc_bbox = o3d.geometry.PointCloud()
    pcd_pc_bbox.points = o3d.utility.Vector3dVector(points_bbox)

    plane_model, inliers_plane = pcd_pc_bbox.segment_plane(distance_threshold=5,
                                                    ransac_n=10,
                                                    num_iterations=1000)

    #display_inlier_outlier(pcd_pc_bbox, inliers_plane)

    choose_bbox = np.delete(choose_bbox, inliers_plane)
    choose = np.intersect1d(choose_bbox, choose_mask)

    points_raw = depth2pc(depth, choose)
    pcd_pc_raw = o3d.geometry.PointCloud()
    pcd_pc_raw.points = o3d.utility.Vector3dVector(points_raw)

    #o3d.visualization.draw_geometries([pcd_pc_raw])

    cl, ind_1 = pcd_pc_raw.remove_statistical_outlier(nb_neighbors=80, std_ratio=2.5)
    pcd_pc_inler1 = pcd_pc_raw.select_by_index(ind_1)
    cl, ind_2 = pcd_pc_inler1.remove_statistical_outlier(nb_neighbors=2000, std_ratio=5.0)
    pcd_pc_inler2 = pcd_pc_inler1.select_by_index(ind_2)

    labels = np.array(pcd_pc_inler2.cluster_dbscan(eps=60, min_points=200))

    if(len(labels) == 0):
        return choose_mask

    max_label = labels.max()
    min_label = labels.min()

    biggest_cluster_idx = 0
    biggest_cluster_elem_count = 0
    if max_label >= 1 or min_label == -1:
        final_cluster_list = []
        for label_idx in range(max_label + 1):
            cluster_elem_count =  len(np.where(labels == label_idx)[0])
            if cluster_elem_count > biggest_cluster_elem_count:
                biggest_cluster_elem_count = len(np.where(labels == label_idx)[0])
                biggest_cluster_idx = label_idx
        
        final_cluster_list.append(biggest_cluster_idx)

        ind_biggest_cluster = np.where(labels == biggest_cluster_idx)[0]
        pcd_pc_biggest_cluster = pcd_pc_inler2.select_by_index(ind_biggest_cluster)
        pcd_pc_biggest_cluster_center = np.mean(np.array(pcd_pc_biggest_cluster.points), axis=0) 

        for label_idx in range(max_label + 1):
            if label_idx == biggest_cluster_idx:
                continue
            label_idx_ind = np.where(labels == label_idx)[0]
            pcd_pc_idx_cluster = pcd_pc_inler2.select_by_index(label_idx_ind)

            pcd_pc_idx_cluster_center = np.mean(np.array(pcd_pc_idx_cluster.points), axis=0) 

            #print(np.linalg.norm(pcd_pc_biggest_cluster_center - pcd_pc_idx_cluster_center))
            if np.linalg.norm(pcd_pc_biggest_cluster_center - pcd_pc_idx_cluster_center) < 200:
                final_cluster_list.append(label_idx)
        
        ind_3 = []
        for idx in final_cluster_list:
            idx_ind = list(np.where(labels == idx)[0])
            ind_3.extend(idx_ind)
        pcd_pc_inler = pcd_pc_inler2.select_by_index(ind_3)
    else:
        pcd_pc_inler = pcd_pc_inler2
        ind_3 = np.array(range(labels.shape[0]))

    #display_inlier_outlier_all(pcd_pc_raw, ind_1, ind_2, ind_3)
    choose_f1 = choose[ind_1]
    choose_del1 = np.delete(choose, ind_1)

    choose_f2 = choose_f1[ind_2]
    choose_del2 = np.delete(choose_f1, ind_2)

    choose_f3 = choose_f2[ind_3]
    choose_del3 = np.delete(choose_f2, ind_3)
    
    choose_final = choose_f3
    choose_deleted = list(set(choose_mask)-set(choose_final))

    return choose_deleted


def outliner_removal_camera(depth, mask_raw, gts, idx):
    instance_id = gts['instance_ids'][idx]
    rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx], depth.shape[0], depth.shape[1])

    mask_i = np.equal(mask_raw, instance_id)
    mask_depth = np.logical_and(mask_i, depth > 0)
    choose_mask = mask_depth.flatten().nonzero()[0]

    black_img = np.zeros_like(depth)    
    black_img[rmin:rmax, cmin:cmax] = 1.0
    choose_bbox = black_img.flatten().nonzero()[0]

    points_bbox = depth2pc(depth, choose_bbox)

    pcd_pc_bbox = o3d.geometry.PointCloud()
    pcd_pc_bbox.points = o3d.utility.Vector3dVector(points_bbox)

    plane_model, inliers_plane = pcd_pc_bbox.segment_plane(distance_threshold=5,
                                                    ransac_n=10,
                                                    num_iterations=1000)

    #display_inlier_outlier(pcd_pc_bbox, inliers_plane)

    choose_bbox = np.delete(choose_bbox, inliers_plane)
    choose = np.intersect1d(choose_bbox, choose_mask)

    points_raw = depth2pc(depth, choose)
    pcd_pc_raw = o3d.geometry.PointCloud()
    pcd_pc_raw.points = o3d.utility.Vector3dVector(points_raw)

    #o3d.visualization.draw_geometries([pcd_pc_raw])

    cl, ind_1 = pcd_pc_raw.remove_statistical_outlier(nb_neighbors=80, std_ratio=1.5)
    pcd_pc_inler1 = pcd_pc_raw.select_by_index(ind_1)
    cl, ind_2 = pcd_pc_inler1.remove_statistical_outlier(nb_neighbors=2000, std_ratio=5.0)
    pcd_pc_inler2 = pcd_pc_inler1.select_by_index(ind_2)

    labels = np.array(pcd_pc_inler2.cluster_dbscan(eps=60, min_points=200))

    if(len(labels) == 0):
        return choose_mask

    max_label = labels.max()
    min_label = labels.min()

    biggest_cluster_idx = 0
    biggest_cluster_elem_count = 0
    if max_label >= 1 or min_label == -1:
        final_cluster_list = []
        for label_idx in range(max_label + 1):
            cluster_elem_count =  len(np.where(labels == label_idx)[0])
            if cluster_elem_count > biggest_cluster_elem_count:
                biggest_cluster_elem_count = len(np.where(labels == label_idx)[0])
                biggest_cluster_idx = label_idx
        
        final_cluster_list.append(biggest_cluster_idx)

        ind_biggest_cluster = np.where(labels == biggest_cluster_idx)[0]
        pcd_pc_biggest_cluster = pcd_pc_inler2.select_by_index(ind_biggest_cluster)
        pcd_pc_biggest_cluster_center = np.mean(np.array(pcd_pc_biggest_cluster.points), axis=0) 

        for label_idx in range(max_label + 1):
            if label_idx == biggest_cluster_idx:
                continue
            label_idx_ind = np.where(labels == label_idx)[0]
            pcd_pc_idx_cluster = pcd_pc_inler2.select_by_index(label_idx_ind)

            pcd_pc_idx_cluster_center = np.mean(np.array(pcd_pc_idx_cluster.points), axis=0) 

            #print(np.linalg.norm(pcd_pc_biggest_cluster_center - pcd_pc_idx_cluster_center))
            if np.linalg.norm(pcd_pc_biggest_cluster_center - pcd_pc_idx_cluster_center) < 120:
                final_cluster_list.append(label_idx)
        
        ind_3 = []
        for idx in final_cluster_list:
            idx_ind = list(np.where(labels == idx)[0])
            ind_3.extend(idx_ind)
        pcd_pc_inler = pcd_pc_inler2.select_by_index(ind_3)
    else:
        pcd_pc_inler = pcd_pc_inler2
        ind_3 = np.array(range(labels.shape[0]))

    #display_inlier_outlier_all(pcd_pc_raw, ind_1, ind_2, ind_3)
    choose_f1 = choose[ind_1]
    choose_del1 = np.delete(choose, ind_1)

    choose_f2 = choose_f1[ind_2]
    choose_del2 = np.delete(choose_f1, ind_2)

    choose_f3 = choose_f2[ind_3]
    choose_del3 = np.delete(choose_f2, ind_3)
    
    choose_final = choose_f3
    choose_deleted = list(set(choose_mask)-set(choose_final))

    return choose_deleted



def outliner_removal(img_name):
    print("os.path.join(data_path, img_name)", os.path.join(data_path, img_name))
    depth = load_depth(os.path.join(data_path, img_name))
    plt.imshow(depth)
    plt.show()
    mask_raw = cv2.imread(os.path.join(data_path, img_name + '_mask.png'))[:, :, 2]
    gts = cPickle.load(open(os.path.join(data_path, img_name + '_label.pkl'), 'rb'))

    depth_flattened = depth.flatten()

    # print("gts", gts['poses'])
    index = 0
    for idx in range(len(gts['instance_ids'])):
        #print(gts['class_ids'][idx])
        if gts['class_ids'][idx] == CLASS_MAP_FOR_CATEGORY['laptop']:
            choose_deleted = outliner_removal_laptop(depth, mask_raw, gts, idx)
        elif gts['class_ids'][idx] == CLASS_MAP_FOR_CATEGORY['bottle']:
            choose_deleted = outliner_removal_bottle_can(depth, mask_raw, gts, idx)
        elif gts['class_ids'][idx] == CLASS_MAP_FOR_CATEGORY['bowl']:
            choose_deleted = outliner_removal_bowl_mug(depth, mask_raw, gts, idx)
        elif gts['class_ids'][idx] == CLASS_MAP_FOR_CATEGORY['camera']:
            choose_deleted = outliner_removal_camera(depth, mask_raw, gts, idx)
        elif gts['class_ids'][idx] == CLASS_MAP_FOR_CATEGORY['can']:
            choose_deleted = outliner_removal_bottle_can(depth, mask_raw, gts, idx)
        elif gts['class_ids'][idx] == CLASS_MAP_FOR_CATEGORY['mug']:
            choose_deleted = outliner_removal_bowl_mug(depth, mask_raw, gts, idx)

        depth_flattened[choose_deleted] = 0

    depth_final = depth_flattened.reshape((480, 640))

    plt.imshow(depth_final)
    plt.show()

    for idx in range(len(gts['instance_ids'])):
        instance_id = gts['instance_ids'][idx]
        pose = gts['poses'][idx]
        mask_i = np.equal(mask_raw, instance_id)
        mask_depth = np.logical_and(mask_i, depth > 0)
        choose = mask_depth.flatten().nonzero()[0]
        points_filtered = depth2pc(depth_final, choose, scaled=True)
        print("points_filtered", points_filtered)
        pc = transform_pcd_to_canonical(pose, points_filtered)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.copy(pc))
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([pcd, mesh])
        filename = 'masked_depth_pcd_' + str(idx)+'.pcd'
        o3d.io.write_point_cloud(filename, pcd)



    if not os.path.exists(os.path.join(result_path, img_name.split('/')[0], img_name.split('/')[1])):
        os.makedirs(os.path.join(result_path, img_name.split('/')[0], img_name.split('/')[1]))

    saved_path = os.path.join(result_path, img_name + '_depth.png')

    print(saved_path)

    numpngw.write_png(saved_path, depth_final)

if __name__ == "__main__":
    # data_list = open(data_list_file).readlines()
    # data_list = [item.strip('\n') for item in  data_list]
    
    data_list = ['test/scene_3/0000']

    for img_name in data_list:
        print(img_name)
        # if img_name.find('fliped') != -1:
        #     continue
        
        # image_name_array = img_name.split('/')
        # if image_name_array[1] != 'scene_7':
        #     continue
        # if int(image_name_array[-1]) <= 500:
        #     continue

        outliner_removal(img_name)
        #break
