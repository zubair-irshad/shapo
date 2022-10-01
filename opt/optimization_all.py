import torch
import numpy as np
import open3d as o3d
from torch.autograd import Variable
from utils.viz_utils import plot_optim_3d
from utils.transform_utils import transform_pcd_tensor, get_abs_pose_from_RTs, transform_pcd_to_canonical
from sklearn.neighbors import KDTree
from sdf_latent_codes.oct_grid import OctGrid
from sdf_latent_codes.get_surface_pointcloud import get_surface_pointclouds_octgrid
from sdf_latent_codes.get_rgb import get_rgb_from_rgbnet

class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
            # scaler.step(op)

def get_opt_params(params, rgbnet, device, mask_area):
    # Transform to torch
    for key, value in params.items():
        params[key] = Variable(torch.Tensor(value).to(device, torch.float32), requires_grad=True)
    # Generate optimization parameter list
    
    # if mask_area>250:
    #     optim_params = []
    #     optim_params += [{'params': params['RT'], 'lr': 0.001}]
    #     optim_params += [{'params': params['scale'], 'lr': 0.0001}]
    #     optim_params += [{'params': params['latent'], 'lr': 0.0003}]

    # else:
    #     optim_params = []
    #     optim_params += [{'params': params['RT'], 'lr': 0.003}]
    #     optim_params += [{'params': params['scale'], 'lr': 0.0001}]
    #     optim_params += [{'params': params['latent'], 'lr': 0.0003}]

    optim_params = []
    optim_params += [{'params': params['RT'], 'lr': 0.0001}]
    optim_params += [{'params': params['scale'], 'lr': 0.0001}]
    optim_params += [{"params": params['appearance'], "lr": 0.01}]
    # optim_params += [{"params": rgbnet.parameters(), "lr": 0.00001}]
    optim_params += [{"params": rgbnet.parameters(), "lr": 0.0001}]
    # optim_params += [{'params': params['latent'], 'lr': 0.0003}]
    optim_params += [{'params': params['latent'], 'lr': 0.00008}]
    
    return params, optim_params


class Optimizer:
    def __init__(self, params, rgbnet, device, weights, mask_area, rot='dcm'):
        self.params, self.optim_params = get_opt_params(params, rgbnet, device, mask_area)
        # self.optim_params_sgd = self.optim_params[:]
        self.optim_params_adam = self.optim_params[0:4]
        self.optim_params_sgd = [self.optim_params[4]]
        self.solver = MultipleOptimizer(
            # torch.optim.Adam(self.optim_params_adam, lr=0.03),
            # torch.optim.SGD(self.optim_params_sgd, lr=0.01, momentum=0.0)
            torch.optim.Adam(self.optim_params_adam),
            torch.optim.SGD(self.optim_params_sgd)
        )
        self.weights = weights
        self.rot = rot

    # def optimize(self, iters_optim, nocs_pred, pcd_frustum_np, dsdf, grid, K, crop_size, viz_type=None, frame_vis=None):
    def optimize_oct_grid(self, iters_optim, pcd_frustum_np, pcd_gt_rgb_np, decoder, rgbnet, optim_foldername, viz_type=None, frame_vis=None):
        """
        Optimization loop
        Args:
            iters_optim (int): Number of iterations
            nocs_pred (torch.Tensor): CSS network prediction
            pcd_frustum_np (np.array): LIDAR point cloud (N,3)
            dsdf: DeepSDF network
            grid: Grid object
            K (torch.Tensor): Camera matrix (3,3)
            crop_size (list): Size of the optimized crop
            viz_type ('2d'/'3d'): Type of the visualization
            frame_vis: Image of the full frame
        """
        self.device = torch.device('cuda')
        lods = [2,3,4,5]
        oct_grid = OctGrid(subdiv=lods[0])
        # self.device = grid.device
        # self.precision = grid.dtype
        # self.renderer = Rasterer(K, crop_size[::-1], precision=K.dtype).to(K.device)
        # Create an open3D visualizer instance if needed
        if viz_type == '3d':
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=640, height=480)

        for e in range(iters_optim):
            # self.solver.zero_grad()
            # Apply inverse scale to scene to not screw up renderer
            pcd_frustum = torch.Tensor(pcd_frustum_np).to(self.device)
            pcd_gt_rgb = torch.Tensor(pcd_gt_rgb_np).to(self.device)
            
            abs_pose = get_abs_pose_from_RTs(self.params['RT'], self.params['scale'], self.device)
            pose_vis = abs_pose.clone().detach().cpu().numpy()
            
            
            # inputs_sdfnet = torch.cat([self.params['latent'].squeeze(-1).float().expand(grid_3d.shape[0], -1), grid_3d], 1).to(self.device)
            # pred_sdf_grid = dsdf(inputs_sdfnet)
            pcd_dsdf, pcd_nrm = get_surface_pointclouds_octgrid(self.params['latent'], oct_grid, lods, decoder)
            pred_rgb = get_rgb_from_rgbnet(self.params['latent'], pcd_dsdf, self.params['appearance'], rgbnet)

            self.solver.zero_grad()
            # pcd_dsdf_trans = transform_pcd_tensor(abs_pose, pcd_dsdf, self.device)
            #canonicalize masked_rgb_pointclouds
            pcd_frustum = transform_pcd_to_canonical(abs_pose, pcd_frustum)
            pcd_dsdf_trans = pcd_dsdf 
            if (pcd_dsdf_trans.nelement() == 0) or (pcd_frustum.nelement() == 0):
                print('Skip frame')
                continue
            # 3D Loss
            loss, dists_3d, idxs_3d, loss_3d, loss_rgb = self.compute_loss_3d(pcd_dsdf_trans, pcd_frustum, pred_rgb, pcd_gt_rgb)
            weight_3d = self.weights['3d']
            loss = weight_3d * loss 
            # Check for nans
            if torch.isnan(loss).sum() > 0 or loss.sum() == 0:
                print('Skip frame')
                continue
            loss.backward()
            self.solver.step()
            print('ITER {} | Total - {}, RGB - {}, 3D - {}'.format(e, weight_3d * loss.item(), loss_rgb.item(), loss_3d.item()))
            if viz_type == '3d':
                plot_optim_3d(vis, pcd_frustum, pcd_gt_rgb, pcd_dsdf_trans, pred_rgb, pcd_nrm, pose_vis, optim_foldername, e)
    # def optimize(self, iters_optim, nocs_pred, pcd_frustum_np, dsdf, grid, K, crop_size, viz_type=None, frame_vis=None):

    def compute_loss_3d(self, pcd_dsdf_trans, pcd_frustum, pred_rgb,  pcd_gt_rgb, threshold=0.2):
        """
        Compute 3D loss between the estimated and LIDAR point clouds computed as a mean of point pair distances between 2 point clouds
        Args:
            pcd_dsdf_trans (torch.Tensor): Estimated point cloud (N,3)
            pcd_frustum (torch.Tensor): LIDAR point cloud (N,3)
            threshold (float): maximum allowed distance between the point pairs to be considered for the loss
        Returns: 3D loss value
        """
        if (pcd_dsdf_trans.nelement() != 0) and (pcd_frustum.nelement() != 0):

            # Estimate nearest neighbors (distances and ids) between given point clouds
            kdtree = KDTree(pcd_frustum.detach().cpu().numpy())
            dists, idxs = kdtree.query(pcd_dsdf_trans.detach().cpu().numpy())

            # Reformat distances and idxs to 1-dim array
            idxs = np.asarray([val for sublist in idxs for val in sublist])
            dists = np.asarray([val for sublist in dists for val in sublist])

            # Measure distances between filtered point pairs
            close_by = dists < threshold 
            dists_thres = (pcd_frustum[idxs[close_by]] - pcd_dsdf_trans[close_by]).norm(p=2, dim=1)

            dists_rgb = (pcd_gt_rgb[idxs[close_by]] - pred_rgb[close_by]).norm(p=2, dim=1)
            # (pcd_gt_rgb - pred_rgb[idxs]).norm(p=2, dim=1)
            # Check if point pairs exist
            if (dists_thres.nelement() != 0):
                loss_3d = dists_thres.mean()
                loss_rgb = dists_rgb.mean()
                loss = loss_3d + loss_rgb
            else:
                loss = torch.tensor(0).to(self.device)
                loss_3d = torch.tensor(0).to(self.device)
                loss_rgb = torch.tensor(0).to(self.device)
        else:
            loss = torch.tensor(0).to(self.device)
            loss_3d = torch.tensor(0).to(self.device)
            loss_rgb = torch.tensor(0).to(self.device)

        return loss, dists, idxs, loss_3d, loss_rgb

    def compute_loss_2d(self, rendering_nocs, css_nocs, diam=5, threshold_nocs=1):
        """
        Compute 2D loss between the CSS net output and the rendering
        Args:
            rendering_nocs (torch.Tensor): NOCS image from the renderer
            css_nocs (torch.Tensor): NOCS image prediction from the CSS network
            diam (int): pixel distance defining the CSS NOCS masks diameter
            threshold_nocs (float): maximum allowed NOCS distance to be considered for the loss
        Returns: 2D loss value
        """
        # Get idxs of nonzero elements
        rendering_nonzero_idxs = rendering_nocs.sum(0).nonzero()
        if rendering_nonzero_idxs.sum():

            # Construct a 2D meshgrid dimensionally equal to the rendering
            xx, yy = torch.meshgrid(torch.arange(rendering_nocs.shape[1]),
                                    torch.arange(rendering_nocs.shape[2]))
            grid_2d = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), -1).float().to(self.device, self.precision)
            grid_2d_ext = grid_2d.unsqueeze(0).expand(rendering_nonzero_idxs.shape[0], *grid_2d.size())
            grid_2d_ext = grid_2d_ext.reshape(rendering_nonzero_idxs.shape[0], -1, 2)

            # For each non-zero rendering point define a circular area on css_nocs based on a given diameter
            vectors_to_point = grid_2d_ext - rendering_nonzero_idxs.view(-1, 1, 2)
            dist_to_point = torch.clamp(diam - vectors_to_point.pow(2).sum(-1).sqrt(), min=0)
            dist_to_point = dist_to_point.reshape(dist_to_point.shape[0], *grid_2d[:, :, 0].size())
            css_nocs_masked = css_nocs.unsqueeze(0) * dist_to_point.unsqueeze(1)

            # Compute minimum distances between rendering points and css_nocs masked points
            rendering_nonzeros = rendering_nocs[:, rendering_nocs.sum(0).nonzero(as_tuple=True)[0],
                             rendering_nocs.sum(0).nonzero(as_tuple=True)[1]].transpose(1, 0)
            diff = (css_nocs_masked - rendering_nonzeros.unsqueeze(-1).unsqueeze(-1)).pow(2).sum(1).sqrt()
            diff_min = diff.view(diff.shape[0], -1).min(1)[0]
            loss_2d = diff_min[diff_min < threshold_nocs].mean()
        else:
            loss_2d = torch.tensor(0).to(self.device, self.precision)
        return loss_2d