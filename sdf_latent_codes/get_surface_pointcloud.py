from sdf_latent_codes.deep_sdf_decoder import Decoder
from sdf_latent_codes.create_mesh import create_mesh
from sdf_latent_codes.grid import Grid3D
from sdf_latent_codes.oct_grid import OctGrid, get_cell_size, subdivide, get_grid_surface_hook
import json
import os
import open3d as o3d
import torch
import numpy as np
from torch.autograd import Variable

def get_surface_pointclouds(latent_vector, sdf_latent_code_dir):
    specs_filename = os.path.join(sdf_latent_code_dir, 'specs.json')
    model_params_subdir = "ModelParameters"
    checkpoint = "2000"

    specs = json.load(open(specs_filename))
    latent_size = 64
    decoder = Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(sdf_latent_code_dir, model_params_subdir, checkpoint + ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    latent_vector = torch.from_numpy(latent_vector).cuda()
    latent_vector = latent_vector.squeeze(-1)
    latent_vector = latent_vector.float()

    # Create 3D grid and form DeepSDF input
    grid_density = 60  # dummy grid density
    device = torch.device('cuda')
    grid_3d = Grid3D(grid_density, device)
    latent_repeat = latent_vector.expand(grid_3d.points.size(0), -1)
    inputs = torch.cat([latent_repeat, grid_3d.points], 1).to(latent_vector.device, latent_vector.dtype)
    # with torch.no_grad():
    pred_sdf_grid = decoder(inputs)
    pcd_dsdf = grid_3d.get_surface_points_given(pred_sdf_grid)
    return pcd_dsdf.detach().cpu().numpy()

def get_sdfnet(sdf_latent_code_dir):
    specs_filename = os.path.join(sdf_latent_code_dir, 'specs.json')
    model_params_subdir = "ModelParameters"
    checkpoint = "2000"
    specs = json.load(open(specs_filename))
    latent_size = 64
    decoder = Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(sdf_latent_code_dir, model_params_subdir, checkpoint + ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    return decoder

# # Function to get surface grid points
def get_grid_surface_octgrid(feat_sdf, lods, octgrid, sdfnet):
    with torch.no_grad():
        xyz_0 = np.array(octgrid.centers)[np.array(octgrid.level) == lods[0]]
        xyz = torch.from_numpy(xyz_0).to(feat_sdf.device)
        for lod in lods:
            inputs_sdfnet = torch.cat([feat_sdf.expand(xyz.shape[0], -1), xyz], 1).to(feat_sdf.device, feat_sdf.dtype)
            sdf = sdfnet(inputs_sdfnet)
            occ = sdf.abs() < get_cell_size(lod)
            xyz = subdivide(xyz[occ[:, 0]], level=lod)
    # points = xyz
    points = Variable(xyz.to(feat_sdf.device, feat_sdf.dtype), requires_grad=True)
    return points

def get_surface_pointclouds_octgrid_viz(latent_vector, lod, sdf_latent_code_dir):
    specs_filename = os.path.join(sdf_latent_code_dir, 'specs.json')
    model_params_subdir = "ModelParameters"
    checkpoint = "2000"

    specs = json.load(open(specs_filename))
    latent_size = 64
    decoder = Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(sdf_latent_code_dir, model_params_subdir, checkpoint + ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    latent_vector = torch.from_numpy(latent_vector).cuda()
    latent_vector = latent_vector.squeeze(-1)
    latent_vector = latent_vector.float()
    lods = list(range(2,lod))
    octgrid = OctGrid(subdiv=lods[0])

    grid_3d = get_grid_surface_octgrid(latent_vector, lods, octgrid, decoder)
    grid_3d = grid_3d.float()
    inputs_sdfnet = torch.cat([latent_vector.expand(grid_3d.shape[0], -1), grid_3d], 1).to(latent_vector.device)
    pred_sdf_grid = decoder(inputs_sdfnet)
    pred_pcd, nrm_pcd = octgrid.get_surface_points_given(pred_sdf_grid, grid_3d, threshold=get_cell_size(lods[-1]))
    return pred_pcd.detach().cpu().numpy(), nrm_pcd.detach().cpu().numpy()

def get_surface_pointclouds_octgrid(latent_vector, octgrid, lods, decoder):
    latent_vector = latent_vector.squeeze(-1)
    latent_vector = latent_vector.float()
    
    # Create 3D grid and form DeepSDF input
    grid_3d = get_grid_surface_hook(latent_vector, lods, octgrid, decoder)
    inputs_sdfnet = torch.cat([latent_vector.expand(grid_3d.shape[0], -1), grid_3d], 1).to(latent_vector.device)
    pred_sdf_grid = decoder(inputs_sdfnet)
    # pred_pcd = octgrid.get_surface_points_given(pred_sdf_grid, grid_3d, threshold=get_cell_size(lods[-1]))
    pred_pcd, pcd_norm = octgrid.get_surface_points_given_hook(pred_sdf_grid, grid_3d, threshold=get_cell_size(lods[-1]))
    del grid_3d
    return pred_pcd, pcd_norm


def get_surface_pointclouds_octgrid_sparse(latent_vector, data_dir, lods=[2,3,4]):
    specs_filename = os.path.join(data_dir, 'sdf_pretrained', 'specs.json')
    model_params_subdir = "ModelParameters"
    checkpoint = "2000"
    sdf_latent_code_dir = os.path.join(data_dir, 'sdf_pretrained')

    specs = json.load(open(specs_filename))
    latent_size = 64
    decoder = Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(sdf_latent_code_dir, model_params_subdir, checkpoint + ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    latent_vector = torch.from_numpy(latent_vector).cuda()
    latent_vector = latent_vector.squeeze(-1)
    latent_vector = latent_vector.float()
    # Create 3D grid and form DeepSDF input
    octgrid = OctGrid(subdiv=lods[0])
    grid_3d = get_grid_surface_octgrid(latent_vector, lods, octgrid, decoder)
    return grid_3d.cpu().numpy()

def get_grid_sdfnet():
    specs_filename = "/home/ubuntu/generalizable-object-representations/sdf_latent_codes/all_ws_no_reg_contrastive0.1/specs.json"
    model_params_subdir = "ModelParameters"
    checkpoint = "2000"
    sdf_latent_code_dir = "/home/ubuntu/generalizable-object-representations/sdf_latent_codes/all_ws_no_reg_contrastive0.1"
    specs = json.load(open(specs_filename))
    latent_size = 64
    decoder = Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(sdf_latent_code_dir, model_params_subdir, checkpoint + ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    grid_density = 40  # dummy grid density
    device = torch.device('cuda')
    grid_3d = Grid3D(grid_density, device)
    return decoder, grid_3d