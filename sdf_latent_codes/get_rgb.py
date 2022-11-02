from siren_pytorch import SirenNet
import torch.nn as nn
import torch 
import os 

def get_rgb(latent_vector, pcd_dsdf, appearance_emb):
    # Define training device
    device = 'cuda'
    if device not in ['cpu', 'cuda']:
        raise ValueError('Unknown device.')
    # Prepare data
    latent_size = 64

    # Load model
    #output_path = io.read_cfg_string(cfg, 'evaluation', 'output_path', default='log/demo')
    rgbnet = SirenNet(
                dim_in=3+64+64,  # input dimension, ex. 2d coor
                dim_hidden=512,  # hidden dimension
                dim_out=3,  # output dimension, ex. rgb value
                num_layers=6,  # number of layers
                final_activation=nn.ReLU(),  # activation of final layer (nn.Identity() for direct output)
                w0_initial=128.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
            ).to(device)
    rgbnet.eval()

    # Recover model and features
    # model_path = io.read_cfg_string(cfg, 'evaluation', 'output_path', default=None)
    model_path = '/home/zubair/generalizable-object-representations/apearance_optimization/log/zubair_full_512'
    if model_path:
        rgbnet_dict = torch.load(os.path.join(model_path, 'reconstructor.pt'))
        model_dict = rgbnet_dict['model']
        rgbnet.load_state_dict(model_dict)

    latent_vector = torch.from_numpy(latent_vector).cuda()
    latent_vector = latent_vector.squeeze(-1)
    latent_vector = latent_vector.float()

    appearance_emb = torch.from_numpy(appearance_emb).cuda()
    appearance_emb = appearance_emb.squeeze(-1)
    appearance_emb = appearance_emb.float()

    pcd_dsdf = torch.from_numpy(pcd_dsdf).cuda()
    pcd_dsdf = pcd_dsdf.float()

    input_rgbnet = torch.cat([pcd_dsdf, latent_vector.expand(pcd_dsdf.shape[0], -1),
                                appearance_emb.expand(pcd_dsdf.shape[0], -1)], dim=-1)
    # Get output
    pred_rgb = rgbnet(input_rgbnet)
    return pred_rgb.detach().cpu().numpy()

def get_rgbnet(model_path):
    # Define training device
    device = 'cuda'
    if device not in ['cpu', 'cuda']:
        raise ValueError('Unknown device.')
    # Load model
    #output_path = io.read_cfg_string(cfg, 'evaluation', 'output_path', default='log/demo')
    rgbnet = SirenNet(
                dim_in=3+64+64,  # input dimension, ex. 2d coor
                dim_hidden=512,  # hidden dimension
                dim_out=3,  # output dimension, ex. rgb value
                num_layers=6,  # number of layers
                final_activation=nn.ReLU(),  # activation of final layer (nn.Identity() for direct output)
                w0_initial=128.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
            ).to(device)
    rgbnet.train()

    # Recover model and features
    # model_path = io.read_cfg_string(cfg, 'evaluation', 'output_path', default=None)
    if model_path:
        rgbnet_dict = torch.load(os.path.join(model_path, 'reconstructor.pt'))
        model_dict = rgbnet_dict['model']
        rgbnet.load_state_dict(model_dict)
    return rgbnet

def get_rgb_from_rgbnet(latent_vector, pcd_dsdf, appearance_emb, rgbnet):

    if not torch.is_tensor(latent_vector):
        latent_vector = torch.from_numpy(latent_vector).cuda()

    latent_vector = latent_vector.squeeze(-1)
    latent_vector = latent_vector.float()

    if not torch.is_tensor(appearance_emb):
        appearance_emb = torch.from_numpy(appearance_emb).cuda()

    # appearance_emb = torch.from_numpy(appearance_emb).cuda()
    appearance_emb = appearance_emb.squeeze(-1)
    appearance_emb = appearance_emb.float()

    if not torch.is_tensor(pcd_dsdf):
        pcd_dsdf = torch.from_numpy(pcd_dsdf).cuda()
    # pcd_dsdf = torch.from_numpy(pcd_dsdf).cuda()
    pcd_dsdf = pcd_dsdf.float()

    input_rgbnet = torch.cat([pcd_dsdf, latent_vector.expand(pcd_dsdf.shape[0], -1),
                                appearance_emb.expand(pcd_dsdf.shape[0], -1)], dim=-1)
    # Get output
    if not torch.is_tensor(pcd_dsdf):
        with torch.no_grad():
            pred_rgb = rgbnet(input_rgbnet)
    else:
        pred_rgb = rgbnet(input_rgbnet)
    return pred_rgb