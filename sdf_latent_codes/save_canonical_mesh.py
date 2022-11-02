from sdf_latent_codes.deep_sdf_decoder import Decoder
from sdf_latent_codes.create_mesh import create_mesh
import json
import os

import torch


def save_mesh(latent_vector, mesh_filename):
    specs_filename = "/home/zubair/generalizable-object-representations/sdf_latent_codes/all_ws_no_reg_contrastive0.1/specs.json"
    model_params_subdir = "ModelParameters"
    checkpoint = "2000"
    sdf_latent_code_dir = "/home/zubair/generalizable-object-representations/sdf_latent_codes/all_ws_no_reg_contrastive0.1"

    specs = json.load(open(specs_filename))
    latent_size = 64
    decoder = Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(sdf_latent_code_dir, model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()

    offset = None
    scale = None

    latent_vector = torch.from_numpy(latent_vector).cuda()
    latent_vector = latent_vector.squeeze(-1)
    latent_vector = latent_vector.float()

    with torch.no_grad():
        create_mesh(
            decoder,
            latent_vector,
            mesh_filename,
            N=256,
            max_batch=int(2 ** 18),
            offset=offset,
            scale=scale,
        )