import argparse
import json
import numpy as np
import os
import torch


from simnet.lib.net.models.deep_sdf_decoder import Decoder

import deep_sdf
import deep_sdf.workspace as ws

def code_to_mesh(experiment_directory, checkpoint, keep_normalized=False):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    print(len(instance_filenames), " vs ", len(latent_vectors))

    for i, latent_vector in enumerate(latent_vectors):

        dataset_name, class_name, instance_name = instance_filenames[i].split("/")
        instance_name = instance_name.split(".")[0]

        print("{} {} {}".format(dataset_name, class_name, instance_name))

        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )
        print(mesh_dir)

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)

        print(instance_filenames[i])

        offset = None
        scale = None

        if not keep_normalized:

            normalization_params = np.load(
                ws.get_normalization_params_filename(
                    data_source, dataset_name, class_name, instance_name
                )
            )
            offset = normalization_params["offset"]
            scale = normalization_params["scale"]

        print("latent vector", latent_vector.device, latent_vector.shape)
        latent_vector = latent_vector.cuda()
        with torch.no_grad():
            deep_sdf.mesh.create_mesh(
                decoder,
                latent_vector,
                mesh_filename,
                N=256,
                max_batch=int(2 ** 18),
                offset=offset,
                scale=scale,
            )