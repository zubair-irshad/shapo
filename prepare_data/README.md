## Preparing NOCS dataset<br>

This Readme should be used if you are preparing [NOCS](https://github.com/hughw19/NOCS_CVPR2019) dataset in the format requried by CenterSnap training and inference or if you would like to train CenterSnap with your custom dataset on a custom camera model.

Note: The data generation script is currently not optimized for speed i.e. it might take a long time to generate the complete data. Please refer to [Datasets](https://github.com/zubair-irshad/CenterSnap/#-dataset) to download preprocessed dataset. Please manually use parallel python scripts to process the data faster using the below script. 


Download [camera_train](http://download.cs.stanford.edu/orion/nocs/camera_train.zip), [camera_val](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip),
[real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground-truth annotations](http://download.cs.stanford.edu/orion/nocs/gts.zip),
[camera_composed_depth](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip), [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip) and [eval_results](https://drive.google.com/file/d/1p72NdY4Bie_sra9U8zoUNI4fTrQZdbnc/view?usp=sharing)provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019) and [nocs preprocess data](https://www.dropbox.com/s/8im9fzopo71h6yw/nocs_preprocess.tar.gz?dl=1).<br/>
Also download [sdf_rgb_pretrained_weights](https://www.dropbox.com/s/929kz7zuxw8jajy/sdf_rgb_pretrained.tar.xz?dl=1). 
Unzip and organize these files in $ShAPO_Repo/data as follows:
```
data
├── CAMERA
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── camera_full_depths
│   ├── train
│   └── val
├── gts
│   ├── val
│   └── real_test
├── auto_encoder_model
│   ├── model_50_nocs.pth
├── results
│   ├── camera
│   ├── mrcnn_results
│   ├── nocs_results
│   └── real
├── sdf_rgb_pretrained
│   ├── LatentCodes
│   ├── Reconstructions
│   ├── ModelParameters
│   ├── OptimizerParameters
│   └── rgb_latent
└── obj_models
    ├── train
    ├── val
    ├── real_train
    ├── real_test
    ├── camera_train.pkl
    ├── camera_val.pkl
    ├── real_train.pkl
    ├── real_test.pkl
    └── mug_meta.pkl

```
Run distributed script to collect data locally in aorund an hour

--type chose from 'camera_train', 'camera_val', 'real_train', 'real_val'
```
python prepare_data/distributed_generate_data.py --data_dir /home/ubuntu/shapoplusplus/data/nocs_data --type camera_train
```

Run python scripts to prepare the datasets.

```
cd $CenterSnap repo
./runner.sh prepare_data/generate_data_nocs.py --data_dir absolute_path\to\downloaded\NOCS\data
```

**Acknowledgement**<br>
The data processing script is adapted from [object-deformnet](https://github.com/mentian/object-deformnet)

