import os
import sys
import math
import torch
import numpy
import argparse
from simnet.lib.net import common
from prepare_data.generate_training_data import annotate_camera_train, annotate_test_data, annotate_real_train
from pathlib import Path

seed = 1
numpy.random.seed(seed)
torch.manual_seed(seed)


def main(args):
    if args.end > args.all_frames:
        args.end = args.all_frames
    if args.type == 'camera_train':
        annotate_camera_train(args.data_dir, args.start, args.end)
    elif args.type == 'camera_val':
        annotate_test_data(args.data_dir, 'CAMERA', 'val', args.start, args.end)
    elif args.type == 'real_train':
        annotate_real_train(args.data_dir, args.start, args.end)
    else:
        annotate_test_data(args.data_dir, 'Real', 'test', args.start, args.end)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='', type=str, help='Is tested on validation data or not.'
    )
    parser.add_argument('--type', type=str, default='train', help='Is tested on validation data or not.'
    )
    parser.add_argument('--id', default=0, type=int, help='Is tested on validation data or not.'
    )
    parser.add_argument('--start', default=0, type=int, help='Is tested on validation data or not.'
    )
    parser.add_argument('--end', default=0, type=int, help='Is tested on validation data or not.'
    )
    parser.add_argument('--all_frames', default=0, type=int, help='Is tested on validation data or not.'
    )
    args = parser.parse_args()
    main(args)