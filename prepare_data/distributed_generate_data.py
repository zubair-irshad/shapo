import os
import sys
import argparse
import math
from subprocess import Popen, PIPE
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--type',type=str, default='train')
    args = parser.parse_args()

    worker_per_gpu = 10
    num_gpus = 6
    # workers = torch.cuda.device_count() * worker_per_gpu

    workers = num_gpus * worker_per_gpu

    print("torch.cuda.device_count()", torch.cuda.device_count())
    
    if args.type=='camera_train':
        list_all = open(os.path.join(args.data_dir, 'CAMERA', 'train_list_all.txt')).read().splitlines()
    elif args.type=='camera_val':
        list_all = open(os.path.join(args.data_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()
        list_all = list_all[:1000]
    elif args.type=='real_train':
        list_all = open(os.path.join(args.data_dir, 'Real', 'train_list_all.txt')).read().splitlines()
    else:
        list_all = open(os.path.join(args.data_dir, 'Real', 'test_list_all.txt')).read().splitlines()
        
    all_frames = range(0,len(list_all))
    frames_per_worker = math.ceil(len(all_frames) / workers)
    for i in range(workers):
        curr_gpu = i // worker_per_gpu

        start = i * frames_per_worker
        end = start + frames_per_worker


        print("start, : end", start, end)

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(curr_gpu+2)

        print(i, curr_gpu+2)
        print(all_frames[start:end])

        command = [
            './runner.sh', 'prepare_data/distributed_worker.py', 
            '--data_dir', str(args.data_dir), 
            '--type', str(args.type),
            '--id', str(i), 
            '--start', str(start), 
            '--end', str(end), 
            '--all_frames', str(len(list_all))
            
        ]
        log = open('worker_{}.txt'.format(i), 'w')
        print(command)
        Popen(command, env=my_env, stderr=log, stdout=log)
  
if __name__ == '__main__':
    main()