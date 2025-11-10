import torch
from utils.loading_utils import load_model, get_device
from utils.event_readers import VoxelGridDataset
from utils.evggs_dataset import EvGGSDataset, load_evggs_splits  # 新增
from os.path import join, basename
import numpy as np
import json
import argparse
import shutil
import os
from depth_prediction import DepthEstimator
from options.inference_options import set_depth_inference_options

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_folder', default=None, type=str,
                        help="name of the folder containing the voxel grids")
    parser.add_argument('--start_time', default=0.0, type=float)
    parser.add_argument('--stop_time', default=0.0, type=float)
    
    # 新增EvGGS相关参数
    parser.add_argument('--dataset_type', default='voxelgrid', type=str,
                        choices=['voxelgrid', 'evggs'],
                        help='Type of dataset: voxelgrid or evggs')
    parser.add_argument('--scene_name', default=None, type=str,
                        help='Scene name for EvGGS dataset')
    parser.add_argument('--start_idx', default=0, type=int,
                        help='Start frame index for EvGGS dataset')
    parser.add_argument('--stop_idx', default=None, type=int,
                        help='Stop frame index for EvGGS dataset')

    set_depth_inference_options(parser)

    args = parser.parse_args()

    print_every_n = 50

    # Load model to device
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model = model.to(device)
    model.eval()

    # 根据数据集类型创建dataset
    if args.dataset_type == 'evggs':
        if args.scene_name is None:
            raise ValueError("--scene_name must be specified for EvGGS dataset")
        
        print(f"\n=== Loading EvGGS Dataset ===")
        
        
        # 创建EvGGS dataset，使用voxel数据
        dummy_dataset = EvGGSDataset(
            base_folder=args.input_folder,
            scene_name=args.scene_name,
            sequence='1',
            start_idx=args.start_idx if args.start_idx > 0 else 1,
            stop_idx=args.stop_idx,
            transform=None,
            load_depth=True,
            use_voxel=True,
            verbose=True  # 打印第一帧的调试信息
        )
        
        data = dummy_dataset[0]
        print(f"\nFirst frame data keys: {data.keys()}")
        print(f"Events shape: {data['events'].shape}")
        
        # events shape应该是 [C, H, W]
        if len(data['events'].shape) == 3:
            num_bins, height, width = data['events'].shape
        else:
            raise ValueError(f"Unexpected events shape: {data['events'].shape}")
        
        height = height - args.low_border_crop
        
        print(f"Model expects {model.num_bins} bins, data has {num_bins} bins")
        
        if model.num_bins != num_bins:
            print(f"WARNING: Model bins ({model.num_bins}) != Data bins ({num_bins})")
        
        estimator = DepthEstimator(model, height, width, model.num_bins, args)
        
        # 创建实际使用的dataset（不打印调试信息）
        events_dataset = EvGGSDataset(
            base_folder=args.input_folder,
            scene_name=args.scene_name,
            sequence='1',
            start_idx=args.start_idx if args.start_idx > 0 else 1,
            stop_idx=args.stop_idx,
            transform=None,
            load_depth=True,
            use_voxel=True,
            verbose=False  # 不打印调试信息
        )
        
        dataset_name = args.scene_name if args.dataset_name is None else args.dataset_name
        
    else:  # voxelgrid
        base_folder = os.path.dirname(args.input_folder)
        event_folder = os.path.basename(args.input_folder)

        # hack to get the image size: create a dummy dataset,
        # grab the first data item and read the required info
        dummy_dataset = VoxelGridDataset(base_folder,
                                         event_folder,
                                         args.start_time,
                                         args.stop_time,
                                         transform=None)
        data = dummy_dataset[0]
        _, height, width = data['events'].shape
        height = height - args.low_border_crop
        
        estimator = DepthEstimator(model, height, width, model.num_bins, args)

        events_dataset = VoxelGridDataset(base_folder=base_folder,
                                   event_folder = event_folder,
                                   start_time=args.start_time,
                                   stop_time=args.stop_time,
                                   transform=None)
        
        dataset_name = args.dataset_name

    output_dir = args.output_folder
    print('Processing {}'.format(dataset_name), end=': ')

    N = len(events_dataset)

    if output_dir is not None:
        os.makedirs(join(output_dir, dataset_name), exist_ok=True)
        
        # 只有voxelgrid数据集才复制这些文件
        if args.dataset_type == 'voxelgrid':
            shutil.copyfile(join(args.input_folder, 'timestamps.txt'),
                            join(output_dir, dataset_name, 'timestamps.txt'))
            shutil.copyfile(join(args.input_folder, 'boundary_timestamps.txt'),
                            join(output_dir, dataset_name, 'boundary_timestamps.txt'))

    idx = 0
    while idx < N:
        if idx % print_every_n == 0:
            print('{} / {}'.format(idx, N))

        data = events_dataset[idx]
        event_tensor = data['events'][:,:height,:]

        estimator.update_reconstruction(event_tensor, idx)
        idx += 1