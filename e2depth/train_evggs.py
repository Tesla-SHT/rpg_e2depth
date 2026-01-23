import os
import json
import logging
import argparse
import torch
os.environ['KORNIA_FEATURE_DISABLE_FLASH'] = '1'
from model.model import *
from model.loss import *
from model.metric import *
from torch.utils.data import DataLoader, ConcatDataset
from data_loader.dataset import *
from trainer.lstm_trainer import LSTMTrainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
from utils.evggs_dataset import EvGGSSequenceDataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='')


def concatenate_evggs_scenes(base_folder, scenes, sequence_length, transform=None,
                              clip_distance=65535.0, normalize=True, scale_factor=1.0, 
                              inverse=False, step_size=1, start_idx=1, stop_idx = 200):
    """
    Create an instance of ConcatDataset by aggregating all EvGGS scenes
    """
    train_datasets = []
    
    for scene_name in scenes:
        print(f'Loading scene: {scene_name}')
        try: 
            #print("basefolder", base_folder)
            dataset = EvGGSSequenceDataset(
                base_folder=base_folder,
                scene_name=scene_name,
                sequence_length=sequence_length,
                transform=transform,
                clip_distance=clip_distance,
                normalize=normalize,
                scale_factor=scale_factor,
                inverse=inverse,
                step_size=step_size,
                use_voxel=True,
                start_idx=start_idx,
                stop_idx = stop_idx
            )
            train_datasets.append(dataset)
            print(f'  Loaded {len(dataset)} sequences from {scene_name}')
        except Exception as e:
            print(f'  Warning: Failed to load scene {scene_name}: {e}')
            continue
    
    if len(train_datasets) == 0:
        raise ValueError("No datasets were loaded successfully!")
    
    concat_dataset = ConcatDataset(train_datasets)
    print(f'Total sequences: {len(concat_dataset)}')
    
    return concat_dataset


def main(config, resume, initial_checkpoint=None):
    train_logger = None

    L = config['trainer']['sequence_length']
    assert(L > 0)

    # Get EvGGS dataset paths
    base_folder = {}
    for split in ['train', 'validation']:
        base_folder[split] = config['data_loader'][split]['base_folder']
    
    # Load scene splits from your EvGGS dataset
    from utils.evggs_dataset import load_evggs_splits_seq
    train_scenes, val_scenes = load_evggs_splits_seq(base_folder['train'])
    
    print(f'\nTraining scenes: {train_scenes}')
    print(f'Validation scenes: {val_scenes}\n')

    clip_distance = config['data_loader']['train'].get('clip_distance', 100.0)
    normalize = config['data_loader'].get('normalize', True)
    inverse = config['data_loader']['train'].get('inverse', False)
    scale_factor = config['data_loader']['train'].get('scale_factor', 1.0)
    step_size = config['data_loader']['train'].get('step_size', 1)

    # Create training dataset
    train_dataset = concatenate_evggs_scenes(
        base_folder['train'],
        train_scenes,
        sequence_length=L,
        transform=Compose([
            RandomRotationFlip(0.0, 0.5, 0.0),
            RandomCrop(112)
        ]),
        clip_distance=clip_distance,
        normalize=normalize,
        scale_factor=scale_factor,
        inverse=inverse,
        step_size=step_size,
        start_idx = config['data_loader']['train'].get('start_idx', 1),
        stop_idx = config['data_loader']['train'].get('stop_idx', 200)
    )

    # Create validation dataset
    validation_dataset = concatenate_evggs_scenes(
        base_folder['validation'],
        val_scenes,
        sequence_length=L,
        transform=CenterCrop(112),
        clip_distance=clip_distance,
        normalize=normalize,
        scale_factor=scale_factor,
        inverse=inverse,
        step_size=step_size,
        start_idx = config['data_loader']['validation'].get('start_idx', 1),
        stop_idx = config['data_loader']['validation'].get('stop_idx', 200)
    )

    # Set up data loaders
    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    
    data_loader = DataLoader(
        train_dataset, 
        batch_size=config['data_loader']['batch_size'],
        shuffle=config['data_loader']['shuffle'], 
        **kwargs
    )
    print("\n=== Checking first batch ===")
    for batch in data_loader:
        print(f"Batch length: {len(batch)}")  # 应该是 sequence_length
        print(f"First item keys: {batch[0].keys()}")
        print(f"Events shape: {batch[0]['events'].shape}")
        print(f"Frame shape: {batch[0]['frame'].shape}")
        print(f"Events min/max: {batch[0]['events'].min():.4f}/{batch[0]['events'].max():.4f}")
        print(f"Frame min/max: {batch[0]['frame'].min():.4f}/{batch[0]['frame'].max():.4f}")
        break
    print("=========================\n")
    valid_data_loader = DataLoader(
        validation_dataset, 
        batch_size=config['data_loader']['batch_size'],
        shuffle=config['data_loader']['shuffle'], 
        **kwargs
    )

    # Initialize model
    model = eval(config['arch'])(config['model'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    model.summary()

    # Set up loss and metrics
    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print(f"Using {config['loss']['type']} with config {config['loss']['config']}")
    
    metrics = [eval(metric) for metric in config['metrics']]

    # Initialize trainer
    trainer = LSTMTrainer(
        model, loss, loss_params, metrics,
        resume=resume,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        train_logger=train_logger
    )

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Training E2DEPTH on EvGGS Dataset')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--initial_checkpoint', default=None, type=str,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning('Warning: --initial_checkpoint overridden by --resume')
        config = torch.load(args.resume)['config']
    
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
    
    assert config is not None

    main(config, args.resume, args.initial_checkpoint)