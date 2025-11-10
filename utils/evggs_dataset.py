import os
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join
import json


class EvGGSDataset(Dataset):
    """
    Dataset for loading EvGGS data format.
    
    EvGGS数据集结构:
    base_folder/
    └── scene_name/
        └── 1/
            ├── images_rgb/          # RGB图像
            │   ├── frame000001.jpg
            │   └── ...
            ├── images/              # 事件帧和voxel
            │   ├── frame000001.png  # 事件可视化帧
            │   ├── frame000001.npz  # voxel数据
            │   └── ...
            ├── depths/              # 深度图
            │   ├── frame000001.jpg.geometric.png
            │   └── ...
            └── masks/               # 掩码
                ├── frame000001.png
                └── ...
        ├── selected_seqs_train.json
        ├── selected_seqs_val.json
        └── selected_seqs_test.json
    """
    
    def __init__(self, 
                 base_folder, 
                 scene_name,
                 sequence='1',
                 start_idx=1,
                 stop_idx=None,
                 transform=None,
                 load_depth=False,
                 load_rgb=False,
                 load_mask=False,
                 use_voxel=True,
                 verbose=False):
        """
        Args:
            base_folder: EvGGS数据集的根目录
            scene_name: 场景名称
            sequence: 序列编号（默认'1'）
            start_idx: 起始帧索引（从1开始）
            stop_idx: 结束帧索引（None表示到最后）
            transform: 数据变换
            load_depth: 是否加载深度图
            load_rgb: 是否加载RGB图像
            load_mask: 是否加载mask
            use_voxel: True使用voxel数据，False使用事件帧PNG
            verbose: 是否打印调试信息
        """
        self.base_folder = base_folder
        self.scene_name = scene_name
        self.sequence = sequence
        self.transform = transform
        self.load_depth = load_depth
        self.load_rgb = load_rgb
        self.load_mask = load_mask
        self.use_voxel = use_voxel
        self.verbose = verbose
        
        # 路径设置
        self.seq_folder = join(base_folder, scene_name, sequence)
        self.images_folder = join(self.seq_folder, 'images')
        self.rgb_folder = join(self.seq_folder, 'images_rgb')
        self.depth_folder = join(self.seq_folder, 'depths')
        self.mask_folder = join(self.seq_folder, 'masks')
        
        # 获取图像文件列表（使用.npz或.png）
        if use_voxel:
            event_files = sorted([f for f in os.listdir(self.images_folder) 
                                if f.endswith('.npz')])
        else:
            event_files = sorted([f for f in os.listdir(self.images_folder) 
                                if f.endswith('.png')])
        
        # 设置索引范围
        self.start_idx = start_idx
        if stop_idx is None:
            self.stop_idx = len(event_files) + start_idx
        else:
            self.stop_idx = stop_idx
        
        self.num_frames = self.stop_idx - self.start_idx
        
        if self.verbose:
            print(f"EvGGS Dataset initialized:")
            print(f"  Scene: {scene_name}, Sequence: {sequence}")
            print(f"  Frames: {self.start_idx} to {self.stop_idx-1} (total: {self.num_frames})")
            print(f"  Using {'voxel (.npz)' if use_voxel else 'event frame (.png)'}")
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        """
        Returns:
            data: dict包含以下内容:
                - 'events': 事件voxel或事件帧 [C, H, W]
                - 'frame_idx': 帧索引
                - 'depth': (可选) 深度图 [H, W]
                - 'rgb': (可选) RGB图像 [3, H, W]
                - 'mask': (可选) 掩码 [H, W]
                - 'camera_intrinsics': (可选) 相机内参
                - 'camera_pose': (可选) 相机位姿
                - 'maximum_depth': (可选) 最大深度值
        """
        actual_idx = self.start_idx + idx
        frame_name = f'frame{actual_idx:06d}'
        
        data = {
            'frame_idx': actual_idx
        }
        
        # 加载事件数据
        if self.use_voxel:
            # 加载voxel数据从.npz文件
            npz_path = join(self.images_folder, f'{frame_name}.npz')
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Voxel file not found: {npz_path}")
            
            npz_data = np.load(npz_path)
            
            # 调试信息（只在第一帧且verbose=True时打印）
            if idx == 0 and self.verbose:
                print(f"\n=== NPZ Debug Info ===")
                print(f"NPZ file keys: {list(npz_data.keys())}")
                if 'voxel' in npz_data:
                    print(f"Original voxel shape: {npz_data['voxel'].shape}")
                    print(f"Voxel dtype: {npz_data['voxel'].dtype}")
                    print(f"Voxel min/max: {npz_data['voxel'].min():.4f}/{npz_data['voxel'].max():.4f}")
            
            voxel = npz_data['voxel']
            
            # 检查voxel的维度顺序并转置（如果需要）
            # 假设: 通道数 < 空间维度
            if len(voxel.shape) == 3:
                if voxel.shape[2] < voxel.shape[0] and voxel.shape[2] < voxel.shape[1]:
                    # [H, W, C] -> [C, H, W]
                    voxel = voxel.transpose(2, 0, 1)
                    if idx == 0 and self.verbose:
                        print(f"Transposed from [H, W, C] to [C, H, W]: {voxel.shape}")
                elif idx == 0 and self.verbose:
                    print(f"Already in [C, H, W] format: {voxel.shape}")
            
            events = torch.from_numpy(voxel).float()
            
            if idx == 0 and self.verbose:
                print(f"Final events tensor shape: {events.shape}")
                print("======================\n")
            
            # 保存额外的元数据
            if 'camera_intrinsics' in npz_data:
                data['camera_intrinsics'] = torch.from_numpy(npz_data['camera_intrinsics']).float()
            if 'camera_pose' in npz_data:
                data['camera_pose'] = torch.from_numpy(npz_data['camera_pose']).float()
            if 'maximum_depth' in npz_data:
                data['maximum_depth'] = float(npz_data['maximum_depth'])
                
        else:
            # 加载事件帧PNG图像
            png_path = join(self.images_folder, f'{frame_name}.png')
            if not os.path.exists(png_path):
                raise FileNotFoundError(f"Event frame not found: {png_path}")
            
            from PIL import Image
            event_frame = Image.open(png_path)
            event_frame = np.array(event_frame)
            
            # 如果是RGB图像，转换为CHW格式
            if len(event_frame.shape) == 3:
                event_frame = event_frame.transpose(2, 0, 1)  # HWC -> CHW
            else:
                # 如果是灰度图，添加channel维度
                event_frame = event_frame[np.newaxis, :, :]
            
            events = torch.from_numpy(event_frame).float() / 255.0
        
        data['events'] = events
        
        # 加载深度图
        if self.load_depth:
            # 注意深度文件名格式: frame000001.jpg.geometric.png
            depth_path = join(self.depth_folder, f'{frame_name}.jpg.geometric.png')
            if os.path.exists(depth_path):
                from PIL import Image
                depth = Image.open(depth_path)
                depth = np.array(depth).astype(np.float32)
                data['depth'] = torch.from_numpy(depth)
        
        # 加载RGB图像
        if self.load_rgb:
            rgb_path = join(self.rgb_folder, f'{frame_name}.jpg')
            if os.path.exists(rgb_path):
                from PIL import Image
                rgb = Image.open(rgb_path)
                rgb = np.array(rgb).transpose(2, 0, 1)  # HWC -> CHW
                data['rgb'] = torch.from_numpy(rgb).float() / 255.0
        
        # 加载掩码
        if self.load_mask:
            mask_path = join(self.mask_folder, f'{frame_name}.png')
            if os.path.exists(mask_path):
                from PIL import Image
                mask = Image.open(mask_path)
                mask = np.array(mask)
                data['mask'] = torch.from_numpy(mask)
        
        # 应用变换
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def get_scene_info(self):
        """返回场景信息"""
        sample = self[0]
        
        if len(sample['events'].shape) == 3:
            num_channels, height, width = sample['events'].shape
        else:
            num_channels = 1
            height, width = sample['events'].shape
        
        info = {
            'scene_name': self.scene_name,
            'sequence': self.sequence,
            'num_frames': len(self),
            'height': height,
            'width': width,
            'num_channels': num_channels,
        }
        
        if 'camera_intrinsics' in sample:
            info['camera_intrinsics'] = sample['camera_intrinsics'].numpy()
        if 'maximum_depth' in sample:
            info['maximum_depth'] = sample['maximum_depth']
        
        return info


def load_evggs_splits(base_folder, scene_name, split='train'):
    """
    加载EvGGS数据集的train/val/test split
    
    Args:
        base_folder: 数据集根目录
        scene_name: 场景名称
        split: 'train', 'val', or 'test'
    
    Returns:
        list of sequence configurations
    """
    split_file = join(base_folder, scene_name, f'selected_seqs_{split}.json')
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        sequences = json.load(f)
    
    return sequences