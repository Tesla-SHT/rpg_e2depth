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
                 verbose=False,
                 clip_distance = 65535.0):
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
        self.clip_distance = clip_distance
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
    
    def __getitem__(self, idx, seed=None):
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
        Args:
            idx: 索引
            seed: 随机种子（用于数据增强的一致性）
        """
        # 如果提供了seed，设置随机种子以保证序列中的transform一致
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        actual_idx = self.start_idx + idx
        frame_name = f'frame{actual_idx:06d}'
        #assert actual_idx <=0, "Index out of range"
        #print("acutal idx", actual_idx)
        #print("fdslkfjskljflksjf")
        data = {
            'frame_idx': actual_idx
        }
        #self.use_voxel = False
        # 加载事件数据
        if self.use_voxel:
            # 加载voxel数据从.npz文件
            # print("idx", idx)
            # print("start idx", self.start_idx)
            # print("actual idx", actual_idx)
            # print("frame_name", frame_name)
            npz_path = join(self.images_folder, f'{frame_name}.npz')
            if actual_idx <=0:
                raise IndexError(f"Index out of range: {actual_idx}")
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
            png_path = join(self.images_folder, f'{frame_name}.jpg')
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
            
            events = torch.from_numpy(event_frame).float()
        
        data['events'] = events
        
        # 加载深度图
        if self.load_depth:
            depth_path = join(self.depth_folder, f'{frame_name}.jpg.geometric.png')
            if os.path.exists(depth_path):
                from PIL import Image
                depth_img = Image.open(depth_path)
                depth = np.array(depth_img).astype(np.float32)
                
                #print("depth range", depth.min(), depth.max())
                # 归一化处理（参考原版）
                depth = np.clip(depth, 0.0, self.clip_distance)
                max_val = np.amax(depth[~np.isnan(depth)])
                if max_val > 0:
                    depth = depth / max_val
                
                # # 转为 log depth
                # depth = 1.0 + np.log(depth + 1e-6) / 3.70378
                # depth = depth.clip(0, 1.0)
                
                # 确保是 [1, H, W] 格式
                if len(depth.shape) == 2:  # [H, W]
                    depth = np.expand_dims(depth, 0)  # [1, H, W]
                elif len(depth.shape) == 3 and depth.shape[2] == 1:  # [H, W, 1]
                    depth = np.moveaxis(depth, -1, 0)  # [1, H, W]
                
                data['depth'] = torch.from_numpy(depth).float()
        
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
                        # 创建一个临时的组合tensor用于transform
            to_transform = data['events']
            
            if 'depth' in data:
                # 将depth添加为额外通道
                #depth_channel = data['depth'].unsqueeze(0)
                to_transform = torch.cat([to_transform, data['depth']], dim=0)
                
                # 应用transform
                transformed = self.transform(to_transform, is_flow=False)
                
                # 分离回去
                data['events'] = transformed[:-1]
                data['depth'] = transformed[-1:]
            else:
                data['events'] = self.transform(to_transform, is_flow=False)
        
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

import random

class EvGGSSequenceDataset(Dataset):
    """
    Load sequences of time-synchronized {event tensors + depth} from EvGGS dataset.
    Similar to SequenceSynchronizedFramesEventsDataset but for EvGGS format.
    """
    
    def __init__(self, base_folder, scene_name, sequence_length=5,
                 transform=None, clip_distance=65535.0, normalize=True,
                 scale_factor=1.0, inverse=False, step_size=1,
                 use_voxel=True, start_idx=1, stop_idx=None):
        
        self.L = sequence_length
        self.transform = transform
        self.clip_distance = clip_distance
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.inverse = inverse
        self.step_size = step_size
        
        # Create base dataset
        self.dataset = EvGGSDataset(
            base_folder=base_folder,
            scene_name=scene_name,
            sequence='1',
            start_idx=start_idx,
            stop_idx=stop_idx,
            transform=transform,
            load_depth=True,
            use_voxel=use_voxel,
            clip_distance = clip_distance
        )
        
        # Calculate sequence length
        if self.L >= len(self.dataset):
            self.length = 0
        else:
            self.length = (len(self.dataset) - self.L) // self.step_size + 1
        
        print(f'EvGGSSequenceDataset: {scene_name}, sequences: {self.length}')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        """
        Returns a list containing synchronized events <-> depth pairs
        """
        assert(i >= 0)
        assert(i < self.length)
        
        # Generate random seed for consistent transforms
        seed = random.randint(0, 2**32)
        
        sequence = []
        
        for k in range(0, self.L):
            j = i * self.step_size + k
            if j >= len(self.dataset):
                break
            item = self.dataset.__getitem__(j, seed)
            #print("item keys:", item.keys())
            # 确保 depth 有正确的维度 [1, H, W]
            if 'depth' in item:
                depth = item['depth']
                if len(depth.shape) == 2:  # [H, W]
                    depth = depth.unsqueeze(0)  # [1, H, W]
            else:
                # 如果没有depth，创建一个零张量
                depth = torch.zeros(1, item['events'].shape[1], item['events'].shape[2])

            # 创建 flow: [2, H, W]
            flow = torch.zeros(2, depth.shape[1], depth.shape[2], dtype=torch.float32)
            transformed_item = {
                'events': item['events'],
                'frame': depth,  # 用 depth 作为 frame
                'flow': flow
            }
            
            # 如果有其他需要的数据也可以保留
            if 'camera_intrinsics' in item:
                transformed_item['camera_intrinsics'] = item['camera_intrinsics']
            if 'camera_pose' in item:
                transformed_item['camera_pose'] = item['camera_pose']
            
            sequence.append(transformed_item)
        
        # Apply downsampling if needed
        if self.scale_factor < 1.0:
            for data_items in sequence:
                for key, item in data_items.items():
                    if key not in ["times", "frame_idx"]:
                        item = item[None]
                        item = torch.nn.functional.interpolate(
                            item, scale_factor=self.scale_factor, 
                            mode='bilinear', align_corners=True
                        )
                        item = item[0]
                        data_items[key] = item
        
        return sequence


def load_evggs_splits_seq(base_folder, train_ratio=0.8):
    """
    Load and split EvGGS scenes into train/val sets
    
    Args:
        base_folder: Path to EvGGS dataset
        train_ratio: Ratio of scenes to use for training
    
    Returns:
        train_scenes, val_scenes: Lists of scene names
    """
    import glob
    
    # Find all scene folders
    scene_paths = glob.glob(os.path.join(base_folder, '*'))
    scenes = [os.path.basename(p) for p in scene_paths if os.path.isdir(p)]
    scenes.sort()
    
    # Split into train/val
    n_train = int(len(scenes) * train_ratio)
    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train:]
    
    return train_scenes, val_scenes