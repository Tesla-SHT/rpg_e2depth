import os
from os.path import join
import argparse
import numpy as np
import torch
import cv2

from utils.loading_utils import load_model, get_device
from utils.evggs_dataset import EvGGSDataset
from depth_prediction import DepthEstimator
from options.inference_options import set_depth_inference_options

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def invlog_to_depth(prediction, reg_factor=3.70378):
    """
    将网络输出的 log depth 转换为绝对深度
    预测值范围: [0, 1] (log space)
    输出范围: 真实深度值
    """
    pred = np.exp(reg_factor * (prediction - np.ones(prediction.shape, dtype=np.float32)))
    return pred

def prepare_gt_depth(gt_normalized, clip_distance=80.0):
    """
    处理从 dataset 加载的已归一化的 GT depth
    
    Args:
        gt_normalized: 从 EvGGSDataset 加载的深度图，已归一化到 [0, 1]
        clip_distance: 用于还原真实深度的最大距离
    
    Returns:
        绝对深度值 (米)
    """
    # Dataset 已经做了归一化，直接乘以 clip_distance 还原
    gt_abs = gt_normalized * clip_distance
    
    # 处理无效值
    gt_abs = np.nan_to_num(gt_abs, nan=0.0, posinf=0.0, neginf=0.0)
    
    return gt_abs

def compute_metrics(gt, pred, eps=1e-6):
    """
    计算深度估计指标
    
    Args:
        gt: ground truth 绝对深度 [H, W]
        pred: predicted 绝对深度 [H, W]
    """
    mask = (gt > eps) & (pred > eps) & np.isfinite(gt) & np.isfinite(pred)
    
    if np.sum(mask) == 0:
        return {"abs_rel": np.nan, "delta1.25": np.nan, "delta1.03": np.nan, "n": 0}
    
    gt_m = gt[mask]
    pred_m = pred[mask]
    
    # Absolute relative error
    abs_rel = np.mean(np.abs(pred_m - gt_m) / (gt_m + eps))
    
    # Threshold accuracy
    ratio = np.maximum(gt_m / (pred_m + eps), pred_m / (gt_m + eps))
    delta1_25 = np.mean(ratio <= 1.25)
    delta1_03 = np.mean(ratio <= 1.03)
    
    return {
        "abs_rel": float(abs_rel), 
        "delta1.25": float(delta1_25), 
        "delta1.03": float(delta1_03), 
        "n": int(np.sum(mask))
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run depth on EvGGS and evaluate against GT")
    parser.add_argument('-c', '--path_to_model', required=True, type=str)
    parser.add_argument('-i', '--input_folder', required=True, type=str)
    parser.add_argument('--scene_name', required=True, type=str)
    parser.add_argument('--start_idx', default=1, type=int)
    parser.add_argument('--stop_idx', default=None, type=int)
    parser.add_argument('--eval_clip_distance', default=80.0, type=float,
                        help='Maximum depth value for evaluation (meters)')
    parser.add_argument('--reg_factor', default=3.70378, type=float)
    parser.add_argument('--save_pred_png', action='store_true')
    parser.add_argument('--save_gt_png', action='store_true')
    parser.add_argument('--save_colormap', action='store_true',
                        help='Save depth as colormap visualization')
    
    set_depth_inference_options(parser)
    args = parser.parse_args()

    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model = model.to(device)
    model.eval()

    # Create dummy dataset to get dimensions
    dummy_ds = EvGGSDataset(
        base_folder=args.input_folder,
        scene_name=args.scene_name,
        sequence='1',
        start_idx=args.start_idx if args.start_idx > 0 else 1,
        stop_idx=args.stop_idx,
        transform=None,
        load_depth=True,
        use_voxel=True,
        verbose=True
    )
    
    sample = dummy_ds[0]
    num_bins, height, width = sample['events'].shape
    height = height - args.low_border_crop

    # Create estimator
    estimator = DepthEstimator(model, height, width, num_bins, args)

    # Create actual dataset
    events_dataset = EvGGSDataset(
        base_folder=args.input_folder,
        scene_name=args.scene_name,
        sequence='1',
        start_idx=args.start_idx if args.start_idx > 0 else 1,
        stop_idx=args.stop_idx,
        transform=None,
        load_depth=True,
        use_voxel=True,
        verbose=False
    )

    N = len(events_dataset)
    print(f"\nProcessing scene: {args.scene_name}")
    print(f"Total frames: {N}")
    print(f"Clip distance: {args.eval_clip_distance}m\n")

    # Setup output folders
    if args.output_folder is None:
        raise ValueError("Please provide --output_folder")

    dataset_name = args.scene_name
    pred_folder = join(args.output_folder, dataset_name, 'predictions')
    gt_folder = join(args.output_folder, dataset_name, 'ground_truth')
    
    safe_mkdir(pred_folder)
    safe_mkdir(gt_folder)
    
    if args.save_pred_png:
        pred_png_folder = join(args.output_folder, dataset_name, 'pred_png')
        safe_mkdir(pred_png_folder)
    if args.save_gt_png:
        gt_png_folder = join(args.output_folder, dataset_name, 'gt_png')
        safe_mkdir(gt_png_folder)
    if args.save_colormap:
        pred_color_folder = join(args.output_folder, dataset_name, 'pred_colormap')
        gt_color_folder = join(args.output_folder, dataset_name, 'gt_colormap')
        safe_mkdir(pred_color_folder)
        safe_mkdir(gt_color_folder)

    # Metrics accumulation
    sum_abs_rel = 0.0
    sum_delta1_25 = 0.0
    sum_delta1_03 = 0.0
    counted_frames = 0

    for idx in range(N):
        if idx % 50 == 0:
            print(f"Processing: {idx}/{N}")

        item = events_dataset[idx]
        
        if 'depth' not in item:
            print(f"Frame {idx}: no GT depth, skipping")
            continue

        # Run inference
        event_tensor = item['events'][:, :height, :].unsqueeze(0)
        
        with torch.no_grad():
            events = event_tensor.to(estimator.device)
            if args.use_fp16:
                events = events.half()
            
            events = estimator.event_preprocessor(events)
            events_padded = estimator.crop.pad(events)
            
            pred_tensor, states = estimator.model(
                events_padded, 
                estimator.last_states_for_each_channel.get('grayscale', None)
            )
            
            if estimator.no_recurrent:
                estimator.last_states_for_each_channel['grayscale'] = None
            else:
                estimator.last_states_for_each_channel['grayscale'] = states
            
            crop = estimator.crop
            pred_logdepth = pred_tensor[0, 0, crop.iy0:crop.iy1, crop.ix0:crop.ix1].cpu().numpy()

        # Save prediction (log depth)
        np.save(join(pred_folder, f'depth_{idx:010d}.npy'), pred_logdepth.astype(np.float32))

        # Get GT depth (already normalized to [0,1] by dataset)
        gt_normalized = item['depth'].numpy().astype(np.float32)
        if gt_normalized.shape[0] == 1:  # Remove channel dimension if present
            gt_normalized = gt_normalized[0]
        if gt_normalized.shape[0] >= height:
            gt_normalized = gt_normalized[:height, :]

        # Save GT (normalized)
        np.save(join(gt_folder, f'depth_{idx:010d}.npy'), gt_normalized)

        # Convert to absolute depth for evaluation
        gt_abs = prepare_gt_depth(gt_normalized, args.eval_clip_distance)
        pred_abs = invlog_to_depth(pred_logdepth, args.reg_factor) * args.eval_clip_distance

        # Compute metrics
        metrics = compute_metrics(gt_abs, pred_abs)
        if metrics["n"] > 0:
            sum_abs_rel += metrics["abs_rel"]
            sum_delta1_25 += metrics["delta1.25"]
            sum_delta1_03 += metrics["delta1.03"]
            counted_frames += 1

        # Save visualization PNGs (grayscale)
        if args.save_pred_png:
            # Normalize to [0, 255] for visualization
            pred_vis = np.clip(pred_abs / args.eval_clip_distance, 0.0, 1.0)
            cv2.imwrite(
                join(pred_png_folder, f'pred_{idx:010d}.png'),
                (pred_vis * 255.0).astype(np.uint8)
            )
        
        if args.save_gt_png:
            # GT 已经在 [0, 1] 范围
            gt_vis = np.clip(gt_normalized, 0.0, 1.0)
            cv2.imwrite(
                join(gt_png_folder, f'gt_{idx:010d}.png'),
                (gt_vis * 255.0).astype(np.uint8)
            )
        
        # Save colormap visualization
        if args.save_colormap:
            pred_vis = np.clip(pred_abs / args.eval_clip_distance, 0.0, 1.0)
            pred_color = cv2.applyColorMap((pred_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(join(pred_color_folder, f'pred_{idx:010d}.png'), pred_color)
            
            gt_vis = np.clip(gt_normalized, 0.0, 1.0)
            gt_color = cv2.applyColorMap((gt_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(join(gt_color_folder, f'gt_{idx:010d}.png'), gt_color)

    # Print results
    print("\n" + "="*50)
    if counted_frames == 0:
        print("No valid frames with ground truth were processed.")
    else:
        avg_abs_rel = sum_abs_rel / counted_frames
        avg_delta1_25 = sum_delta1_25 / counted_frames
        avg_delta1_03 = sum_delta1_03 / counted_frames

        print(f"Evaluation results ({counted_frames} frames):")
        print(f"  abs_rel:      {avg_abs_rel:.6f}")
        print(f"  delta <= 1.25: {avg_delta1_25:.6f}")
        print(f"  delta <= 1.03: {avg_delta1_03:.6f}")
    print("="*50)