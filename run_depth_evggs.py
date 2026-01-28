import os
from os.path import join
import argparse
import numpy as np
import torch
import cv2
import csv

from utils.loading_utils import load_model, get_device
from utils.evggs_dataset import EvGGSDataset
from depth_prediction import DepthEstimator
from options.inference_options import set_depth_inference_options

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def apply_mask_to_depth(depth, mask):
    """
    将mask应用到depth图上
    mask为False的区域深度设为0
    
    Args:
        depth: numpy array [H, W]
        mask: torch.Tensor or numpy array [H, W], True表示有效区域
    """
    if mask is None:
        return depth
    
    # Convert mask to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    # 转换为布尔类型
    mask_bool = mask > 0
    
    # 确保mask和depth尺寸一致
    if mask_bool.shape != depth.shape:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), 
                               (depth.shape[1], depth.shape[0]), 
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    
    masked_depth = depth.copy()
    masked_depth[~mask_bool] = 0.0
    return masked_depth
def depth_to_jet_colormap(depth, out_path, cmap='jet_r'):
    #change the depth=0 to depth 1
    depth[depth==0]=1.0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(depth, cmap=cmap)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150)
    plt.close()


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
    gt_abs = gt_normalized
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
        return {
            "abs_rel": np.nan, "sq_rel": np.nan, "rmse": np.nan, "rmse_log": np.nan, "si_log": np.nan,
            "delta1.25": np.nan, "delta1.25^2": np.nan, "delta1.25^3": np.nan, "n": 0
        }
    
    gt_m = gt[mask]
    pred_m = pred[mask]
    
    # Absolute relative error
    abs_rel = np.mean(np.abs(pred_m - gt_m) / (gt_m + eps))

    # Squared relative error
    sq_rel = np.mean(((pred_m - gt_m) ** 2) / (gt_m + eps))

    # RMSE
    rmse = np.sqrt(np.mean((pred_m - gt_m) ** 2))

    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(pred_m + eps) - np.log(gt_m + eps)) ** 2))

    # Scale-invariant log error
    log_diff = np.log(pred_m + eps) - np.log(gt_m + eps)
    si_log = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))

    # Threshold accuracy
    ratio = np.maximum(gt_m / (pred_m + eps), pred_m / (gt_m + eps))
    delta1_25 = np.mean(ratio <= 1.25)
    delta1_25_2 = np.mean(ratio <= 1.25 ** 2)
    delta1_25_3 = np.mean(ratio <= 1.25 ** 3)

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "si_log": float(si_log),
        "delta1.25": float(delta1_25),
        "delta1.25^2": float(delta1_25_2),
        "delta1.25^3": float(delta1_25_3),
        "delta1.03": float(np.mean(ratio <= 1.03)),
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
    parser.add_argument('--use_mask', action='store_true',
                        help='Load and apply masks from dataset')
    
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
        load_mask=args.use_mask,
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
        load_mask=args.use_mask,
        use_voxel=True,
        verbose=False
    )

    N = len(events_dataset)
    print(f"\nProcessing scene: {args.scene_name}")
    print(f"Total frames: {N}")
    print(f"Clip distance: {args.eval_clip_distance}m")
    if args.use_mask:
        print(f"Using masks from dataset")
    print()

    # Setup output folders
    if args.output_folder is None:
        raise ValueError("Please provide --output_folder")

    dataset_name = args.scene_name
    
    if args.save_pred_png:
        pred_png_folder = join(args.output_folder, dataset_name, 'pred_png')
        safe_mkdir(pred_png_folder)
    if args.save_gt_png:
        gt_png_folder = join(args.output_folder, dataset_name, 'gt_png')
        safe_mkdir(gt_png_folder)

    # Metrics accumulation
    sum_abs_rel = 0.0
    sum_delta1_25 = 0.0
    sum_delta1_03 = 0.0
    counted_frames = 0
    
    # Per-frame metrics storage
    per_frame_metrics = []
    
    # Best frames tracking
    best_abs_rel = {"frame": -1, "value": float('inf')}
    best_delta1_25 = {"frame": -1, "value": -1.0}
    best_delta1_03 = {"frame": -1, "value": -1.0}

    for idx in range(N):
        if idx % 50 == 0:
            print(f"Processing: {idx}/{N}")

        item = events_dataset[idx]
        
        if 'depth' not in item:
            print(f"Frame {idx}: no GT depth, skipping")
            continue

        # Get mask from dataset if available
        mask = item.get('mask', None)
        if ('mask' not in item):

            print("mask is None:", mask is None)
        print("mask range", np.min(mask.numpy()) if mask is not None else 'N/A', 
              np.max(mask.numpy()) if mask is not None else 'N/A')
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

        # Get GT depth (already normalized to [0,1] by dataset)
        gt_normalized = item['depth'].numpy().astype(np.float32)
        if gt_normalized.shape[0] == 1:  # Remove channel dimension if present
            gt_normalized = gt_normalized[0]
        if gt_normalized.shape[0] >= height:
            gt_normalized = gt_normalized[:height, :]

        # Convert to absolute depth for evaluation
        gt_abs = prepare_gt_depth(gt_normalized, args.eval_clip_distance)
        pred_abs = invlog_to_depth(pred_logdepth, args.reg_factor)
        gt_abs = (gt_abs - np.min(gt_abs[gt_abs>0])) / (np.max(gt_abs[gt_abs>0]) - np.min(gt_abs[gt_abs>0]) + 1e-6)
        print("gt_abs range:", np.min(gt_abs[gt_abs>0]), np.max(gt_abs))
        print("pred_abs range:", np.min(pred_abs[pred_abs>0]), np.max(pred_abs))
        pred_abs = apply_mask_to_depth(pred_abs, mask)
        gt_abs = apply_mask_to_depth(gt_abs, mask)
        # Compute metrics (metrics函数会自动忽略depth=0的区域)
        metrics = compute_metrics(gt_abs, pred_abs)
        if metrics["n"] > 0:
            sum_abs_rel += metrics["abs_rel"]
            sum_delta1_25 += metrics["delta1.25"]
            sum_delta1_03 += metrics["delta1.03"]
            counted_frames += 1
            
            # Store per-frame metrics
            per_frame_metrics.append({
                "frame": idx,
                "abs_rel": metrics["abs_rel"],
                "delta1.25": metrics["delta1.25"],
                "delta1.03": metrics["delta1.03"],
                "valid_pixels": metrics["n"]
            })
            
            # Track best frames
            if metrics["abs_rel"] < best_abs_rel["value"]:
                best_abs_rel = {"frame": idx, "value": metrics["abs_rel"]}
            if metrics["delta1.25"] > best_delta1_25["value"]:
                best_delta1_25 = {"frame": idx, "value": metrics["delta1.25"]}
            if metrics["delta1.03"] > best_delta1_03["value"]:
                best_delta1_03 = {"frame": idx, "value": metrics["delta1.03"]}

        # Save visualization with JET colormap
        if args.save_pred_png:
            pred_color = depth_to_jet_colormap(pred_abs, join(pred_png_folder, f'pred_{idx:06d}.png'),"jet_r")
            # cv2.imwrite(
            #     join(pred_png_folder, f'pred_{idx:010d}.png'),
            #     pred_color
            # )
        
        if args.save_gt_png:
            # GT 已经在 [0, 1] 范围
            #gt_vis = np.clip(gt_normalized, 0.0, 1.0)
            gt_color = depth_to_jet_colormap(gt_abs, join(gt_png_folder, f'gt_{idx:06d}.png'),"jet_r")
            # cv2.imwrite(
            #     join(gt_png_folder, f'gt_{idx:010d}.png'),
            #     gt_color
            # )

    # Save per-frame metrics to CSV
    csv_path = join(args.output_folder, dataset_name, 'metrics_per_frame.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'abs_rel', 'delta1.25', 'delta1.03', 'valid_pixels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_frame_metrics:
            writer.writerow(row)
    
    # Print results
    print("\n" + "="*50)
    if counted_frames == 0:
        print("No valid frames with ground truth were processed.")
    else:
        avg_abs_rel = sum_abs_rel / counted_frames
        avg_delta1_25 = sum_delta1_25 / counted_frames
        avg_delta1_03 = sum_delta1_03 / counted_frames

        print(f"Evaluation results ({counted_frames} frames):")
        print(f"  abs_rel:       {avg_abs_rel:.6f}")
        print(f"  delta <= 1.25: {avg_delta1_25:.6f}")
        print(f"  delta <= 1.03: {avg_delta1_03:.6f}")
        
        print("\n" + "-"*50)
        print("Best frames:")
        print(f"  Best abs_rel:       Frame {best_abs_rel['frame']:04d} = {best_abs_rel['value']:.6f}")
        print(f"  Best delta <= 1.25: Frame {best_delta1_25['frame']:04d} = {best_delta1_25['value']:.6f}")
        print(f"  Best delta <= 1.03: Frame {best_delta1_03['frame']:04d} = {best_delta1_03['value']:.6f}")
        
        # Save summary to text file
        summary_path = join(args.output_folder, dataset_name, 'metrics_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Scene: {args.scene_name}\n")
            f.write(f"Model: {args.path_to_model}\n")
            f.write(f"Frames evaluated: {counted_frames}\n")
            f.write(f"Clip distance: {args.eval_clip_distance}m\n")
            f.write(f"Reg factor: {args.reg_factor}\n")
            if args.use_mask:
                f.write(f"Using masks: Yes\n")
            f.write("\n")
            
            f.write("Average Metrics:\n")
            f.write(f"  abs_rel:       {avg_abs_rel:.6f}\n")
            f.write(f"  delta <= 1.25: {avg_delta1_25:.6f}\n")
            f.write(f"  delta <= 1.03: {avg_delta1_03:.6f}\n\n")
            
            f.write("Best Frames:\n")
            f.write(f"  Best abs_rel:       Frame {best_abs_rel['frame']:04d} = {best_abs_rel['value']:.6f}\n")
            f.write(f"  Best delta <= 1.25: Frame {best_delta1_25['frame']:04d} = {best_delta1_25['value']:.6f}\n")
            f.write(f"  Best delta <= 1.03: Frame {best_delta1_03['frame']:04d} = {best_delta1_03['value']:.6f}\n")
        
        print(f"\nMetrics saved to:")
        print(f"  CSV:     {csv_path}")
        print(f"  Summary: {summary_path}")
    
    print("="*50)
'''
python run_depth_evggs.py -c "saved/e2depth_evggs-debug-smooth-v2/checkpoint-epoch082-loss-0.0412.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor" -o "./output/Tartanair_custom" --scene_name hospital_easy_P019 --save_pred_png --save_gt_png --use_mask --use_gpu --start_idx 350 --stop_idx 400

python run_depth_evggs.py -c "saved/e2depth_evggs-debug-smooth-v2/checkpoint-epoch082-loss-0.0412.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor" -o "./output/Tartanair_custom" --scene_name hospital_easy_P028 --save_pred_png --save_gt_png --use_mask --use_gpu --start_idx 180 --stop_idx 210

python run_depth_evggs.py -c "saved/e2depth_evggs-debug-smooth-v2/checkpoint-epoch082-loss-0.0412.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor" -o "./output/Tartanair_custom" --scene_name hospital_easy_P015 --save_pred_png --save_gt_png --use_mask --use_gpu --start_idx 150 --stop_idx 210

python run_depth_evggs.py -c "saved/e2depth_evggs-debug-smooth-v2/checkpoint-epoch082-loss-0.0412.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor" -o "./output/Tartanair_custom" --scene_name office2_easy_P011 --save_pred_png --save_gt_png --use_mask --use_gpu --start_idx 5 --stop_idx 60

python run_depth_evggs.py -c "pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/MVSEC_all" -o "./output/MVSEC_custom" --scene_name indoor_flying_3 --save_pred_png --save_gt_png --use_mask --use_gpu --start_idx 1200 --stop_idx 1300

python run_depth_evggs.py -c "pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/MVSEC_all" -o "./output/MVSEC_custom" --scene_name outdoor_day_1 --save_pred_png --save_gt_png --use_mask --use_gpu --start_idx 3300 --stop_idx 3400
'''