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
from utils.pointcloud_metrics import (
    _as_intrinsics_matrix,
    depth_to_corresponding_pointclouds,
    depth_to_corresponding_pointclouds_with_pixels,
    depth_to_pointcloud,
    align_scale_to_reference,
    accuracy_completion,
    chamfer_distance,
    depth_to_normal_map,
    fscore,
    nn_query_1nn,
    normal_consistency_from_normals,
    pointwise_l2_stats,
)

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

def decode_depth_to_unit_range(depth_raw: np.ndarray) -> np.ndarray:
    """Decode GT depth image to [0,1] range.

    Many EvGGS/TartanAir depth PNGs are stored as uint16 in [0,65535] or
    uint8 in [0,255]. This function normalizes them to float32 [0,1].
    If the input already looks like [0,1], it is returned as-is.
    """
    d = np.asarray(depth_raw)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if d.dtype == np.uint16:
        return (d.astype(np.float32) / 65535.0).clip(0.0, 1.0)
    if d.dtype == np.uint8:
        return (d.astype(np.float32) / 255.0).clip(0.0, 1.0)

    d = d.astype(np.float32)
    dmax = float(np.max(d)) if d.size else 0.0
    if dmax > 255.0:
        return (d / 65535.0).clip(0.0, 1.0)
    if dmax > 1.5:
        return (d / 255.0).clip(0.0, 1.0)
    return d.clip(0.0, 1.0)


def _irls_l1_scale_shift(pred: np.ndarray, gt: np.ndarray, iters: int = 10, eps: float = 1e-6):
    """Robust scale+shift fit using IRLS to approximate L1 (LAD).

    Minimizes sum |s*pred + t - gt| approximately via iteratively reweighted
    least squares. This is a cheap per-frame alternative to scipy-based LAD.
    """
    pred = pred.astype(np.float64, copy=False)
    gt = gt.astype(np.float64, copy=False)

    med_p = float(np.median(pred))
    med_g = float(np.median(gt))
    s = med_g / (med_p + 1e-12)
    t = 0.0

    for _ in range(int(iters)):
        res = s * pred + t - gt
        w = 1.0 / (np.abs(res) + eps)

        a11 = float(np.sum(w * pred * pred))
        a12 = float(np.sum(w * pred))
        a22 = float(np.sum(w))
        b1 = float(np.sum(w * pred * gt))
        b2 = float(np.sum(w * gt))

        det = a11 * a22 - a12 * a12
        if not np.isfinite(det) or abs(det) < 1e-12:
            break

        s_new = (b1 * a22 - b2 * a12) / det
        t_new = (a11 * b2 - a12 * b1) / det

        if not (np.isfinite(s_new) and np.isfinite(t_new)):
            break
        s, t = float(s_new), float(t_new)

    if s < 0:
        s = 0.0
    return s, t


def align_depth(pred_abs: np.ndarray, gt_abs: np.ndarray, valid_mask: np.ndarray, mode: str):
    """Align pred depth to gt depth (single-frame) using scale/shift.

    mode:
      - 'none': no alignment
      - 'lstsq': least squares solve for s,t: s*pred + t ~= gt
      - 'median_scale': s = median(gt)/median(pred), t=0
            - 'lad': robust (L1) scale/shift via IRLS (LAD-style)
    """
    mode = (mode or 'none').lower()
    if mode == 'none':
        return pred_abs, 1.0, 0.0

    pred = pred_abs[valid_mask].reshape(-1).astype(np.float64)
    gt = gt_abs[valid_mask].reshape(-1).astype(np.float64)
    if pred.size == 0:
        return pred_abs, 1.0, 0.0

    if mode == 'median_scale':
        med_p = np.median(pred)
        med_g = np.median(gt)
        s = float(med_g / (med_p + 1e-12))
        t = 0.0
    elif mode == 'lad':
        s, t = _irls_l1_scale_shift(pred, gt)
    elif mode == 'lstsq':
        A = np.stack([pred, np.ones_like(pred)], axis=1)
        x, *_ = np.linalg.lstsq(A, gt, rcond=None)
        s = float(x[0])
        t = float(x[1])
    else:
        raise ValueError(f"Unknown depth alignment mode: {mode}")

    pred_aligned = (s * pred_abs + t).astype(np.float32)
    pred_aligned = np.nan_to_num(pred_aligned, nan=0.0, posinf=0.0, neginf=0.0)
    pred_aligned[pred_aligned < 0] = 0.0
    return pred_aligned, s, t

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
    delta1_03 = np.mean(ratio <= 1.03)
    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "si_log": float(si_log),
        "delta1.25": float(delta1_25),
        "delta1.25^2": float(delta1_25_2),
        "delta1.25^3": float(delta1_25_3),
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
    parser.add_argument('--use_mask', action='store_true',
                        help='Load and apply masks from dataset')

    # Depth alignment (optional, matches depth/tools.py spirit)
    parser.add_argument('--depth_align', default='none', type=str,
                        choices=['none', 'lstsq', 'median_scale', 'lad'],
                        help='Align predicted depth to GT before metrics (scale/shift)')

    # Point cloud evaluation (optional)
    parser.add_argument('--eval_pointcloud', action='store_true',
                        help='Evaluate point cloud metrics derived from depth')
    parser.add_argument('--pc_max_points', type=int, default=20000,
                        help='Max points for NN-based metrics (subsample)')
    parser.add_argument('--pc_fscore_tau', type=float, default=0.10,
                        help='F-score distance threshold (meters)')
    parser.add_argument('--pc_align_scale', action='store_true',
                        help='Align predicted point cloud scale to GT (recommended when scale is ambiguous)')
    parser.add_argument('--pc_eval_normals', action='store_true',
                        help='Also compute normal consistency metrics (NC1/NC2) derived from depth normals')
    parser.add_argument('--fx', type=float, default=None)
    parser.add_argument('--fy', type=float, default=None)
    parser.add_argument('--cx', type=float, default=None)
    parser.add_argument('--cy', type=float, default=None)
    
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
    print(f"Depth align: {args.depth_align}")
    if args.use_mask:
        print(f"Using masks from dataset")
    print()

    # Setup output folders
    if args.output_folder is None:
        raise ValueError("Please provide --output_folder")

    dataset_name = args.scene_name
    scene_out_dir = join(args.output_folder, dataset_name)
    safe_mkdir(scene_out_dir)

    if args.save_pred_png:
        pred_png_folder = join(scene_out_dir, 'pred_png')
        safe_mkdir(pred_png_folder)
    if args.save_gt_png:
        gt_png_folder = join(scene_out_dir, 'gt_png')
        safe_mkdir(gt_png_folder)

    # Metrics accumulation
    sum_abs_rel = 0.0
    sum_delta1_25 = 0.0
    sum_delta1_03 = 0.0
    counted_frames = 0

    # Point cloud metrics accumulation
    sum_pc_l2_mean = 0.0
    sum_pc_l2_median = 0.0
    sum_pc_chamfer = 0.0
    sum_pc_fscore = 0.0
    sum_pc_acc = 0.0
    sum_pc_comp = 0.0
    sum_pc_acc_median = 0.0
    sum_pc_comp_median = 0.0
    sum_pc_nc1 = 0.0
    sum_pc_nc2 = 0.0
    sum_pc_nc1_median = 0.0
    sum_pc_nc2_median = 0.0
    counted_pc_frames = 0
    
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
        #print("mask range", np.min(mask.numpy()) if mask is not None else 'N/A', 
        #      np.max(mask.numpy()) if mask is not None else 'N/A')
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

        # Get GT depth (often stored as uint8/uint16 PNG)
        gt_raw = item['depth']
        if isinstance(gt_raw, torch.Tensor):
            gt_raw = gt_raw.detach().cpu().numpy()
        else:
            gt_raw = np.asarray(gt_raw)

        # Remove channel dim if present (both CHW and HWC cases)
        if gt_raw.ndim == 3 and gt_raw.shape[0] == 1:
            gt_raw = gt_raw[0]
        elif gt_raw.ndim == 3 and gt_raw.shape[-1] == 1:
            gt_raw = gt_raw[..., 0]

        if gt_raw.ndim != 2:
            raise ValueError(f"Unexpected GT depth shape: {gt_raw.shape}")

        if gt_raw.shape[0] >= height:
            gt_raw = gt_raw[:height, :]

        # Convert to absolute depth for evaluation
        gt_unit = decode_depth_to_unit_range(gt_raw)
        if gt_unit.shape[0] >= height:
            gt_unit = gt_unit[:height, :]
        pred_norm = invlog_to_depth(pred_logdepth, args.reg_factor)

        # EvGGS depths are often normalized to [0, 1]. Recover metric scale.
        # Prefer per-frame maximum_depth from npz; fallback to eval_clip_distance.
        depth_scale = item.get('maximum_depth', None)
        try:
            if isinstance(depth_scale, torch.Tensor):
                depth_scale = float(depth_scale.item())
            elif depth_scale is not None:
                depth_scale = float(depth_scale)
        except Exception:
            depth_scale = None
        if depth_scale is None or not np.isfinite(depth_scale) or depth_scale <= 0:
            depth_scale = float(args.eval_clip_distance)

        # Heuristic: if maximum_depth looks like millimeters, convert to meters
        if depth_scale > 1e3:
            depth_scale = depth_scale / 1000.0

        gt_abs = gt_unit.astype(np.float32) * float(depth_scale)
        pred_abs = pred_norm * depth_scale

        # Apply dataset mask before alignment/metrics
        pred_abs = apply_mask_to_depth(pred_abs, mask)
        gt_abs = apply_mask_to_depth(gt_abs, mask)

        # Optional depth alignment (on valid pixels)
        eps = 1e-6
        align_mask = (gt_abs > eps) & (pred_abs > eps) & np.isfinite(gt_abs) & np.isfinite(pred_abs)
        if mask is not None:
            try:
                align_mask &= (mask.numpy() > 0)
            except Exception:
                align_mask &= (np.asarray(mask) > 0)
        pred_abs, depth_s, depth_t = align_depth(pred_abs, gt_abs, align_mask, args.depth_align)

        # Debug ranges in meters (after mask + optional alignment)
        if idx == 0 or idx % 50 == 0:
            if np.any(gt_abs > 0):
                print("gt_abs(m) range:", float(np.min(gt_abs[gt_abs > 0])), float(np.max(gt_abs)))
            if np.any(pred_abs > 0):
                print("pred_abs(m) range:", float(np.min(pred_abs[pred_abs > 0])), float(np.max(pred_abs)))
            if args.depth_align != 'none':
                try:
                    print(f"depth_align({args.depth_align}) s={float(depth_s):.6f}, t={float(depth_t):.6f}")
                except Exception:
                    print(f"depth_align({args.depth_align}) applied")
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

            # Optional: point cloud metrics
            if args.eval_pointcloud:
                # Build intrinsics (prefer dataset-provided)
                K_item = item.get('camera_intrinsics', None)
                if K_item is not None:
                    try:
                        K_item = K_item.detach().cpu().numpy()
                    except Exception:
                        pass

                try:
                    K = _as_intrinsics_matrix(K_item, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
                except Exception as e:
                    raise ValueError(
                        "Point cloud evaluation needs camera intrinsics. "
                        "Provide them in the dataset npz (camera_intrinsics) or via --fx/--fy/--cx/--cy. "
                        f"Original error: {e}"
                    )

                eps = 1e-6
                pc_mask = (gt_abs > eps) & (pred_abs > eps) & np.isfinite(gt_abs) & np.isfinite(pred_abs)
                if mask is not None:
                    try:
                        pc_mask &= (mask.numpy() > 0)
                    except Exception:
                        pc_mask &= (np.asarray(mask) > 0)

                if args.pc_eval_normals:
                    pts_gt, pts_pr, ys, xs = depth_to_corresponding_pointclouds_with_pixels(gt_abs, pred_abs, K, mask=pc_mask)
                else:
                    pts_gt, pts_pr = depth_to_corresponding_pointclouds(gt_abs, pred_abs, K, mask=pc_mask)
                l2 = pointwise_l2_stats(pts_gt, pts_pr)

                # For NN metrics, use full point clouds (same mask) and optional scale alignment
                pts_gt_nn = pts_gt
                pts_pr_nn = pts_pr
                scale = None
                if args.pc_align_scale and pts_pr_nn.shape[0] > 0 and pts_gt_nn.shape[0] > 0:
                    pts_pr_nn, scale = align_scale_to_reference(pts_pr_nn, pts_gt_nn)

                ac = accuracy_completion(pts_pr_nn, pts_gt_nn)

                # Optional: normal consistency (NC1/NC2) using depth-derived normals
                nc1 = nc2 = nc1_med = nc2_med = float('nan')
                if args.pc_eval_normals and pts_pr_nn.shape[0] > 0 and pts_gt_nn.shape[0] > 0:
                    nmap_gt = depth_to_normal_map(gt_abs, K, mask=pc_mask)
                    nmap_pr = depth_to_normal_map(pred_abs, K, mask=pc_mask)
                    n_gt = nmap_gt[ys, xs]
                    n_pr = nmap_pr[ys, xs]

                    d1, idx1 = nn_query_1nn(pts_pr_nn, pts_gt_nn)
                    d2, idx2 = nn_query_1nn(pts_gt_nn, pts_pr_nn)
                    nc1_stats = normal_consistency_from_normals(n_pr, n_gt, idx1)
                    nc2_stats = normal_consistency_from_normals(n_gt, n_pr, idx2)
                    nc1 = nc1_stats["mean"]
                    nc1_med = nc1_stats["median"]
                    nc2 = nc2_stats["mean"]
                    nc2_med = nc2_stats["median"]

                # NN-based metrics (subsample)
                rng = np.random.default_rng(0)
                pts_gt_s = pts_gt
                pts_pr_s = pts_pr
                if args.pc_max_points is not None and pts_gt.shape[0] > int(args.pc_max_points):
                    sel = rng.choice(pts_gt.shape[0], size=int(args.pc_max_points), replace=False)
                    pts_gt_s = pts_gt[sel]
                    pts_pr_s = pts_pr[sel]

                if args.pc_align_scale and pts_pr_s.shape[0] > 0 and pts_gt_s.shape[0] > 0:
                    pts_pr_s, _ = align_scale_to_reference(pts_pr_s, pts_gt_s)

                ch = chamfer_distance(pts_pr_s, pts_gt_s)
                fs = fscore(pts_pr_s, pts_gt_s, tau=args.pc_fscore_tau)

                per_frame_metrics[-1].update({
                    "pc_l2_mean": l2["mean"],
                    "pc_l2_median": l2["median"],
                    "pc_acc": ac["acc"],
                    "pc_comp": ac["comp"],
                    "pc_acc_median": ac["acc_median"],
                    "pc_comp_median": ac["comp_median"],
                    "pc_nc1": nc1,
                    "pc_nc2": nc2,
                    "pc_nc1_median": nc1_med,
                    "pc_nc2_median": nc2_med,
                    "pc_chamfer": ch,
                    "pc_fscore": fs,
                    "pc_points": l2["n"],
                    "pc_scale": (float(scale) if scale is not None else float('nan')),
                })

                if l2["n"] > 0 and np.isfinite(l2["mean"]):
                    sum_pc_l2_mean += l2["mean"]
                    sum_pc_l2_median += l2["median"]
                    sum_pc_acc += ac["acc"]
                    sum_pc_comp += ac["comp"]
                    sum_pc_acc_median += ac["acc_median"]
                    sum_pc_comp_median += ac["comp_median"]
                    if np.isfinite(nc1):
                        sum_pc_nc1 += float(nc1)
                        sum_pc_nc2 += float(nc2)
                        sum_pc_nc1_median += float(nc1_med)
                        sum_pc_nc2_median += float(nc2_med)
                    sum_pc_chamfer += ch
                    sum_pc_fscore += fs
                    counted_pc_frames += 1
            
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
    #create csv file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'abs_rel', 'delta1.25', 'delta1.03', 'valid_pixels']
        if args.eval_pointcloud:
            fieldnames += ['pc_l2_mean', 'pc_l2_median', 'pc_acc', 'pc_comp', 'pc_acc_median', 'pc_comp_median', 'pc_chamfer', 'pc_fscore', 'pc_points', 'pc_scale']
            # Keep NC fields in CSV even if not requested, to avoid field mismatch
            # when rows contain these keys (e.g., due to shared code paths).
            fieldnames += ['pc_nc1', 'pc_nc2', 'pc_nc1_median', 'pc_nc2_median']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
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

        if args.eval_pointcloud and counted_pc_frames > 0:
            avg_pc_l2_mean = sum_pc_l2_mean / counted_pc_frames
            avg_pc_l2_median = sum_pc_l2_median / counted_pc_frames
            avg_pc_acc = sum_pc_acc / counted_pc_frames
            avg_pc_comp = sum_pc_comp / counted_pc_frames
            avg_pc_acc_median = sum_pc_acc_median / counted_pc_frames
            avg_pc_comp_median = sum_pc_comp_median / counted_pc_frames
            avg_pc_chamfer = sum_pc_chamfer / counted_pc_frames
            avg_pc_fscore = sum_pc_fscore / counted_pc_frames
            print("\nPoint cloud metrics:")
            print(f"  pc_l2_mean:    {avg_pc_l2_mean:.6f} m")
            print(f"  pc_l2_median:  {avg_pc_l2_median:.6f} m")
            print(f"  pc_acc:        {avg_pc_acc:.6f} m")
            print(f"  pc_comp:       {avg_pc_comp:.6f} m")
            print(f"  pc_acc_median: {avg_pc_acc_median:.6f} m")
            print(f"  pc_comp_median:{avg_pc_comp_median:.6f} m")
            print(f"  pc_chamfer:    {avg_pc_chamfer:.6f} m")
            print(f"  pc_fscore@{args.pc_fscore_tau:.3f}m: {avg_pc_fscore:.6f}")

            if args.pc_eval_normals:
                avg_pc_nc1 = sum_pc_nc1 / counted_pc_frames
                avg_pc_nc2 = sum_pc_nc2 / counted_pc_frames
                avg_pc_nc1_median = sum_pc_nc1_median / counted_pc_frames
                avg_pc_nc2_median = sum_pc_nc2_median / counted_pc_frames
                print(f"  pc_nc1:        {avg_pc_nc1:.6f}")
                print(f"  pc_nc2:        {avg_pc_nc2:.6f}")
                print(f"  pc_nc1_median: {avg_pc_nc1_median:.6f}")
                print(f"  pc_nc2_median: {avg_pc_nc2_median:.6f}")
        
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
            f.write(f"Depth align: {args.depth_align}\n")
            if args.use_mask:
                f.write(f"Using masks: Yes\n")
            f.write("\n")
            
            f.write("Average Metrics:\n")
            f.write(f"  abs_rel:       {avg_abs_rel:.6f}\n")
            f.write(f"  delta <= 1.25: {avg_delta1_25:.6f}\n")
            f.write(f"  delta <= 1.03: {avg_delta1_03:.6f}\n\n")

            if args.eval_pointcloud and counted_pc_frames > 0:
                f.write("Point cloud Metrics:\n")
                f.write(f"  pc_l2_mean:    {avg_pc_l2_mean:.6f} m\n")
                f.write(f"  pc_l2_median:  {avg_pc_l2_median:.6f} m\n")
                f.write(f"  pc_acc:        {avg_pc_acc:.6f} m\n")
                f.write(f"  pc_comp:       {avg_pc_comp:.6f} m\n")
                f.write(f"  pc_acc_median: {avg_pc_acc_median:.6f} m\n")
                f.write(f"  pc_comp_median:{avg_pc_comp_median:.6f} m\n")
                f.write(f"  pc_chamfer:    {avg_pc_chamfer:.6f} m\n")
                f.write(f"  pc_fscore@{args.pc_fscore_tau:.3f}m: {avg_pc_fscore:.6f}\n\n")

                if args.pc_eval_normals:
                    f.write(f"  pc_nc1:        {avg_pc_nc1:.6f}\n")
                    f.write(f"  pc_nc2:        {avg_pc_nc2:.6f}\n")
                    f.write(f"  pc_nc1_median: {avg_pc_nc1_median:.6f}\n")
                    f.write(f"  pc_nc2_median: {avg_pc_nc2_median:.6f}\n\n")
            
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

python run_pt3d.py -c "saved/e2depth_evggs-debug-smooth-v2/checkpoint-epoch082-loss-0.0412.pth.tar" -i "/run/user/1000/gvfs/sftp:host=login.cvgl.lab,port=22332,user=sht/datasets/feed_forward_event/Tartanair_tmp/indoor" -o "./output/Tartanair_custom" --scene_name hospital_easy_P019 --use_mask --use_gpu --start_idx 1 --stop_idx 10 --eval_pointcloud --pc_align_scale --depth_align lad
'''