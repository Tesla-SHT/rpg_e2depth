#!/usr/bin/env python3
"""
批量运行深度估计评估脚本
对指定数据集下的所有场景进行评估，并统计平均指标
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import re
import csv

def find_scenes(data_root):
    """查找所有有效的场景目录"""
    scenes = []
    data_root = Path(data_root)
    
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        return scenes
    
    for scene_dir in data_root.iterdir():
        if scene_dir.is_dir():
            # 检查是否有序列1的数据
            seq1_dir = scene_dir / "1"
            if seq1_dir.exists() and seq1_dir.is_dir():
                scenes.append(scene_dir.name)
    
    return sorted(scenes)

def _extract_float(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_metrics_from_summary_file(summary_path: Path):
    """从每个场景生成的 metrics_summary.txt 中提取指标（更稳定）。"""
    if not summary_path.exists():
        return {}

    text = summary_path.read_text(errors='ignore')
    metrics = {}

    metrics['abs_rel'] = _extract_float(r"abs_rel:\s+([\d.eE+-]+)", text)
    metrics['delta1.25'] = _extract_float(r"delta <= 1\.25:\s+([\d.eE+-]+)", text)
    metrics['delta1.03'] = _extract_float(r"delta <= 1\.03:\s+([\d.eE+-]+)", text)

    # Point cloud metrics (optional)
    metrics['pc_l2_mean'] = _extract_float(r"pc_l2_mean:\s+([\d.eE+-]+)", text)
    metrics['pc_l2_median'] = _extract_float(r"pc_l2_median:\s+([\d.eE+-]+)", text)
    metrics['pc_acc'] = _extract_float(r"pc_acc:\s+([\d.eE+-]+)", text)
    metrics['pc_comp'] = _extract_float(r"pc_comp:\s+([\d.eE+-]+)", text)
    metrics['pc_acc_median'] = _extract_float(r"pc_acc_median:\s+([\d.eE+-]+)", text)
    metrics['pc_comp_median'] = _extract_float(r"pc_comp_median:\s+([\d.eE+-]+)", text)
    metrics['pc_nc1'] = _extract_float(r"pc_nc1:\s+([\d.eE+-]+)", text)
    metrics['pc_nc2'] = _extract_float(r"pc_nc2:\s+([\d.eE+-]+)", text)
    metrics['pc_nc1_median'] = _extract_float(r"pc_nc1_median:\s+([\d.eE+-]+)", text)
    metrics['pc_nc2_median'] = _extract_float(r"pc_nc2_median:\s+([\d.eE+-]+)", text)
    metrics['pc_chamfer'] = _extract_float(r"pc_chamfer:\s+([\d.eE+-]+)", text)
    # pc_fscore@0.100m: 0.123456
    metrics['pc_fscore'] = _extract_float(r"pc_fscore@[^:]+:\s+([\d.eE+-]+)", text)

    # Remove None entries
    return {k: v for k, v in metrics.items() if v is not None}

def run_evaluation(model_path, data_root, output_folder, scene_name, 
                   start_idx, stop_idx, eval_clip_distance, reg_factor,
                   save_pred_png, save_gt_png, use_gpu, use_mask,
                   depth_align='none',
                   eval_pointcloud=False, pc_align_scale=False, pc_eval_normals=False,
                   pc_max_points=20000, pc_fscore_tau=0.10):
    """运行单个场景的评估"""
    cmd = [
        'python', 'run_pt3d.py',
        '-c', model_path,
        '-i', data_root,
        '-o', output_folder,
        '--scene_name', scene_name,
        '--start_idx', str(start_idx),
        '--eval_clip_distance', str(eval_clip_distance),
        '--reg_factor', str(reg_factor),
    ]

    if depth_align is not None:
        cmd.extend(['--depth_align', str(depth_align)])
    
    if stop_idx is not None:
        cmd.extend(['--stop_idx', str(stop_idx)])
    
    if save_pred_png:
        cmd.append('--save_pred_png')
    
    if save_gt_png:
        cmd.append('--save_gt_png')
    
    if use_gpu:
        cmd.append('--use_gpu')
    if use_mask:
        cmd.append('--use_mask')

    if eval_pointcloud:
        cmd.append('--eval_pointcloud')
        cmd.extend(['--pc_max_points', str(pc_max_points)])
        cmd.extend(['--pc_fscore_tau', str(pc_fscore_tau)])
        if pc_align_scale:
            cmd.append('--pc_align_scale')
        if pc_eval_normals:
            cmd.append('--pc_eval_normals')
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0
                # 写日志到每个场景目录，方便排查
        scene_out_dir = Path(output_folder) / scene_name
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        (scene_out_dir / "batch_run.log").write_text(output, errors="ignore")

        return success, output
    
    except subprocess.TimeoutExpired:
        return False, "Timeout: Process took longer than 1 hour"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Batch evaluation on multiple scenes')
    
    # 必需参数
    parser.add_argument('-c', '--model_path', required=True, type=str,
                        help='Path to model checkpoint')
    parser.add_argument('-i', '--data_root', required=True, type=str,
                        help='Root directory containing scene folders')
    parser.add_argument('-o', '--output_folder', required=True, type=str,
                        help='Output folder for results')
    
    # 可选参数
    parser.add_argument('--start_idx', default=1, type=int,
                        help='Start frame index')
    parser.add_argument('--frames_per_scene', default=100, type=int,
                        help='Number of frames to evaluate per scene (default: 100)')
    parser.add_argument('--stop_idx', default=None, type=int,
                        help='Optional explicit stop frame index (overrides frames_per_scene)')
    parser.add_argument('--eval_clip_distance', default=80.0, type=float,
                        help='Maximum depth value for evaluation (meters)')
    parser.add_argument('--reg_factor', default=3.70378, type=float,
                        help='Regularization factor for depth conversion')
    parser.add_argument('--save_pred_png', action='store_true',
                        help='Save predicted depth as PNG')
    parser.add_argument('--save_gt_png', action='store_true',
                        help='Save ground truth depth as PNG')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--no_confirm', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--use_mask', action='store_true',
                        help='Use valid depth mask during evaluation')

    # Depth alignment pass-through
    parser.add_argument('--depth_align', default='none', type=str,
                        choices=['none', 'lstsq', 'median_scale', 'lad'],
                        help='Align predicted depth to GT before metrics (scale/shift)')

    # Point cloud evaluation pass-through
    parser.add_argument('--eval_pointcloud', action='store_true',
                        help='Also evaluate point cloud metrics per scene')
    parser.add_argument('--pc_align_scale', action='store_true',
                        help='Align predicted point cloud scale to GT')
    parser.add_argument('--pc_eval_normals', action='store_true',
                        help='Also compute normal consistency metrics (NC1/NC2)')
    parser.add_argument('--pc_max_points', type=int, default=20000)
    parser.add_argument('--pc_fscore_tau', type=float, default=0.10)

    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip scenes that already have output folder')
    args = parser.parse_args()
    
    # 查找所有场景
    print("=== Scanning for scenes ===")
    scenes = find_scenes(args.data_root)
    
    if not scenes:
        print(f"Error: No valid scenes found in {args.data_root}")
        return 1
    
    print(f"Found {len(scenes)} scenes:")
    for scene in scenes:
        print(f"  - {scene}")
    print()
    
    # 确认是否继续
    if not args.no_confirm:
        response = input(f"Process all {len(scenes)} scenes? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return 0
    
    # 创建输出目录
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 初始化统计变量
    success_count = 0
    failed_count = 0
    failed_scenes = []
    all_metrics = []
    
    # 创建汇总文件
    summary_file = output_folder / 'evaluation_summary.txt'
    summary_csv = output_folder / 'evaluation_summary.csv'

    csv_fieldnames = [
        'scene', 'status',
        'abs_rel', 'delta1.25', 'delta1.03',
        'pc_l2_mean', 'pc_l2_median', 'pc_acc', 'pc_comp', 'pc_acc_median', 'pc_comp_median',
        'pc_nc1', 'pc_nc2', 'pc_nc1_median', 'pc_nc2_median',
        'pc_chamfer', 'pc_fscore',
    ]

    with open(summary_csv, 'w', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
        writer.writeheader()

    with open(summary_file, 'w') as f:
        f.write("=== Evaluation Summary ===\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Eval Clip Distance: {args.eval_clip_distance}\n")
        f.write(f"Reg Factor: {args.reg_factor}\n")
        f.write(f"Frames per scene: {args.frames_per_scene}\n")
        f.write(f"Depth align: {args.depth_align}\n")
        f.write(f"Eval pointcloud: {args.eval_pointcloud}\n")
        f.write(f"PC eval normals: {args.pc_eval_normals}\n")
        f.write("\n")
        f.write(f"{'Scene':<30} | {'abs_rel':>7} | {'d1.25':>7} | {'d1.03':>7} | {'pc_acc':>7} | {'pc_comp':>7} | Status\n")
        f.write("-" * 80 + "\n")
    
    # 遍历所有场景
    for i, scene in enumerate(scenes, 1):
        if ("indoor" in scene.lower()):
            continue
        scene_out_dir = output_folder / scene
        scene_summary_path = scene_out_dir / 'metrics_summary.txt'

        # 如果已经产生了，那么就跳过这个场景（可选）
        if args.skip_existing and scene_out_dir.exists():
            print(f"\n=== Skipping already processed scene: {scene} ===")
            metrics = extract_metrics_from_summary_file(scene_summary_path)
            with open(summary_file, 'a') as f:
                f.write(
                    f"{scene:<30} | {metrics.get('abs_rel', float('nan')):7.6f} | "
                    f"{metrics.get('delta1.25', float('nan')):7.6f} | "
                    f"{metrics.get('delta1.03', float('nan')):7.6f} | "
                    f"{metrics.get('pc_acc', float('nan')):7.6f} | "
                    f"{metrics.get('pc_comp', float('nan')):7.6f} | OK (skipped)\n"
                )
            with open(summary_csv, 'a', newline='') as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
                row = {"scene": scene, "status": "OK_SKIPPED"}
                row.update(metrics)
                writer.writerow(row)
            success_count += 1
            continue
        print("\n" + "=" * 50)
        print(f"[{i}/{len(scenes)}] Processing scene: {scene}")
        print("=" * 50)
        
        if args.stop_idx is not None:
            stop_idx = args.stop_idx
        else:
            stop_idx = args.start_idx + int(args.frames_per_scene)

        success, output = run_evaluation(
            model_path=args.model_path,
            data_root=args.data_root,
            output_folder=args.output_folder,
            scene_name=scene,
            start_idx=args.start_idx,
            stop_idx=stop_idx,
            eval_clip_distance=args.eval_clip_distance,
            reg_factor=args.reg_factor,
            save_pred_png=args.save_pred_png,
            save_gt_png=args.save_gt_png,
            use_gpu=args.use_gpu,
            use_mask=args.use_mask,
            depth_align=args.depth_align,
            eval_pointcloud=args.eval_pointcloud,
            pc_align_scale=args.pc_align_scale,
            pc_eval_normals=args.pc_eval_normals,
            pc_max_points=args.pc_max_points,
            pc_fscore_tau=args.pc_fscore_tau,
        )
        
        if success:
            print(f"✓ Successfully processed {scene}")
            success_count += 1

            # 提取指标：优先从场景 summary 文件
            metrics = extract_metrics_from_summary_file(scene_summary_path)
            if metrics:
                all_metrics.append(metrics)

            with open(summary_file, 'a') as f:
                f.write(
                    f"{scene:<30} | {metrics.get('abs_rel', float('nan')):7.6f} | "
                    f"{metrics.get('delta1.25', float('nan')):7.6f} | "
                    f"{metrics.get('delta1.03', float('nan')):7.6f} | "
                    f"{metrics.get('pc_acc', float('nan')):7.6f} | "
                    f"{metrics.get('pc_comp', float('nan')):7.6f} | OK\n"
                )

            with open(summary_csv, 'a', newline='') as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
                row = {"scene": scene, "status": "OK"}
                row.update(metrics)
                writer.writerow(row)

            # 简要显示
            if 'abs_rel' in metrics:
                print(f"  abs_rel:       {metrics.get('abs_rel')}")
            if 'delta1.25' in metrics:
                print(f"  delta <= 1.25: {metrics.get('delta1.25')}")
            if args.eval_pointcloud and 'pc_acc' in metrics:
                print(f"  pc_acc:        {metrics.get('pc_acc')} m")
        else:
            print(f"✗ Failed to process {scene}")
            failed_count += 1
            failed_scenes.append(scene)
            
            # 写入汇总文件
            with open(summary_file, 'a') as f:
                f.write(f"{scene:<30} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | FAILED\n")

            with open(summary_csv, 'a', newline='') as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
                writer.writerow({"scene": scene, "status": "FAILED"})
    
    # 计算平均指标
    avg_metrics = {}
    if all_metrics:
        def _avg(key):
            vals = [m.get(key) for m in all_metrics if m.get(key) is not None]
            vals = [v for v in vals if isinstance(v, (int, float))]
            return (sum(vals) / len(vals)) if vals else None

        avg_metrics['abs_rel'] = _avg('abs_rel')
        avg_metrics['delta1.25'] = _avg('delta1.25')
        avg_metrics['delta1.03'] = _avg('delta1.03')
        if args.eval_pointcloud:
            avg_metrics['pc_acc'] = _avg('pc_acc')
            avg_metrics['pc_comp'] = _avg('pc_comp')
            avg_metrics['pc_fscore'] = _avg('pc_fscore')
    
    # 写入总结
    summary_text = [
        "\n" + "=" * 50,
        "Processing Summary",
        "=" * 50,
        f"Total scenes: {len(scenes)}",
        f"Successfully processed: {success_count}",
        f"Failed: {failed_count}",
    ]
    
    if avg_metrics:
        summary_text.extend([
            "",
            f"Average Metrics (across {len(all_metrics)} scenes):",
            f"  abs_rel:       {avg_metrics['abs_rel']:.6f}" if avg_metrics.get('abs_rel') is not None else "  abs_rel:       N/A",
            f"  delta <= 1.25: {avg_metrics['delta1.25']:.6f}" if avg_metrics.get('delta1.25') is not None else "  delta <= 1.25: N/A",
            f"  delta <= 1.03: {avg_metrics['delta1.03']:.6f}" if avg_metrics.get('delta1.03') is not None else "  delta <= 1.03: N/A",
        ])
        if args.eval_pointcloud:
            summary_text.extend([
                f"  pc_acc:        {avg_metrics['pc_acc']:.6f} m" if avg_metrics.get('pc_acc') is not None else "  pc_acc:        N/A",
                f"  pc_comp:       {avg_metrics['pc_comp']:.6f} m" if avg_metrics.get('pc_comp') is not None else "  pc_comp:       N/A",
                f"  pc_fscore:     {avg_metrics['pc_fscore']:.6f}" if avg_metrics.get('pc_fscore') is not None else "  pc_fscore:     N/A",
            ])
    
    if failed_scenes:
        summary_text.extend([
            "",
            "Failed scenes:",
        ])
        summary_text.extend([f"  - {scene}" for scene in failed_scenes])
    
    summary_text.extend([
        "",
        f"Results saved to: {output_folder}",
        f"Summary file: {summary_file}",
        "Each scene has:",
    ])
    
    if args.save_pred_png:
        summary_text.append("  - pred_png/: predicted depth visualizations (.png)")
    if args.save_gt_png:
        summary_text.append("  - gt_png/: ground truth depth visualizations (.png)")
    
    # 打印并写入文件
    for line in summary_text:
        print(line)
        with open(summary_file, 'a') as f:
            f.write(line + "\n")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())


'''

python run_depth_evggs_batch.py -c "pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/replica_event_frame_processed_check/" -o "./output/Ev3D" --save_pred_png --save_gt_png

python run_depth_evggs_batch.py -c "pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/MVSEC_all" -o "./output/MVSEC" --save_pred_png --save_gt_png --start_idx 150 --stop_idx 250

python run_depth_evggs_batch.py -c "saved/e2depth_evggs-debug-smooth-v2/checkpoint-epoch082-loss-0.0412.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor" -o "./output/Tartanair_new" --save_pred_png --save_gt_png

python run_pt3d_batch.py -c "pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar" -i "/run/determined/workdir/data/feed_forward_event/MVSEC_all" -o "./output/MVSEC_depth_align_lad_1500" --start_idx 500 --frames_per_scene 1500 --use_mask --use_gpu --pc_align_scale --skip_existing --depth_align lad --pc_eval_normals

python run_pt3d_batch.py -c "saved/e2depth_evggs-debug-smooth-v2/checkpoint-epoch082-loss-0.0412.pth.tar" -i "/run/user/1000/gvfs/sftp:host=login.cvgl.lab,port=22332,user=sht/datasets/feed_forward_event/Tartanair_tmp/indoor" -o "./output/Tartanair_pt3d_batch" --start_idx 1 --frames_per_scene 100 --use_mask --use_gpu --eval_pointcloud --pc_align_scale --skip_existing --depth_align median_scale --pc_eval_normals
'''