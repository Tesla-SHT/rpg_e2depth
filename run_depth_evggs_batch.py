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

def extract_metrics(output_text):
    """从输出文本中提取评估指标"""
    metrics = {}
    
    # 匹配 abs_rel
    match = re.search(r'abs_rel:\s+([\d.]+)', output_text)
    if match:
        metrics['abs_rel'] = float(match.group(1))
    
    # 匹配 delta <= 1.25
    match = re.search(r'delta <= 1\.25:\s+([\d.]+)', output_text)
    if match:
        metrics['delta1.25'] = float(match.group(1))
    
    # 匹配 delta <= 1.03
    match = re.search(r'delta <= 1\.03:\s+([\d.]+)', output_text)
    if match:
        metrics['delta1.03'] = float(match.group(1))
    
    return metrics

def run_evaluation(model_path, data_root, output_folder, scene_name, 
                   start_idx, stop_idx, eval_clip_distance, reg_factor,
                   save_pred_png, save_gt_png, use_gpu, use_mask):
    """运行单个场景的评估"""
    cmd = [
        'python', 'run_depth_evggs.py',
        '-c', model_path,
        '-i', data_root,
        '-o', output_folder,
        '--scene_name', scene_name,
        '--start_idx', str(start_idx),
        '--eval_clip_distance', str(eval_clip_distance),
        '--reg_factor', str(reg_factor),
    ]
    
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
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0
        
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
    parser.add_argument('--stop_idx', default=200, type=int,
                        help='Stop frame index (None for all frames)')
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
    with open(summary_file, 'w') as f:
        f.write("=== Evaluation Summary ===\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Eval Clip Distance: {args.eval_clip_distance}\n")
        f.write(f"Reg Factor: {args.reg_factor}\n")
        f.write("\n")
        f.write(f"{'Scene':<30} | {'abs_rel':>7} | {'delta1.25':>9} | {'delta1.03':>9} | Status\n")
        f.write("-" * 80 + "\n")
    
    # 遍历所有场景
    for i, scene in enumerate(scenes, 1):
        #如果已经产生了，那么就跳过这个场景
        if (output_folder / scene).exists():
            print(f"\n=== Skipping already processed scene: {scene} ===")
            success_count += 1
            continue
        print("\n" + "=" * 50)
        print(f"[{i}/{len(scenes)}] Processing scene: {scene}")
        print("=" * 50)
        
        success, output = run_evaluation(
            model_path=args.model_path,
            data_root=args.data_root,
            output_folder=args.output_folder,
            scene_name=scene,
            start_idx=args.start_idx,
            stop_idx=args.stop_idx,
            eval_clip_distance=args.eval_clip_distance,
            reg_factor=args.reg_factor,
            save_pred_png=args.save_pred_png,
            save_gt_png=args.save_gt_png,
            use_gpu=args.use_gpu,
            use_mask=args.use_mask
        )
        
        if success:
            print(f"✓ Successfully processed {scene}")
            success_count += 1
            
            # 提取指标
            metrics = extract_metrics(output)
            
            if metrics:
                all_metrics.append(metrics)
                
                # 写入汇总文件
                with open(summary_file, 'a') as f:
                    f.write(f"{scene:<30} | {metrics.get('abs_rel', 0):7.6f} | "
                           f"{metrics.get('delta1.25', 0):9.6f} | "
                           f"{metrics.get('delta1.03', 0):9.6f} | OK\n")
                
                # 显示结果
                print(f"  abs_rel:       {metrics.get('abs_rel', 'N/A')}")
                print(f"  delta <= 1.25: {metrics.get('delta1.25', 'N/A')}")
                print(f"  delta <= 1.03: {metrics.get('delta1.03', 'N/A')}")
            else:
                with open(summary_file, 'a') as f:
                    f.write(f"{scene:<30} | {'N/A':>7} | {'N/A':>9} | {'N/A':>9} | OK (no metrics)\n")
        else:
            print(f"✗ Failed to process {scene}")
            failed_count += 1
            failed_scenes.append(scene)
            
            # 写入汇总文件
            with open(summary_file, 'a') as f:
                f.write(f"{scene:<30} | {'N/A':>7} | {'N/A':>9} | {'N/A':>9} | FAILED\n")
    
    # 计算平均指标
    avg_metrics = {}
    if all_metrics:
        avg_metrics['abs_rel'] = sum(m.get('abs_rel', 0) for m in all_metrics) / len(all_metrics)
        avg_metrics['delta1.25'] = sum(m.get('delta1.25', 0) for m in all_metrics) / len(all_metrics)
        avg_metrics['delta1.03'] = sum(m.get('delta1.03', 0) for m in all_metrics) / len(all_metrics)
    
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
            f"  abs_rel:       {avg_metrics['abs_rel']:.6f}",
            f"  delta <= 1.25: {avg_metrics['delta1.25']:.6f}",
            f"  delta <= 1.03: {avg_metrics['delta1.03']:.6f}",
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
'''